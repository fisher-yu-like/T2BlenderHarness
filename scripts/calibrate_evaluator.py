"""Calibrate evaluator channels against an independently scored golden set."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "prompt_compliance",
    "physical_plausibility",
    "camera_coverage",
    "camera_innovation",
    "character_trajectory",
    "object_trajectory",
    "event_timing",
    "temporal_smoothness",
    "visual_clarity",
    "appearance_detail",
    "physical_realism",
    "spatial_consistency",
    "motion_naturalness",
    "visual_presentation",
)
SEMANTIC_DIMENSIONS = (
    "prompt_compliance",
    "physical_plausibility",
    "object_trajectory",
    "event_timing",
    "character_trajectory",
)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = average
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    left_scale = sum((a - mean_left) ** 2 for a in left) ** 0.5
    right_scale = sum((b - mean_right) ** 2 for b in right) ** 0.5
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def _bootstrap_ci(left: list[float], right: list[float], *, seed: int, iterations: int) -> tuple[float, float]:
    if len(left) < 2 or iterations <= 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        indices = [rng.randrange(len(left)) for _ in left]
        sample_left = [left[index] for index in indices]
        sample_right = [right[index] for index in indices]
        values.append(_spearman(sample_left, sample_right))
    values.sort()
    lower_index = max(0, min(len(values) - 1, int(0.025 * len(values))))
    upper_index = max(0, min(len(values) - 1, int(0.975 * len(values)) - 1))
    return (round(values[lower_index], 4), round(values[upper_index], 4))


def _metric(actual: list[float], predicted: list[float], *, seed: int, iterations: int) -> dict[str, Any]:
    spearman = round(_spearman(actual, predicted), 4)
    pearson = round(_pearson(actual, predicted), 4)
    return {
        "n": len(actual),
        "spearman": spearman,
        "pearson": pearson,
        "spearman_bootstrap_95_ci": list(_bootstrap_ci(actual, predicted, seed=seed, iterations=iterations)),
        "status": "uninformative" if abs(spearman) < 0.3 else "informative",
    }


def _values(records: list[dict[str, Any]], section: str, field: str) -> list[float]:
    values = []
    for record in records:
        value = (record.get(section) or {}).get(field)
        if value is None:
            raise ValueError(f"record {record.get('case_id')} missing {section}.{field}")
        values.append(float(value))
    return values


def calibrate_records(
    records: list[dict[str, Any]], *, seed: int = 20260827, bootstrap_iterations: int = 500
) -> dict[str, Any]:
    if not records:
        raise ValueError("calibration requires at least one aligned record")
    sources = ("vlm", "frame_statistics", "deterministic")
    dimensions: dict[str, dict[str, Any]] = {}
    for dimension_index, dimension in enumerate(DIMENSIONS):
        actual = [float(record["human"]["dimensions"][dimension]) for record in records]
        dimensions[dimension] = {}
        for source_index, source in enumerate(sources):
            predicted = [float((record[source]["dimensions"])[dimension]) for record in records]
            dimensions[dimension][source] = _metric(
                actual,
                predicted,
                seed=seed + dimension_index * 100 + source_index,
                iterations=bootstrap_iterations,
            )
    human_task = _values(records, "human", "task_vlm")
    human_realism = _values(records, "human", "realism_final")
    vlm_task = _values(records, "vlm", "task_vlm")
    vlm_realism = _values(records, "vlm", "realism_final")
    semantic_status = all(dimensions[dimension]["frame_statistics"]["status"] == "uninformative" for dimension in SEMANTIC_DIMENSIONS)
    overall = {
        "vlm_task_vlm_spearman": _spearman(human_task, vlm_task) >= 0.6,
        "vlm_task_vlm_spearman_value": round(_spearman(human_task, vlm_task), 4),
        "vlm_realism_spearman": _spearman(human_realism, vlm_realism) >= 0.6,
        "vlm_realism_spearman_value": round(_spearman(human_realism, vlm_realism), 4),
        "frame_statistics_semantics_uninformative": semantic_status,
    }
    overall["phase3_admission"] = bool(
        overall["vlm_task_vlm_spearman"]
        and overall["vlm_realism_spearman"]
        and overall["frame_statistics_semantics_uninformative"]
    )
    return {
        "evaluator_version": "evaluator-v5-calibration",
        "case_count": len(records),
        "dimensions": dimensions,
        "overall_gates": overall,
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "sources": list(sources),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="JSON array of aligned human and evaluator records")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-iterations", type=int, default=500)
    args = parser.parse_args()
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    report = calibrate_records(records, seed=args.seed, bootstrap_iterations=args.bootstrap_iterations)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
