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
REQUIRED_EVENT_F1_THRESHOLD = 0.85
MAX_DIMENSION_MAE_THRESHOLD = 10.0
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
    mae = round(sum(abs(left - right) for left, right in zip(actual, predicted)) / len(actual), 4) if actual else None
    return {
        "n": len(actual),
        "spearman": spearman,
        "pearson": pearson,
        "mae": mae,
        "spearman_bootstrap_95_ci": list(_bootstrap_ci(actual, predicted, seed=seed, iterations=iterations)),
        "status": "uninformative" if abs(spearman) < 0.3 else "informative",
    }


def _event_labels(section: Any) -> dict[str, bool] | None:
    if not isinstance(section, dict):
        return None
    for key in ("required_event_labels", "required_event_success"):
        value = section.get(key)
        if isinstance(value, dict) and value:
            if all(isinstance(item, bool) for item in value.values()):
                return {str(event_id): bool(item) for event_id, item in value.items()}
            return None
    for key in ("required_event_scores", "event_scores"):
        value = section.get(key)
        if isinstance(value, dict) and value:
            labels: dict[str, bool] = {}
            for event_id, score in value.items():
                if score is None or isinstance(score, bool):
                    return None
                try:
                    labels[str(event_id)] = float(score) >= 25.0
                except (TypeError, ValueError):
                    return None
            return labels
    return None


def _event_f1(records: list[dict[str, Any]]) -> tuple[float | None, str]:
    actual: list[bool] = []
    predicted: list[bool] = []
    for record in records:
        human = _event_labels(record.get("human"))
        vlm = _event_labels(record.get("vlm"))
        if not human or not vlm:
            return None, "unavailable"
        common = sorted(set(human) & set(vlm))
        if not common or set(human) != set(vlm):
            return None, "unavailable"
        actual.extend(human[event_id] for event_id in common)
        predicted.extend(vlm[event_id] for event_id in common)
    if not actual:
        return None, "unavailable"
    true_positive = sum(left and right for left, right in zip(actual, predicted))
    false_positive = sum((not left) and right for left, right in zip(actual, predicted))
    false_negative = sum(left and (not right) for left, right in zip(actual, predicted))
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator if denominator else 0.0), "scored"


def _confidence_reliability(records: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = ((0.0, 0.6), (0.6, 0.8), (0.8, 1.0))
    values: dict[str, list[tuple[float, float]]] = {f"{low:.1f}-{high:.1f}": [] for low, high in buckets}
    for record in records:
        vlm = record.get("vlm") or {}
        human = record.get("human") or {}
        confidence = vlm.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            continue
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            continue
        try:
            task_error = abs(float(vlm["task_vlm"]) - float(human["task_vlm"]))
            realism_error = abs(float(vlm["realism_final"]) - float(human["realism_final"]))
        except (KeyError, TypeError, ValueError):
            continue
        for low, high in buckets:
            if low <= confidence <= high and (confidence < high or high == 1.0):
                values[f"{low:.1f}-{high:.1f}"].append((task_error, realism_error))
                break
    return {
        "version": "confidence-reliability-v1",
        "status": "scored" if any(values.values()) else "unavailable",
        "buckets": {
            name: {
                "count": len(items),
                "mean_absolute_task_error": round(sum(item[0] for item in items) / len(items), 4) if items else None,
                "mean_absolute_realism_error": round(sum(item[1] for item in items) / len(items), 4) if items else None,
            }
            for name, items in values.items()
        },
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
    event_f1, event_f1_status = _event_f1(records)
    confidence_reliability = _confidence_reliability(records)
    vlm_mae_values = [
        dimensions[dimension]["vlm"]["mae"]
        for dimension in DIMENSIONS
        if dimensions[dimension]["vlm"]["mae"] is not None
    ]
    max_vlm_mae = max(vlm_mae_values) if vlm_mae_values else None
    overall = {
        "vlm_task_vlm_spearman": _spearman(human_task, vlm_task) >= 0.6,
        "vlm_task_vlm_spearman_value": round(_spearman(human_task, vlm_task), 4),
        "vlm_realism_spearman": _spearman(human_realism, vlm_realism) >= 0.6,
        "vlm_realism_spearman_value": round(_spearman(human_realism, vlm_realism), 4),
        "frame_statistics_semantics_uninformative": semantic_status,
        "required_event_f1_status": event_f1_status,
        "required_event_f1_value": None if event_f1 is None else round(event_f1, 4),
        "required_event_f1": bool(event_f1 is not None and event_f1 >= REQUIRED_EVENT_F1_THRESHOLD),
        "vlm_max_dimension_mae": None if max_vlm_mae is None else round(max_vlm_mae, 4),
        "vlm_dimension_mae": bool(max_vlm_mae is not None and max_vlm_mae <= MAX_DIMENSION_MAE_THRESHOLD),
    }
    overall["phase3_admission"] = bool(
        overall["vlm_task_vlm_spearman"]
        and overall["vlm_realism_spearman"]
        and overall["frame_statistics_semantics_uninformative"]
        and overall["required_event_f1"]
        and overall["vlm_dimension_mae"]
    )
    return {
        "evaluator_version": "evaluator-v5-calibration",
        "case_count": len(records),
        "dimensions": dimensions,
        "overall_gates": overall,
        "seed": seed,
        "bootstrap_iterations": bootstrap_iterations,
        "sources": list(sources),
        "confidence_reliability": confidence_reliability,
        "calibration_thresholds": {
            "required_event_f1": REQUIRED_EVENT_F1_THRESHOLD,
            "max_dimension_mae": MAX_DIMENSION_MAE_THRESHOLD,
        },
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
