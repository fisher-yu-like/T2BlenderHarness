"""Finalize a complete two-annotator blind review bundle and compute ICC(2,1)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_golden_review_set import GOLDEN_DIMENSIONS, validate_golden_review_set


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def _icc_2_1(matrix: list[list[float]]) -> float:
    """Calculate the two-way random, single-measure absolute-agreement ICC."""

    n = len(matrix)
    k = len(matrix[0]) if matrix else 0
    if n < 2 or k < 2:
        raise ValueError("ICC(2,1) requires at least two samples and two annotators")
    grand_mean = sum(sum(row) for row in matrix) / (n * k)
    row_means = [sum(row) / k for row in matrix]
    column_means = [sum(matrix[row][column] for row in range(n)) / n for column in range(k)]
    ms_rows = k * sum((mean - grand_mean) ** 2 for mean in row_means) / (n - 1)
    ms_columns = n * sum((mean - grand_mean) ** 2 for mean in column_means) / (k - 1)
    residual = sum(
        (matrix[row][column] - row_means[row] - column_means[column] + grand_mean) ** 2
        for row in range(n)
        for column in range(k)
    )
    ms_error = residual / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    if abs(denominator) < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, (ms_rows - ms_error) / denominator))


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _agreement_for_bundle(bundle: Path) -> tuple[dict[str, Any], list[str]]:
    manifest = _read_jsonl(bundle / "manifest.jsonl")
    score_rows = _read_jsonl(bundle / "human_scores.jsonl")
    if not 30 <= len(manifest) <= 50:
        raise ValueError(f"golden review requires 30-50 cases, got {len(manifest)}")
    expected: list[tuple[str, str]] = []
    for row in manifest:
        case_id = str(row.get("case_id") or "")
        frames = row.get("sampled_frames")
        videos = row.get("sampled_videos")
        if not case_id or not isinstance(frames, dict) or not isinstance(videos, dict):
            raise ValueError(f"manifest case {case_id} has invalid sample media")
        expected.extend((case_id, str(sample_id)) for sample_id in set(frames) & set(videos))
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in score_rows:
        key = (str(row.get("case_id") or ""), str(row.get("sample_id") or ""))
        annotator = str(row.get("annotator_id") or "").strip()
        scores = row.get("scores")
        if key not in expected or not annotator or not isinstance(scores, dict):
            raise ValueError(f"invalid score row for {key[0]}/{key[1]}")
        if key in by_key and annotator in by_key[key]:
            raise ValueError(f"duplicate score row for {key[0]}/{key[1]}/{annotator}")
        if set(scores) != set(GOLDEN_DIMENSIONS):
            raise ValueError(f"score row {key[0]}/{key[1]} must contain all 14 dimensions")
        for dimension in GOLDEN_DIMENSIONS:
            value = scores[dimension]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 100:
                raise ValueError(f"score row {key[0]}/{key[1]} has invalid {dimension}")
        by_key.setdefault(key, {})[annotator] = scores
    missing = [key for key in expected if len(by_key.get(key, {})) < 2]
    if missing:
        raise ValueError(f"each blind video needs two annotators; missing coverage for {missing[:5]}")
    annotators = sorted({annotator for values in by_key.values() for annotator in values})
    if len(annotators) < 2:
        raise ValueError("golden review requires two independent annotators")
    agreement: dict[str, Any] = {}
    for dimension in GOLDEN_DIMENSIONS:
        matrix = []
        for key in expected:
            values = by_key[key]
            chosen = sorted(values)[:2]
            matrix.append([float(values[annotator][dimension]) for annotator in chosen])
        agreement[dimension] = {
            "metric": "icc_2_1",
            "value": round(_icc_2_1(matrix), 6),
            "sample_count": len(matrix),
            "annotator_count": 2,
        }
    return agreement, annotators


def finalize_golden_review(root: str | Path) -> dict[str, Any]:
    bundle = Path(root).resolve()
    metadata_path = bundle / "metadata.json"
    try:
        old_text = metadata_path.read_text(encoding="utf-8")
        metadata = json.loads(old_text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"metadata is unreadable: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    agreement, annotators = _agreement_for_bundle(bundle)
    updated = {
        **metadata,
        "status": "annotations_finalized",
        "annotations_complete": True,
        "annotators": annotators,
        "inter_rater_agreement": agreement,
        "agreement_method": "icc_2_1_first_two_sorted_annotators",
        "finalized_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_json(metadata_path, updated)
    try:
        report = validate_golden_review_set(bundle)
        if report["status"] != "pass":
            raise ValueError(f"finalized bundle failed validation: {report.get('errors')}")
    except Exception:
        metadata_path.write_text(old_text, encoding="utf-8")
        raise
    return {**report, "finalized": True, "agreement": agreement, "annotators": annotators}


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize two-annotator golden review and compute ICC(2,1)")
    parser.add_argument("root_positional", nargs="?")
    parser.add_argument("--root", dest="root_option")
    args = parser.parse_args()
    root = args.root_option or args.root_positional or "dataset/golden-review-exact-v2"
    try:
        report = finalize_golden_review(root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
