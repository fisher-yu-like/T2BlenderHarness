"""Export gated plan and trajectory preference pairs as separate JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_preference_pairs(
    output_dir: str | Path,
    *,
    calibration_report: dict[str, Any],
    acceptance_record: dict[str, Any],
    reproducibility_report: dict[str, Any],
    plan_pairs: list[dict[str, Any]],
    trajectory_pairs: list[dict[str, Any]],
) -> dict[str, int]:
    if calibration_report.get("status") != "ready":
        raise ValueError("calibration must be ready before preference export")
    if acceptance_record.get("accepted") is not True:
        raise ValueError("acceptance of a Harness candidate is required before preference export")
    if reproducibility_report.get("reproducible") is not True:
        raise ValueError("test reproducibility report is required before preference export")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "plan_preference_pairs.jsonl", plan_pairs)
    _write_jsonl(destination / "trajectory_preference_pairs.jsonl", trajectory_pairs)
    return {"plan_pairs": len(plan_pairs), "trajectory_pairs": len(trajectory_pairs)}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
