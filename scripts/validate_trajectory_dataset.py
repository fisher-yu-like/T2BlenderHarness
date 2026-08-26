"""Validate the complex trajectory-v2 dataset and planner-derived metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.scene_contract import SceneContractBuilder
from videoact.trajectory import TrajectoryPlanner


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(root: str | Path = "dataset/trajectory-v2") -> dict[str, Any]:
    destination = Path(root)
    manifest = _records(destination / "manifest.jsonl")
    splits = json.loads((destination / "splits.json").read_text(encoding="utf-8"))
    labels = _records(destination / "labels.jsonl")
    errors: list[str] = []
    case_ids = [record.get("case_id") for record in manifest]
    if len(manifest) != 80:
        errors.append(f"expected 80 cases, got {len(manifest)}")
    if len(case_ids) != len(set(case_ids)):
        errors.append("case IDs must be unique")
    expected_counts = {"train": 50, "dev": 20, "test": 10}
    all_split_ids: list[str] = []
    for split, expected_count in expected_counts.items():
        values = splits.get(split, [])
        if len(values) != expected_count:
            errors.append(f"{split} must contain {expected_count} cases")
        all_split_ids.extend(values)
    if len(all_split_ids) != len(set(all_split_ids)) or set(all_split_ids) != set(case_ids):
        errors.append("splits must be disjoint and cover every case")

    prompt_hashes: set[str] = set()
    builder = SceneContractBuilder()
    planner = TrajectoryPlanner()
    for record in manifest:
        prompt_hash = hashlib.sha256(record.get("prompt", "").encode("utf-8")).hexdigest()
        if prompt_hash in prompt_hashes:
            errors.append(f"duplicate prompt hash for {record.get('case_id')}")
        prompt_hashes.add(prompt_hash)
        expectations = record.get("trajectory_expectations", {})
        if len(expectations.get("event_order", [])) < 5:
            errors.append(f"{record.get('case_id')} lacks a complex event sequence")
        if not expectations.get("camera_types"):
            errors.append(f"{record.get('case_id')} lacks camera trajectory metadata")
        try:
            contract = builder.build(record["prompt"], duration_s=record["duration_s"], fps=record["fps"])
            plan = planner.plan(contract)
            actual_events = [event.id for event in contract.events]
            actual_cameras = sorted({shot.trajectory_type for shot in plan.camera.shots})
            if actual_events != expectations["event_order"]:
                errors.append(f"{record['case_id']} event metadata drift")
            if actual_cameras != expectations["camera_types"]:
                errors.append(f"{record['case_id']} camera metadata drift")
        except Exception as exc:
            errors.append(f"{record.get('case_id')} planner failure: {type(exc).__name__}: {exc}")

    manifest_ids = set(case_ids)
    for label in labels:
        if label.get("case_id") not in manifest_ids:
            errors.append(f"label references unknown case {label.get('case_id')}")
    if errors:
        raise ValueError("trajectory dataset validation failed: " + "; ".join(errors))
    return {
        "dataset_id": "trajectory-v2",
        "cases": len(manifest),
        "splits": {name: len(values) for name, values in splits.items()},
        "labels": len(labels),
        "unique_prompt_hashes": len(prompt_hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset/trajectory-v2")
    args = parser.parse_args()
    try:
        print(json.dumps(validate_dataset(args.root), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
