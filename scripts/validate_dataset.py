"""Validate the frozen 40-case proxy benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "case_id",
    "prompt",
    "category",
    "entities",
    "required_events",
    "expected_relations",
    "duration_s",
    "fps",
    "evaluator_version",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(root: str | Path = "dataset") -> dict[str, Any]:
    root = Path(root)
    manifest = _load_jsonl(root / "manifest.jsonl")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    labels = _load_jsonl(root / "labels.jsonl")
    errors: list[str] = []
    case_ids = [record.get("case_id") for record in manifest]
    if len(manifest) != 40:
        errors.append(f"expected 40 manifest records, got {len(manifest)}")
    if len(case_ids) != len(set(case_ids)):
        errors.append("manifest case IDs must be unique")
    if set(splits) != {"train", "dev", "test"}:
        errors.append("splits must contain train, dev, and test")
    all_split_ids = []
    for name, expected_count in {"train": 20, "dev": 10, "test": 10}.items():
        values = splits.get(name, [])
        if len(values) != expected_count:
            errors.append(f"{name} must contain {expected_count} cases")
        all_split_ids.extend(values)
    if len(all_split_ids) != len(set(all_split_ids)):
        errors.append("split IDs must be disjoint")
    if set(all_split_ids) != set(case_ids):
        errors.append("splits must cover every manifest case exactly once")
    prompt_hashes = set()
    for record in manifest:
        missing = REQUIRED_FIELDS - set(record)
        if missing:
            errors.append(f"{record.get('case_id', '<unknown>')} missing {sorted(missing)}")
        prompt_hash = hashlib.sha256(str(record.get("prompt", "")).encode("utf-8")).hexdigest()
        if prompt_hash in prompt_hashes:
            errors.append(f"duplicate prompt hash for {record.get('case_id')}")
        prompt_hashes.add(prompt_hash)
    manifest_ids = set(case_ids)
    for label in labels:
        if label.get("case_id") not in manifest_ids:
            errors.append(f"label references unknown case {label.get('case_id')}")
        for field in ("pass_fail", "event_coverage", "physics_plausibility", "camera_quality", "primary_failure_owner"):
            if field not in label:
                errors.append(f"label {label.get('case_id')} missing {field}")
    if errors:
        raise ValueError("dataset validation failed: " + "; ".join(errors))
    return {
        "cases": len(manifest),
        "splits": {name: len(values) for name, values in splits.items()},
        "labels": len(labels),
        "unique_prompt_hashes": len(prompt_hashes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset")
    args = parser.parse_args()
    try:
        print(json.dumps(validate_dataset(args.root), sort_keys=True))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
