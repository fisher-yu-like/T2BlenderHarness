"""Validate the hard dataset without treating runtime planner output as gold."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_dataset(root: str | Path = "dataset/trajectory-v3-hard") -> dict[str, Any]:
    destination = Path(root)
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    manifest = _records(destination / "manifest.jsonl")
    labels = _records(destination / "labels.jsonl")
    proxy_specs = _records(destination / "proxy_specs.jsonl")
    splits = json.loads((destination / "splits.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if metadata.get("cases") != 140 or len(manifest) != 140:
        errors.append(f"expected 140 cases, got metadata={metadata.get('cases')} manifest={len(manifest)}")
    expected_splits = {"train": 60, "dev": 60, "test": 20}
    if {key: len(value) for key, value in splits.items()} != expected_splits:
        errors.append(f"expected splits {expected_splits}, got { {key: len(value) for key, value in splits.items()} }")
    case_ids = [record.get("case_id") for record in manifest]
    if len(case_ids) != len(set(case_ids)):
        errors.append("case IDs must be unique")
    prompt_hashes = [hashlib.sha256(record.get("prompt", "").encode("utf-8")).hexdigest() for record in manifest]
    if len(prompt_hashes) != len(set(prompt_hashes)):
        errors.append("prompt hashes must be unique")
    family_to_split: dict[str, str] = {}
    split_ids = set()
    for split, ids in splits.items():
        split_ids.update(ids)
        for case_id in ids:
            record = next((item for item in manifest if item.get("case_id") == case_id), None)
            if record is None:
                errors.append(f"split references unknown case {case_id}")
                continue
            if record.get("split") != split:
                errors.append(f"{case_id} split metadata mismatch")
            family = record.get("template_family")
            previous = family_to_split.setdefault(family, split)
            if previous != split:
                errors.append(f"template family {family} leaks across splits")
    if split_ids != set(case_ids):
        errors.append("splits must cover every case exactly once")

    for record in manifest:
        oracle = record.get("oracle_expectations", {})
        proxy = record.get("proxy_scene", {})
        if len(oracle.get("event_order", [])) < 7:
            errors.append(f"{record.get('case_id')} oracle event sequence is too short")
        if not oracle.get("required_camera_types"):
            errors.append(f"{record.get('case_id')} lacks oracle camera expectations")
        if not proxy.get("entities") or not proxy.get("required_artifacts"):
            errors.append(f"{record.get('case_id')} lacks proxy scene contract")
        if not proxy.get("scene_id") or not proxy.get("layout", {}).get("path_shape"):
            errors.append(f"{record.get('case_id')} lacks unique proxy layout")
        if record.get("difficulty", 0) < 3:
            errors.append(f"{record.get('case_id')} difficulty is below hard-dataset threshold")
    if len(labels) != 140 or len(proxy_specs) != 140:
        errors.append("labels and proxy specs must each contain 140 cases")
    scene_ids = [record.get("proxy_scene", {}).get("scene_id") for record in manifest]
    if len(scene_ids) != len(set(scene_ids)):
        errors.append("proxy scene IDs must be unique")
    if errors:
        raise ValueError("hard trajectory dataset validation failed: " + "; ".join(errors))
    return {
        "dataset_id": metadata["dataset_id"],
        "cases": len(manifest),
        "splits": {name: len(values) for name, values in splits.items()},
        "families": len(family_to_split),
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "split_policy": metadata["split_policy"],
    }


if __name__ == "__main__":
    print(json.dumps(validate_dataset(), indent=2, sort_keys=True))
