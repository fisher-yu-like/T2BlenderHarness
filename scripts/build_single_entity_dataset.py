"""Build a single-character/single-prop phase-1 dataset from the hard set.

The original hard dataset is preserved.  This derived set removes multi-actor
and multi-prop cases for a controlled first training phase; its audit records
exactly what was excluded so the later compositional upgrade is reproducible.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_single_entity(record: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    entities = record.get("proxy_scene", {}).get("entities", [])
    characters = [entity for entity in entities if entity.get("kind") == "character"]
    props = [entity for entity in entities if entity.get("kind") == "prop"]
    details = {
        "character_ids": [entity.get("id") for entity in characters],
        "prop_ids": [entity.get("id") for entity in props],
        "character_count": len(characters),
        "prop_count": len(props),
    }
    return len(characters) == 1 and len(props) == 1, details


def build_dataset(source_root: str | Path = "dataset/trajectory-v3-hard", output_root: str | Path = "dataset/trajectory-v3-single") -> dict[str, Any]:
    source = Path(source_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(source / "manifest.jsonl")
    labels = {item["case_id"]: item for item in _read_jsonl(source / "labels.jsonl")}
    proxy_specs = {item["case_id"]: item for item in _read_jsonl(source / "proxy_specs.jsonl")}
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded: list[dict[str, Any]] = []
    for record in records:
        keep, details = _is_single_entity(record)
        if keep:
            by_split[record["split"]].append(copy.deepcopy(record))
        else:
            excluded.append({"case_id": record["case_id"], "split": record["split"], "template_family": record.get("template_family"), **details})

    expected = {"train": 50, "dev": 50, "test": 20}
    actual = {split: len(values) for split, values in by_split.items()}
    if actual != expected:
        raise ValueError(f"single-entity selection mismatch: expected {expected}, got {actual}")

    output_records: list[dict[str, Any]] = []
    splits = {"train": [], "dev": [], "test": []}
    source_to_new: dict[str, str] = {}
    next_family = 1
    global_index = 1
    for split in ("train", "dev", "test"):
        family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in sorted(by_split[split], key=lambda item: item["case_id"]):
            family_groups[record["case_id"].rsplit("-", 1)[0]].append(record)
        for source_family in sorted(family_groups):
            group = family_groups[source_family]
            if len(group) != 10:
                raise ValueError(f"expected complete family of 10, got {source_family}={len(group)}")
            for variant, record in enumerate(group, start=1):
                old_id = record["case_id"]
                new_id = f"single-{next_family:02d}-{variant:02d}"
                source_to_new[old_id] = new_id
                record["case_id"] = new_id
                record["dataset_source_case_id"] = old_id
                record["proxy_scene"]["scene_id"] = f"single-proxy-scene-{global_index:03d}"
                record["proxy_scene"]["dataset_source_case_id"] = old_id
                output_records.append(record)
                splits[split].append(new_id)
                global_index += 1
            next_family += 1

    output_labels = []
    output_specs = []
    for record in output_records:
        old_id = record["dataset_source_case_id"]
        label = copy.deepcopy(labels[old_id])
        label["case_id"] = record["case_id"]
        label["dataset_source_case_id"] = old_id
        output_labels.append(label)
        spec = copy.deepcopy(proxy_specs[old_id])
        spec["case_id"] = record["case_id"]
        spec["proxy_scene"] = copy.deepcopy(record["proxy_scene"])
        spec["dataset_source_case_id"] = old_id
        output_specs.append(spec)

    for name, payload in (
        ("manifest.jsonl", output_records),
        ("labels.jsonl", output_labels),
        ("proxy_specs.jsonl", output_specs),
    ):
        (destination / name).write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in payload), encoding="utf-8")
    (destination / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "dataset_id": "trajectory-v3-single",
        "schema_version": "trajectory-dataset-v3-single",
        "source_dataset": "dataset/trajectory-v3-hard",
        "selection_policy": "exactly one character entity and one prop entity in proxy_scene.entities",
        "cases": len(output_records),
        "families": next_family - 1,
        "splits": {split: len(ids) for split, ids in splits.items()},
        "fingerprint": _fingerprint(output_records),
        "test_policy": "frozen test remains single-character/single-prop and is never used for patch selection",
        "excluded_case_count": len(excluded),
        "excluded_by_reason": dict(Counter("multiple_people" if item["character_count"] > 1 else "multiple_objects" for item in excluded)),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = {
        "audit_version": "single-entity-selection-v1",
        "source_root": str(source.resolve()),
        "output_root": str(destination.resolve()),
        "source_case_count": len(records),
        "selected_case_count": len(output_records),
        "selected_split_counts": {split: len(ids) for split, ids in splits.items()},
        "selected_case_policy": metadata["selection_policy"],
        "excluded_case_count": len(excluded),
        "excluded_cases": excluded,
        "source_to_output_case_id": source_to_new,
        "output_fingerprint": metadata["fingerprint"],
    }
    (destination / "selection_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--output-root", default="dataset/trajectory-v3-single")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source_root, args.output_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
