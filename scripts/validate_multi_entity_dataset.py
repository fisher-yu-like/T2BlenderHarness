"""Validate the frozen trajectory-v4-multi dataset without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {"train": 50, "dev": 60, "test": 30}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: item["case_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_acyclic(events: list[dict[str, Any]]) -> None:
    graph = {event["id"]: list(event.get("depends_on", [])) for event in events}
    if any(dependency not in graph for dependencies in graph.values() for dependency in dependencies):
        raise ValueError("event graph references an unknown dependency")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(event_id: str) -> None:
        if event_id in visited:
            return
        if event_id in visiting:
            raise ValueError(f"event graph cycle at {event_id}")
        visiting.add(event_id)
        for dependency in graph[event_id]:
            visit(dependency)
        visiting.remove(event_id)
        visited.add(event_id)

    for event_id in graph:
        visit(event_id)


def validate_dataset(dataset_root: str | Path = "dataset/trajectory-v4-multi") -> dict[str, Any]:
    root = Path(dataset_root)
    required_files = ("manifest.jsonl", "proxy_specs.jsonl", "labels.jsonl", "splits.json", "metadata.json")
    missing = [name for name in required_files if not (root / name).is_file()]
    if missing:
        raise ValueError(f"missing dataset files: {missing}")
    records = _read_jsonl(root / "manifest.jsonl")
    labels = _read_jsonl(root / "labels.jsonl")
    specs = _read_jsonl(root / "proxy_specs.jsonl")
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if {name: len(values) for name, values in splits.items()} != EXPECTED:
        raise ValueError(f"split counts differ: {splits}")
    if len(records) != sum(EXPECTED.values()):
        raise ValueError(f"expected 140 records, got {len(records)}")
    ids = [record["case_id"] for record in records]
    if len(ids) != len(set(ids)) or ids != sorted(ids):
        raise ValueError("manifest case IDs must be unique and sorted")
    split_ids = {split: set(values) for split, values in splits.items()}
    if set().union(*split_ids.values()) != set(ids) or any(split_ids[left] & split_ids[right] for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("case IDs are not partitioned into disjoint splits")
    by_id = {record["case_id"]: record for record in records}
    if {item["case_id"] for item in labels} != set(ids) or {item["case_id"] for item in specs} != set(ids):
        raise ValueError("labels/proxy specs do not cover exactly the manifest cases")
    for record in records:
        required = {"prompt_hash", "event_graph", "interactions", "camera_evidence", "negative_constraints", "composition_signature", "template_family", "difficulty"}
        if not required <= set(record):
            raise ValueError(f"case {record['case_id']} is missing authored fields")
        expected_hash = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
        if record["prompt_hash"] != expected_hash:
            raise ValueError(f"prompt hash mismatch for {record['case_id']}")
        entities = record["entities"]
        entity_ids = [entity["id"] for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError(f"duplicate entity ID in {record['case_id']}")
        if len([entity for entity in entities if entity["kind"] == "actor"]) < 2 or len([entity for entity in entities if entity["kind"] == "prop"]) < 2:
            raise ValueError(f"case {record['case_id']} is not multi-entity")
        _assert_acyclic(record["event_graph"])
        event_ids = {event["id"] for event in record["event_graph"]}
        if set(record["required_events"]) != event_ids:
            raise ValueError(f"required event set mismatch for {record['case_id']}")
        for interaction in record["interactions"]:
            for field in ("attach_event_id", "transfer_event_id", "detach_event_id"):
                if interaction.get(field) and interaction[field] not in event_ids:
                    raise ValueError(f"interaction references unknown event in {record['case_id']}")
        for shot in record["camera_evidence"]:
            if not shot["required_event_ids"] or not shot["target_ids"] or not shot["visibility_predicates"]:
                raise ValueError(f"incomplete camera evidence in {record['case_id']}")
        if not record["negative_constraints"]:
            raise ValueError(f"missing negative constraints in {record['case_id']}")
    families = {split: {by_id[case_id]["template_family"] for case_id in case_ids} for split, case_ids in split_ids.items()}
    compositions = {split: {by_id[case_id]["composition_signature"] for case_id in case_ids} for split, case_ids in split_ids.items()}
    family_overlap = sorted((families["train"] & families["dev"]) | (families["train"] & families["test"]) | (families["dev"] & families["test"]))
    composition_overlap = sorted((compositions["train"] & compositions["dev"]) | (compositions["train"] & compositions["test"]) | (compositions["dev"] & compositions["test"]))
    if family_overlap or composition_overlap:
        raise ValueError(f"split leakage family={family_overlap} composition={composition_overlap}")
    mean_difficulty = {split: sum(by_id[case_id]["difficulty"] for case_id in case_ids) / len(case_ids) for split, case_ids in split_ids.items()}
    if not mean_difficulty["train"] < mean_difficulty["dev"] < mean_difficulty["test"]:
        raise ValueError(f"dev/test are not harder than train: {mean_difficulty}")
    fingerprint = _fingerprint(records)
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("metadata fingerprint does not match manifest")
    return {
        "status": "pass",
        "dataset_id": metadata.get("dataset_id"),
        "case_count": len(records),
        "splits": {name: len(values) for name, values in splits.items()},
        "family_overlap": family_overlap,
        "composition_overlap": composition_overlap,
        "mean_difficulty": mean_difficulty,
        "fingerprint": fingerprint,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/trajectory-v4-multi")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
