"""Validate the 60/60/20 dataset used by the six-round agent protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SPLITS = {"train": 60, "dev": 60, "test": 20}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: item["case_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assert_acyclic(events: list[dict[str, Any]], case_id: str) -> None:
    graph = {str(event["id"]): list(event.get("depends_on", [])) for event in events}
    if len(graph) != len(events):
        raise ValueError(f"case {case_id} has duplicate event IDs")
    if any(dependency not in graph for dependencies in graph.values() for dependency in dependencies):
        raise ValueError(f"case {case_id} event graph references an unknown dependency")
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(event_id: str) -> None:
        if event_id in visited:
            return
        if event_id in visiting:
            raise ValueError(f"case {case_id} event graph contains a cycle")
        visiting.add(event_id)
        for dependency in graph[event_id]:
            visit(dependency)
        visiting.remove(event_id)
        visited.add(event_id)

    for event_id in graph:
        visit(event_id)


def validate_dataset(root: str | Path = "dataset/trajectory-v5-agent-codegen") -> dict[str, Any]:
    destination = Path(root)
    required = ("manifest.jsonl", "labels.jsonl", "proxy_specs.jsonl", "splits.json", "metadata.json")
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        raise ValueError(f"missing dataset files: {missing}")
    records = _read_jsonl(destination / "manifest.jsonl")
    labels = _read_jsonl(destination / "labels.jsonl")
    specs = _read_jsonl(destination / "proxy_specs.jsonl")
    splits = json.loads((destination / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    counts = {name: len(values) for name, values in splits.items()}
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"expected split counts {EXPECTED_SPLITS}, got {counts}")
    if len(records) != 140:
        raise ValueError(f"expected 140 records, got {len(records)}")
    case_ids = [str(record.get("case_id", "")) for record in records]
    if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)) or "" in case_ids:
        raise ValueError("manifest case IDs must be non-empty, unique, and sorted")
    split_ids = {name: set(values) for name, values in splits.items()}
    if set().union(*split_ids.values()) != set(case_ids):
        raise ValueError("splits must cover every case exactly once")
    if any(split_ids[left] & split_ids[right] for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("split case IDs overlap")
    by_id = {record["case_id"]: record for record in records}
    if {item.get("case_id") for item in labels} != set(case_ids):
        raise ValueError("labels do not cover exactly the manifest cases")
    if {item.get("case_id") for item in specs} != set(case_ids):
        raise ValueError("proxy specs do not cover exactly the manifest cases")

    prompt_hashes: list[str] = []
    families_by_split: dict[str, set[str]] = {name: set() for name in EXPECTED_SPLITS}
    for record in records:
        case_id = record["case_id"]
        prompt = str(record.get("prompt", ""))
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if record.get("prompt_hash") != expected_hash:
            raise ValueError(f"prompt hash mismatch for {case_id}")
        prompt_hashes.append(expected_hash)
        split = str(record.get("split", ""))
        if split not in EXPECTED_SPLITS or case_id not in split_ids[split]:
            raise ValueError(f"split metadata mismatch for {case_id}")
        family = str(record.get("template_family", ""))
        if not family:
            raise ValueError(f"missing template family for {case_id}")
        families_by_split[split].add(family)
        entities = record.get("entities") or []
        actors = [entity for entity in entities if entity.get("kind") == "actor"]
        props = [entity for entity in entities if entity.get("kind") == "prop"]
        if len(actors) < 2 or len(props) < 2:
            raise ValueError(f"case {case_id} is not a two-actor/two-prop case")
        entity_ids = [str(entity.get("id")) for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError(f"duplicate entity ID in {case_id}")
        events = record.get("event_graph") or []
        if len(events) < 5:
            raise ValueError(f"case {case_id} has too few events")
        _assert_acyclic(events, case_id)
        event_ids = {str(event["id"]) for event in events}
        if set(record.get("required_events") or []) != event_ids:
            raise ValueError(f"required event set mismatch for {case_id}")
        camera = record.get("camera_evidence") or []
        if not camera or any(not shot.get("required_event_ids") or not shot.get("target_ids") for shot in camera):
            raise ValueError(f"incomplete camera evidence for {case_id}")
        proxy = record.get("proxy_scene") or {}
        if not proxy.get("scene_id") or not proxy.get("required_artifacts"):
            raise ValueError(f"incomplete proxy scene contract for {case_id}")

    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ValueError("prompt hashes must be unique")
    all_families = [family for families in families_by_split.values() for family in families]
    if len(all_families) != len(set(all_families)):
        raise ValueError("template families overlap across splits")
    expected_family_counts = {"train": 6, "dev": 6, "test": 2}
    if {name: len(values) for name, values in families_by_split.items()} != expected_family_counts:
        raise ValueError(f"expected family counts {expected_family_counts}, got {families_by_split}")
    for split, case_set in split_ids.items():
        family_counts: dict[str, int] = {}
        for case_id in case_set:
            family = by_id[case_id]["template_family"]
            family_counts[family] = family_counts.get(family, 0) + 1
        if any(count != 10 for count in family_counts.values()):
            raise ValueError(f"each {split} family must contain ten cases: {family_counts}")
    if metadata.get("fingerprint") != _fingerprint(records):
        raise ValueError("metadata fingerprint does not match manifest")
    if metadata.get("round_count") != 6:
        raise ValueError("dataset metadata must declare six rounds")
    return {
        "status": "pass",
        "dataset_id": metadata.get("dataset_id"),
        "case_count": len(records),
        "splits": counts,
        "round_count": metadata.get("round_count"),
        "families": {name: sorted(values) for name, values in families_by_split.items()},
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "fingerprint": metadata.get("fingerprint"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    args = parser.parse_args()
    print(json.dumps(validate_dataset(args.dataset_root), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
