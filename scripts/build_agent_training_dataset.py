"""Build the frozen six-round dataset for provider-backed Harness training.

The historical ``trajectory-v4-multi`` dataset is left untouched because its
50/60/30 protocol belongs to an earlier experiment.  This builder reuses its
authored case generator, assigns six train and six dev ten-case families, and
keeps a disjoint twenty-case blind test split.
"""

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

from scripts.build_multi_entity_dataset import (
    DEV_FAMILIES,
    TEST_FAMILIES,
    TRAIN_FAMILIES,
    _fingerprint,
    _json_line,
    _make_case,
)


AGENT_TRAIN_FAMILIES = (*TRAIN_FAMILIES, "subjectless_handoff")
AGENT_DEV_FAMILIES = DEV_FAMILIES
AGENT_TEST_FAMILIES = TEST_FAMILIES[:2]


def _rename_case(record: dict[str, Any], *, split: str, family_number: int, variant: int) -> dict[str, Any]:
    case_id = f"agent-{split}-{family_number:02d}-{variant:02d}"
    record = json.loads(json.dumps(record))
    record["case_id"] = case_id
    record["composition_signature"] = (
        f"agent:{split}:{record['template_family']}:{len(record['entities'])}:v{variant:02d}"
    )
    record["prompt_hash"] = hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
    record["proxy_scene"]["scene_id"] = f"agent-proxy-scene-{split}-{family_number:02d}-{variant:02d}"
    record["proxy_scene"]["scene_seed"] = 80000 + family_number * 100 + variant
    return record


def build_dataset(output_root: str | Path = "dataset/trajectory-v5-agent-codegen") -> dict[str, Any]:
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    family_map = {
        "train": AGENT_TRAIN_FAMILIES,
        "dev": AGENT_DEV_FAMILIES,
        "test": AGENT_TEST_FAMILIES,
    }
    records: list[dict[str, Any]] = []
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    global_variant = 1
    for split, families in family_map.items():
        for family_offset, family in enumerate(families, 1):
            for slot in range(1, 11):
                record = _make_case(split, family, global_variant)
                record = _rename_case(
                    record,
                    split=split,
                    family_number=(family_offset if split == "train" else 6 + family_offset if split == "dev" else 12 + family_offset),
                    variant=slot,
                )
                records.append(record)
                splits[split].append(record["case_id"])
                global_variant += 1

    records.sort(key=lambda item: item["case_id"])
    labels = [
        {
            "case_id": record["case_id"],
            "split": record["split"],
            "template_family": record["template_family"],
            "event_graph": record["event_graph"],
            "interactions": record["interactions"],
            "camera_evidence": record["camera_evidence"],
            "negative_constraints": record["negative_constraints"],
            "oracle_expectations": record["oracle_expectations"],
        }
        for record in records
    ]
    specs = [
        {"case_id": record["case_id"], "split": record["split"], "proxy_scene": record["proxy_scene"]}
        for record in records
    ]
    for name, payload in (("manifest.jsonl", records), ("labels.jsonl", labels), ("proxy_specs.jsonl", specs)):
        (destination / name).write_text("".join(_json_line(item) for item in payload), encoding="utf-8")
    split_payload = {split: sorted(values) for split, values in splits.items()}
    (destination / "splits.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "dataset_id": "trajectory-v5-agent-codegen",
        "schema_version": "trajectory-dataset-v5-agent-codegen",
        "generator_version": "agent-codegen-dataset-v1",
        "cases": len(records),
        "splits": {split: len(values) for split, values in split_payload.items()},
        "round_count": 6,
        "families": {split: list(families) for split, families in family_map.items()},
        "fingerprint": _fingerprint(records),
        "train_policy": "six disjoint ten-case families; only repeated failures across two train cases may propose a patch",
        "dev_policy": "six disjoint paired holdout families; dev is never edited or used to construct labels",
        "test_policy": "twenty disjoint blind cases; test is milestone-only and cannot enter proposal or acceptance selection",
        "prompt_policy": "every case has a unique prompt hash and multi-entity scene with at least two actors and two props",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset/trajectory-v5-agent-codegen")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
