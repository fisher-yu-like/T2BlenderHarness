from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest


def test_agent_training_dataset_has_six_rounds_and_disjoint_case_families(tmp_path: Path) -> None:
    from scripts.build_agent_training_dataset import build_dataset
    from scripts.validate_agent_training_dataset import validate_dataset

    metadata = build_dataset(tmp_path / "agent-v1")
    report = validate_dataset(tmp_path / "agent-v1")

    assert metadata["dataset_id"] == "trajectory-v5-agent-codegen"
    assert report["splits"] == {"train": 60, "dev": 60, "test": 20}
    assert report["unique_prompt_hashes"] == 140
    assert report["round_count"] == 6

    records = [
        json.loads(line)
        for line in (tmp_path / "agent-v1" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(len([entity for entity in record["entities"] if entity["kind"] == "actor"]) >= 2 for record in records)
    assert all(len([entity for entity in record["entities"] if entity["kind"] == "prop"]) >= 2 for record in records)


def test_agent_training_dataset_validator_rejects_prompt_hash_tampering(tmp_path: Path) -> None:
    from scripts.build_agent_training_dataset import build_dataset
    from scripts.validate_agent_training_dataset import validate_dataset

    root = tmp_path / "agent-v1"
    build_dataset(root)
    manifest = root / "manifest.jsonl"
    lines = manifest.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["prompt_hash"] = "0" * 64
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prompt hash mismatch"):
        validate_dataset(root)


def test_frozen_eval_validator_requires_disjoint_source_identity(tmp_path: Path) -> None:
    from scripts.validate_frozen_eval_set import validate_frozen_eval_set

    root = tmp_path / "frozen"
    root.mkdir()
    rows = [
        {
            "case_id": f"frozen-{index:02d}",
            "prompt": f"A bird flies over fountain {index}.",
            "prompt_hash": hashlib.sha256(f"A bird flies over fountain {index}.".encode("utf-8")).hexdigest(),
            "source_dataset": "independent-vbench-holdout",
            "source_index": index,
            "semantic_signature": f"bird-over-fountain-{index}",
        }
        for index in range(10)
    ]
    (root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (root / "metadata.json").write_text(
        json.dumps({"dataset_id": "frozen-eval-v1", "train_dev_prompt_hashes": [rows[0]["prompt_hash"]]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        validate_frozen_eval_set(root)
