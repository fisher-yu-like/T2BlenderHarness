from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "field",
    ("case_id", "prompt_hash", "source_identity", "semantic_signature"),
)
def test_frozen_validator_rejects_each_reference_overlap_kind(
    tmp_path: Path, field: str
) -> None:
    from scripts.validate_frozen_eval_set import validate_frozen_eval_set

    frozen_root = Path(__file__).resolve().parents[1] / "dataset" / "frozen-eval-v1"
    frozen_record = json.loads((frozen_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    reference_value: str | tuple[str, int] = {
        "case_id": frozen_record["case_id"],
        "prompt_hash": frozen_record["prompt_hash"],
        "source_identity": (frozen_record["source_dataset"], int(frozen_record["source_index"])),
        "semantic_signature": frozen_record["semantic_signature"],
    }[field]
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    record = {
        "case_id": "reference-only-case",
        "prompt": "A different reference prompt.",
        "prompt_hash": "different-prompt-hash",
        "source_dataset": "reference-dataset",
        "source_index": 99,
        "semantic_signature": "different-semantic-signature",
    }
    if field == "case_id":
        record["case_id"] = reference_value
    elif field == "prompt_hash":
        record["prompt_hash"] = reference_value
    elif field == "source_identity":
        record["source_dataset"], record["source_index"] = reference_value
    else:
        record["semantic_signature"] = reference_value
    (reference_root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (reference_root / "metadata.json").write_text(json.dumps({"dataset_id": "reference-dataset"}), encoding="utf-8")

    report = validate_frozen_eval_set(frozen_root, reference_roots=[reference_root])

    assert report["status"] == "fail"
    assert any(field in error for error in report["errors"])


def test_frozen_builder_selects_raw_source_outside_reference_and_training_sets(tmp_path: Path) -> None:
    from scripts.build_frozen_eval_set import build_frozen_eval_set

    raw_source = tmp_path / "raw.json"
    raw_records = [
        {"prompt_en": f"raw prompt {index}", "dimension": [dimension], "auxiliary_info": f"cue-{index}"}
        for index, dimension in enumerate(
            (
                "Camera_Motion",
                "Mechanics",
                "Material",
                "Thermotics",
                "Composition",
                "Human_Anatomy",
                "Dynamic_Attribute",
                "Motion_Rationality",
                "Multi-View_Consistency",
                "Human_Clothes",
                "Instance_Preservation",
                "Diversity",
                "Complex_Landscape",
            )
        )
    ]
    raw_records[7]["prompt_en"] = raw_records[2]["prompt_en"]
    raw_source.write_text(json.dumps(raw_records), encoding="utf-8")

    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    reference_record = {
        "case_id": "reference-raw-0",
        "prompt": raw_records[0]["prompt_en"],
        "prompt_hash": "reference-prompt-hash",
        "source_dataset": "VBench-2.0",
        "source_index": 0,
        "semantic_signature": "reference-semantic",
    }
    (reference_root / "manifest.jsonl").write_text(json.dumps(reference_record) + "\n", encoding="utf-8")
    (reference_root / "metadata.json").write_text(json.dumps({"dataset_id": "vbench-derived-100-v1"}), encoding="utf-8")

    training_root = tmp_path / "training"
    training_root.mkdir()
    (training_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "prompt_hash": hashlib.sha256(raw_records[1]["prompt_en"].encode("utf-8")).hexdigest(),
                "source_dataset": "VBench-2.0",
                "source_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "frozen"
    report = build_frozen_eval_set(
        source_root=raw_source,
        training_root=training_root,
        reference_roots=[reference_root],
        output_root=output_root,
        per_category=1,
        category_count=10,
    )

    assert report["status"] == "pass"
    rows = [json.loads(line) for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10
    assert len({row["prompt_hash"] for row in rows}) == len(rows)
    assert all(row["source_index"] not in {0, 1} for row in rows)
    assert report["excluded_reference_count"] == 1
