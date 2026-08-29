from __future__ import annotations

import json
from pathlib import Path


TRAIN_DIMENSIONS = (
    "Camera_Motion",
    "Human_Interaction",
    "Motion_Order_Understanding",
    "Complex_Plot",
    "Dynamic_Spatial_Relationship",
    "Mechanics",
)
TEST_DIMENSIONS = ("Motion_Rationality", "Human_Clothes")


def _write_source(path: Path, per_dimension: int = 20) -> list[dict]:
    rows = []
    for dimension in (*TRAIN_DIMENSIONS, *TEST_DIMENSIONS):
        for index in range(per_dimension):
            rows.append(
                {
                    "prompt_en": f"Benchmark {dimension} prompt {index}.",
                    "dimension": [dimension],
                    "auxiliary_info": {"index": index},
                }
            )
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def test_benchmark_index_preserves_verbatim_prompts_without_authored_labels(tmp_path: Path) -> None:
    from scripts.build_benchmark_prompt_index import build_index

    source = tmp_path / "VBench2_full_info.json"
    raw_rows = _write_source(source)
    output = tmp_path / "index"

    report = build_index(source, output)

    assert report["dataset_kind"] == "benchmark_prompt_index"
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 140
    assert all(record["prompt"] == record["source_prompt"] for record in records)
    assert all(record["prompt_origin"] == "benchmark_verbatim" for record in records)
    assert all(record["benchmark_prompt_only"] is True for record in records)
    assert all("event_graph" not in record and "oracle_expectations" not in record for record in records)
    assert {record["prompt"] for record in records}.issubset({row["prompt_en"] for row in raw_rows})


def test_benchmark_index_validator_rejects_prompt_mutation_and_missing_provenance(tmp_path: Path) -> None:
    from scripts.build_benchmark_prompt_index import build_index
    from scripts.validate_benchmark_prompt_index import validate_benchmark_prompt_index

    source = tmp_path / "VBench2_full_info.json"
    _write_source(source)
    output = tmp_path / "index"
    build_index(source, output)

    first = json.loads((output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
    first["prompt"] += " synthetic addition"
    first.pop("prompt_origin")
    lines = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps(first, ensure_ascii=False, sort_keys=True)
    (output / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = validate_benchmark_prompt_index(output, source_path=source)

    assert report["status"] == "fail"
    assert any("prompt" in error or "prompt_origin" in error for error in report["errors"])


def test_non_benchmark_training_dataset_is_not_eligible() -> None:
    from scripts.validate_benchmark_prompt_index import validate_benchmark_prompt_index

    report = validate_benchmark_prompt_index("dataset/trajectory-v5-agent-codegen")

    assert report["status"] == "fail"
    assert any("benchmark" in error for error in report["errors"])


def test_training_runner_rejects_historical_self_built_dataset() -> None:
    from scripts.train_real_harness import require_benchmark_training_dataset

    try:
        require_benchmark_training_dataset("dataset/trajectory-v5-agent-codegen")
    except ValueError as exc:
        assert "benchmark prompt index" in str(exc)
    else:  # pragma: no cover - assertion documents the fail-closed contract
        raise AssertionError("self-built dataset unexpectedly passed the training entry gate")


def test_benchmark_index_validator_rejects_wrong_round_family_metadata(tmp_path: Path) -> None:
    from scripts.build_benchmark_prompt_index import build_index
    from scripts.validate_benchmark_prompt_index import validate_benchmark_prompt_index

    source = tmp_path / "VBench2_full_info.json"
    _write_source(source)
    output = tmp_path / "index"
    build_index(source, output)

    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    metadata["train_families"] = ["vbench2-train-99"]
    (output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    report = validate_benchmark_prompt_index(output, source_path=source)

    assert report["status"] == "fail"
    assert any("family" in error for error in report["errors"])
