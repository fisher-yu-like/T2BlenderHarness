import json
import subprocess
import sys


def test_dataset_has_forty_unique_cases_and_disjoint_splits():
    root = __import__("pathlib").Path("dataset")
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    splits = json.loads((root / "splits.json").read_text(encoding="utf-8"))

    ids = [record["case_id"] for record in records]
    assert len(records) == 40
    assert len(ids) == len(set(ids))
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 20,
        "dev": 10,
        "test": 10,
    }
    assert set(splits["train"]).isdisjoint(splits["dev"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["dev"]).isdisjoint(splits["test"])
    assert set(ids) == set().union(*[set(values) for values in splits.values()])


def test_dataset_records_have_required_contract_fields_and_unique_prompt_hashes():
    root = __import__("pathlib").Path("dataset")
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
    required = {"case_id", "prompt", "category", "entities", "required_events", "expected_relations", "duration_s", "fps", "evaluator_version"}

    assert all(required <= set(record) for record in records)
    assert len({record["prompt"] for record in records}) == 40


def test_dataset_validator_accepts_checked_in_files():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/validate_dataset.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
