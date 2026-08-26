import json


def test_dual_handoff_dataset_oracle_follows_prompt(tmp_path):
    from scripts.build_hard_trajectory_dataset import build_dataset

    build_dataset(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    hard03 = [record for record in records if record["case_id"].startswith("hard-03-")]
    assert len(hard03) == 10
    for record in hard03:
        assert "red cup" in record["prompt"].lower()
        assert "blue cube" in record["prompt"].lower()
        assert record["oracle_expectations"]["required_entity_ids"] == ["character", "table", "drop_zone", "red_cup", "blue_cube"]
        assert record["oracle_expectations"]["required_entity_kinds"]["red_cup"] == "prop"
        assert record["oracle_expectations"]["required_entity_kinds"]["blue_cube"] == "prop"
