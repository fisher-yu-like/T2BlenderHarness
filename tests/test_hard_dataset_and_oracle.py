import json


def test_hard_dataset_has_unique_scenes_and_family_holdout(tmp_path):
    from scripts.build_hard_trajectory_dataset import build_dataset
    from scripts.validate_hard_trajectory_dataset import validate_dataset

    summary = build_dataset(tmp_path)
    validation = validate_dataset(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["cases"] == 140
    assert validation["unique_prompt_hashes"] == 140
    assert len({record["proxy_scene"]["scene_id"] for record in records}) == 140
    assert {split: len(ids) for split, ids in json.loads((tmp_path / "splits.json").read_text()).items()} == {
        "train": 60,
        "dev": 60,
        "test": 20,
    }
    families = {record["template_family"]: record["split"] for record in records}
    assert len(families) == 14
    assert set(families.values()) == {"train", "dev", "test"}


def test_independent_oracle_accepts_supported_case_and_catches_unsupported_case():
    from evaluator.independent_oracle import evaluate_independent_oracle
    from scripts.build_hard_trajectory_dataset import FAMILIES
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    supported = {
        "case_id": "supported",
        "prompt": FAMILIES[0]["template"].format(object="red cup", support="table", drop_zone="drop zone"),
        "duration_s": 16.0,
        "fps": 24,
        "oracle_expectations": {
            "event_order": FAMILIES[0]["events"],
            "required_camera_types": FAMILIES[0]["cameras"],
            "required_camera_constraints": FAMILIES[0]["constraints"],
            "required_motion_primitives": FAMILIES[0]["primitives"],
            "required_attachment_actions": FAMILIES[0]["attachments"],
            "required_entity_kinds": {"character": "character", "table": "support", "drop_zone": "support", "red_cup": "prop"},
        },
    }
    contract = SceneContractBuilder().build(supported["prompt"], duration_s=16.0, fps=24)
    plan = TrajectoryPlanner().plan(contract)
    assert evaluate_independent_oracle(supported, contract, plan) == []

    unsupported = dict(supported)
    unsupported["oracle_expectations"] = {**supported["oracle_expectations"], "event_order": ["stroll", *FAMILIES[0]["events"][1:]]}
    findings = evaluate_independent_oracle(unsupported, contract, plan)
    assert any(finding.failure_id == "oracle_event_order_mismatch" for finding in findings)
