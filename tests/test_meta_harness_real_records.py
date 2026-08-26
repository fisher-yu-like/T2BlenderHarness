import json

import pytest


def failure_record(case_id="case-01", owner="camera_planner"):
    return {
        "case_id": case_id,
        "split": "train",
        "status": "fail",
        "score": 45,
        "findings": [
            {
                "failure_id": "camera_event_uncovered",
                "owner": owner,
                "category": "camera_coverage",
                "severity": "hard",
                "message": "grasp not covered",
                "evidence": ["deterministic_report.json"],
                "repair_route": "camera_repair",
            }
        ],
    }


def test_meta_harness_proposes_one_owner_from_real_train_records(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    optimizer = MetaHarnessOptimizer(output_dir=tmp_path)
    proposal = optimizer.propose([failure_record("case-01"), failure_record("case-02")])

    assert proposal.owner == "camera_planner"
    assert proposal.affected_files == ["src/videoact/camera.py"]
    assert proposal.patch_scope == "one-harness-owner"


def test_meta_harness_rejects_test_case_leakage(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    optimizer = MetaHarnessOptimizer(output_dir=tmp_path)
    with pytest.raises(ValueError, match="test split"):
        optimizer.propose([failure_record("case-test")], forbidden_case_ids={"case-test"})


def test_meta_harness_records_train_dev_acceptance_and_patch_diff(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    optimizer = MetaHarnessOptimizer(output_dir=tmp_path)
    proposal = optimizer.propose([failure_record("case-01"), failure_record("case-02")])
    record = optimizer.record_acceptance(
        proposal,
        before={"train_score": 60, "dev_score": 58},
        after={"train_score": 63, "dev_score": 58},
        train={"hard_regression": False},
        dev={"hard_regression": False},
        patch_diff="diff --git a/src/videoact/camera.py b/src/videoact/camera.py",
    )

    assert record["acceptance"]["accepted"] is True
    assert record["owner"] == "camera_planner"
    saved = json.loads((tmp_path / "optimization_record.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert saved["patch_diff"].startswith("diff --git")
