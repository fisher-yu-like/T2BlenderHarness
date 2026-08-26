import json

import pytest


def ready_kwargs():
    return {
        "calibration_report": {"status": "ready", "labeled_cases": 10},
        "acceptance_record": {"accepted": True, "train_score_after": 80, "dev_score_after": 75},
        "reproducibility_report": {"reproducible": True, "test_report_hash": "abc"},
        "plan_pairs": [{"prompt_hash": "p", "bad": "bad-plan", "good": "good-plan", "score_delta": 4, "failure_tags": ["camera"]}],
        "trajectory_pairs": [{"prompt_hash": "p", "bad": "bad-traj", "good": "good-traj", "score_delta": 3, "failure_tags": ["physics"]}],
    }


def test_preference_export_refuses_unready_calibration(tmp_path):
    from training.export_preference_pairs import export_preference_pairs

    kwargs = ready_kwargs()
    kwargs["calibration_report"] = {"status": "not_ready", "labeled_cases": 0}

    with pytest.raises(ValueError, match="calibration"):
        export_preference_pairs(tmp_path, **kwargs)


def test_preference_export_refuses_rejected_harness_candidate(tmp_path):
    from training.export_preference_pairs import export_preference_pairs

    kwargs = ready_kwargs()
    kwargs["acceptance_record"] = {"accepted": False}

    with pytest.raises(ValueError, match="acceptance"):
        export_preference_pairs(tmp_path, **kwargs)


def test_preference_export_keeps_plan_and_trajectory_pairs_separate(tmp_path):
    from training.export_preference_pairs import export_preference_pairs

    result = export_preference_pairs(tmp_path, **ready_kwargs())

    assert result["plan_pairs"] == 1
    assert result["trajectory_pairs"] == 1
    plan_lines = (tmp_path / "plan_preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    trajectory_lines = (tmp_path / "trajectory_preference_pairs.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(plan_lines[0])["good"] == "good-plan"
    assert json.loads(trajectory_lines[0])["good"] == "good-traj"
