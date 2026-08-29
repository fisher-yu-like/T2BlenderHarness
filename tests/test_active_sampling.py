from __future__ import annotations

import pytest


def _record(case_id, *, split="train", root="camera:coverage", confidence=0.9, disagreement=0.0, owner="camera_planner"):
    return {
        "case_id": case_id,
        "split": split,
        "prompt": "this must never be used by the sampler",
        "review_confidence": confidence,
        "judge_disagreement": disagreement,
        "findings": [{"root_cause_id": root, "owner": owner, "severity": "error"}],
    }


def test_active_sampler_prioritizes_uncertainty_disagreement_and_new_owner() -> None:
    from videoact.active_sampling import sample_failure_cases

    report = sample_failure_cases(
        [
            _record("easy", confidence=0.98),
            _record("uncertain", confidence=0.2),
            _record("disputed", confidence=0.8, disagreement=0.9),
            _record("new-owner", owner="new_owner", confidence=0.8),
        ],
        budget=2,
        known_owners={"camera_planner"},
        seed=7,
    )

    selected = [item["case_id"] for item in report["selected"]]
    assert selected[0] in {"uncertain", "disputed", "new-owner"}
    assert len(selected) == 2
    assert sum(item["sampling_probability"] for item in report["candidates"]) == pytest.approx(1.0)
    assert all(item["selection_reason"] for item in report["selected"])


def test_active_sampler_rejects_test_records_and_does_not_emit_prompt_text() -> None:
    from videoact.active_sampling import sample_failure_cases

    with pytest.raises(ValueError, match="test split"):
        sample_failure_cases([_record("test", split="test")], budget=1)

    report = sample_failure_cases([_record("train-1")], budget=1)
    assert "this must never be used by the sampler" not in str(report)


def test_sequential_stopping_is_conservative_near_the_margin() -> None:
    from videoact.active_sampling import sequential_stopping_decision

    assert sequential_stopping_decision([2.0] * 12, target_lower_bound=1.0, min_cases=10)["decision"] == "stop_success"
    assert sequential_stopping_decision([0.0] * 12, target_lower_bound=1.0, min_cases=10)["decision"] == "stop_failure"
    assert sequential_stopping_decision([1.0, 0.0], target_lower_bound=1.0, min_cases=10)["decision"] == "continue"


def test_active_sampling_replay_reports_render_reduction_and_decision_agreement() -> None:
    from videoact.active_sampling import audit_sampling_replay

    records = [
        {**_record("certain-1", confidence=0.99), "paired_delta": 2.0, "render_cost": 1},
        {**_record("certain-2", confidence=0.99), "paired_delta": 2.0, "render_cost": 1},
        {**_record("uncertain-1", confidence=0.1), "paired_delta": 2.0, "render_cost": 1},
        {**_record("uncertain-2", confidence=0.1), "paired_delta": 2.0, "render_cost": 1},
    ]

    report = audit_sampling_replay(
        [{"batch_id": "replay-1", "records": records}],
        budget=2,
        target_lower_bound=1.0,
        min_cases=2,
        min_reduction=0.30,
        min_agreement=0.95,
    )

    assert report["status"] == "pass"
    assert report["render_reduction"] == pytest.approx(0.5)
    assert report["decision_agreement"] == pytest.approx(1.0)
    assert report["batches"][0]["selected_case_ids"]
    assert "prompt" not in str(report)


def test_active_sampling_replay_fails_closed_on_decision_disagreement() -> None:
    from videoact.active_sampling import audit_sampling_replay

    records = [
        {**_record("high-confidence-positive", confidence=0.99), "paired_delta": 2.0},
        {**_record("high-confidence-positive-2", confidence=0.99), "paired_delta": 2.0},
        {**_record("uncertain-negative", confidence=0.1), "paired_delta": 0.0},
        {**_record("uncertain-negative-2", confidence=0.1), "paired_delta": 0.0},
    ]

    report = audit_sampling_replay(
        [{"batch_id": "replay-disagreement", "records": records}],
        budget=2,
        target_lower_bound=1.0,
        min_cases=2,
        min_reduction=0.30,
        min_agreement=0.95,
    )

    assert report["status"] == "fail"
    assert report["decision_agreement"] == pytest.approx(0.0)
    assert report["batches"][0]["decision_match"] is False


def test_active_sampling_replay_rejects_test_and_incomplete_batches() -> None:
    from videoact.active_sampling import audit_sampling_replay

    with pytest.raises(ValueError, match="test split"):
        audit_sampling_replay(
            [{"batch_id": "bad-test", "records": [_record("test", split="test", confidence=0.2)]}],
            budget=1,
            target_lower_bound=1.0,
        )

    with pytest.raises(ValueError, match="paired_delta"):
        audit_sampling_replay(
            [{"batch_id": "bad-schema", "records": [_record("missing-delta")]}],
            budget=1,
            target_lower_bound=1.0,
        )
