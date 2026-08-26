from __future__ import annotations

import pytest

from videoact.contracts import Finding


def _finding(*, case_id: str, owner: str = "director_camera", root: str = "camera:coverage"):
    return {
        "case_id": case_id,
        "split": "train",
        "findings": [
            Finding(
                failure_id="director_target_invisible",
                owner=owner,
                category="director_plan",
                severity="hard",
                message="target is not visible",
                root_cause_id=root,
                evidence=[f"{case_id}.json"],
                repair_route="camera_repair",
            )
        ],
    }


def _metrics(*, paired_train=True, paired_dev=True, artifact_before=1.0, artifact_after=1.0):
    return (
        {"train_score": 70.0, "dev_score": 68.0, "artifact_completion": artifact_before},
        {"train_score": 72.0, "dev_score": 68.0, "artifact_completion": artifact_after},
        {
            "hard_regression": False,
            "paired_train_improvement": paired_train,
            "paired_case_deltas": [1.0, 2.0] if paired_train else [1.0, -1.0],
        },
        {
            "hard_regression": False,
            "paired_dev_non_regression": paired_dev,
            "paired_case_deltas": [0.0, 1.0] if paired_dev else [0.0, -1.0],
        },
    )


def test_failure_groups_include_root_cause_and_distinct_cases():
    from videoact.evolution import aggregate_failures

    summary = aggregate_failures([_finding(case_id="case-01"), _finding(case_id="case-02")])

    assert summary.groups[0].root_cause_id == "camera:coverage"
    assert summary.groups[0].affected_case_ids == ["case-01", "case-02"]


def test_meta_requires_two_distinct_train_cases_for_a_patch(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    with pytest.raises(ValueError, match="two distinct train cases"):
        MetaHarnessOptimizer(output_dir=tmp_path).propose([_finding(case_id="case-01")])


def test_meta_rejects_mixed_owner_proposal(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    records = [
        _finding(case_id="case-01", owner="director_camera", root="camera:coverage"),
        _finding(case_id="case-02", owner="director_camera", root="camera:coverage"),
        _finding(case_id="case-03", owner="director_trajectory", root="trajectory:collision"),
        _finding(case_id="case-04", owner="director_trajectory", root="trajectory:collision"),
    ]
    with pytest.raises(ValueError, match="mixed-owner"):
        MetaHarnessOptimizer(output_dir=tmp_path).propose(records)


def test_acceptance_rejects_paired_train_regression_and_records_checks():
    from videoact.outer_loop import evaluate_candidate

    before, after, train, dev = _metrics(paired_train=False)
    decision = evaluate_candidate(before, after, train, dev)

    assert decision.accepted is False
    assert "paired_train" in decision.failed_checks
    assert decision.checks["overall_train_improved"] is True


def test_acceptance_rejects_artifact_regression_and_realism_task_tradeoff():
    from videoact.outer_loop import evaluate_candidate

    before, after, train, dev = _metrics(artifact_before=1.0, artifact_after=0.9)
    before.update(realism_score=60.0, task_score=80.0)
    after.update(realism_score=65.0, task_score=79.0)
    decision = evaluate_candidate(before, after, train, dev, owner="proxy_renderer")

    assert decision.accepted is False
    assert "artifact_completion" in decision.failed_checks
    assert "task_non_regression" in decision.failed_checks
