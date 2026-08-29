from __future__ import annotations


def _failure(case_id: str) -> dict:
    return {
        "case_id": case_id,
        "split": "train",
        "findings": [
            {
                "failure_id": "camera_event_uncovered",
                "owner": "camera_planner",
                "category": "camera_coverage",
                "severity": "error",
                "root_cause_id": "camera_required_event_coverage",
                "message": "the required handoff is not covered",
                "evidence": [f"{case_id}/deterministic_report.json"],
                "repair_route": "camera_repair",
            }
        ],
    }


def test_proposal_declares_falsifiable_case_predictions(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    proposal = MetaHarnessOptimizer(output_dir=tmp_path).propose([_failure("train-01"), _failure("train-02")])

    assert proposal.predicted_fixes == ["train-01", "train-02"]
    assert proposal.predicted_regressions == []
    assert "no known regression" in proposal.prediction_rationale
    assert set(proposal.predicted_fixes).issubset({"train-01", "train-02"})

