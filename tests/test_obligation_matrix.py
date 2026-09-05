from __future__ import annotations

import pytest


def _record(obligation_id: str, *, kind: str = "event", applicable: bool = True):
    from videoact.obligations import ObligationRecord

    return ObligationRecord(
        obligation_id=obligation_id,
        case_id="train-matrix-01",
        kind=kind,
        required=applicable,
        applicable=applicable,
        expected={"obligation_id": obligation_id},
        pass_rule="fixture evidence satisfies the obligation",
        evidence_sources=["fixture"],
    )


def test_matrix_reports_planned_as_first_divergence_when_receiver_is_missing() -> None:
    from videoact.obligation_matrix import build_obligation_matrix

    matrix = build_obligation_matrix(
        [_record("receiver")],
        planned=set(),
        implemented={"receiver"},
        executed={"receiver"},
        visible={"receiver"},
        judged={"receiver"},
    )

    row = matrix.rows[0]
    assert row.first_divergence_stage == "planned"
    assert row.primary_root_cause_id == "obligation_planning"
    assert row.owner_candidate == "director_event_scheduler"
    assert len(matrix.primary_failures) == 1


@pytest.mark.parametrize(
    ("missing_stage", "expected_owner"),
    [
        ("implemented", "blender_code_agent"),
        ("executed", "interaction_library"),
        ("visible", "director_camera"),
    ],
)
def test_matrix_selects_only_the_first_missing_downstream_stage(missing_stage: str, expected_owner: str) -> None:
    from videoact.obligation_matrix import build_obligation_matrix

    all_stages = {"planned", "implemented", "executed", "visible", "judged"}
    values = {stage: {"event-01"} for stage in all_stages}
    values[missing_stage] = set()
    matrix = build_obligation_matrix([_record("event-01", kind="ownership_transition")], **values)

    row = matrix.rows[0]
    assert row.first_divergence_stage == missing_stage
    assert row.owner_candidate == expected_owner
    assert len(matrix.primary_failures) == 1
    assert matrix.primary_failures[0]["obligation_id"] == "event-01"


def test_judge_disagreement_is_judged_divergence_not_an_upstream_failure() -> None:
    from videoact.obligation_matrix import build_obligation_matrix

    matrix = build_obligation_matrix(
        [_record("camera-visibility", kind="camera_visibility")],
        planned={"camera-visibility"},
        implemented={"camera-visibility"},
        executed={"camera-visibility"},
        visible={"camera-visibility"},
        judged={"camera-visibility": {"status": "disagreement", "evidence_refs": ["judge_a", "judge_b"]}},
    )

    row = matrix.rows[0]
    assert row.first_divergence_stage == "judged"
    assert row.primary_root_cause_id == "judge_disagreement"
    assert row.owner_candidate == "evaluator"
    assert row.evidence_refs == ["judge_a", "judge_b"]


def test_non_applicable_obligation_is_not_scored_as_a_pass() -> None:
    from videoact.obligation_matrix import build_obligation_matrix

    matrix = build_obligation_matrix(
        [_record("camera-motion-na", kind="camera_motion", applicable=False)],
        planned=set(),
        implemented=set(),
        executed=set(),
        visible=set(),
        judged=set(),
    )

    row = matrix.rows[0]
    assert row.status == "not_applicable"
    assert row.first_divergence_stage is None
    assert matrix.primary_failures == []
    assert "camera_motion" in matrix.na_dimensions


def test_matrix_deletion_fails_closed_and_fingerprint_is_stable() -> None:
    from videoact.obligation_matrix import (
        build_obligation_matrix,
        validate_obligation_matrix,
    )

    records = [_record("event-01"), _record("event-02")]
    kwargs = {
        "planned": {"event-01", "event-02"},
        "implemented": {"event-01", "event-02"},
        "executed": {"event-01", "event-02"},
        "visible": {"event-01", "event-02"},
        "judged": {"event-01", "event-02"},
    }
    first = build_obligation_matrix(records, **kwargs)
    second = build_obligation_matrix(records, **kwargs)
    assert first.fingerprint == second.fingerprint

    with pytest.raises(ValueError, match="missing obligation"):
        validate_obligation_matrix(first.model_copy(update={"rows": first.rows[:1]}), expected_ids=first.obligation_ids)
