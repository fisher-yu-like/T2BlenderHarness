from __future__ import annotations

import math

from evaluator.schemas import VLMJudgeResponse


def _response(**updates) -> VLMJudgeResponse:
    payload = {
        "prompt_compliance": 80,
        "physical_plausibility": 70,
        "camera_coverage": 90,
        "camera_innovation": 0,
        "character_trajectory": 10,
        "object_trajectory": 60,
        "event_timing": 75,
        "temporal_smoothness": 85,
        "visual_clarity": 90,
        "appearance_detail": 80,
        "physical_realism": 80,
        "spatial_consistency": 80,
        "motion_naturalness": 80,
        "visual_presentation": 80,
        "visible_evidence": ["frame 1 shows the requested object"],
        "weaknesses": [],
        "confidence": 0.9,
    }
    payload.update(updates)
    return VLMJudgeResponse.model_validate(payload)


def test_static_camera_is_not_penalized_when_prompt_does_not_require_camera_motion() -> None:
    from evaluator.scoring_v7 import score_v7

    result = score_v7(
        _response(),
        applicability={"character_trajectory": False, "camera_motion": False},
        required_event_ids=[],
    )

    assert result.status == "scored"
    assert result.applicability["character_trajectory"] is False
    assert result.applicability["camera_innovation"] is False
    assert result.camera_effectiveness == 90
    assert result.task_score is not None
    expected_semantic = math.prod([80, 70, 60, 75]) ** 0.25
    expected_observability = math.sqrt(90 * 90)
    expected_task = 0.75 * expected_semantic + 0.25 * expected_observability
    assert result.task_score == round(expected_task, 4)
    expected_choreography = math.sqrt(90 * 85)
    assert result.choreography_score == round(expected_choreography, 4)


def test_camera_effectiveness_and_choreography_are_separate_channels() -> None:
    from evaluator.scoring_v7 import score_v7

    result = score_v7(
        _response(camera_coverage=80, camera_innovation=60, character_trajectory=70, temporal_smoothness=90),
        applicability={"camera_motion": True},
        required_event_ids=[],
    )

    assert result.camera_effectiveness == round(math.sqrt(80 * 60), 4)
    assert result.choreography_score == round(math.prod([80, 60, 70, 90]) ** 0.25, 4)
    assert result.choreography_score != result.observability_score


def test_required_event_evidence_below_25_caps_task_at_49() -> None:
    from evaluator.scoring_v7 import score_v7

    result = score_v7(
        _response(),
        applicability={"camera_motion": True},
        required_event_ids=["handoff_01"],
        required_event_scores={"handoff_01": 12},
    )

    assert result.status == "failed_required_event"
    assert result.required_event_gate_failed is True
    assert result.task_score == 49
    assert result.semantic_core is not None and result.semantic_core <= 49


def test_missing_required_event_evidence_is_uncertain_without_zero_imputation() -> None:
    from evaluator.scoring_v7 import score_v7

    result = score_v7(
        _response(),
        applicability={"camera_motion": True},
        required_event_ids=["handoff_01"],
        required_event_scores={"handoff_01": None},
    )

    assert result.status == "uncertain"
    assert result.task_score is not None
    assert result.task_score > 0
    assert result.required_event_gate_failed is False


def test_event_scores_are_evidence_bound_to_the_response_schema() -> None:
    from evaluator.scoring_v7 import score_v7

    result = score_v7(
        _response(event_scores={"handoff_01": 14}),
        required_event_ids=["handoff_01"],
    )

    assert result.status == "failed_required_event"
    assert result.required_event_scores == {"handoff_01": 14.0}


def test_non_applicable_dimensions_do_not_enter_geometric_mean() -> None:
    from evaluator.scoring_v7 import score_v7

    low_character = score_v7(
        _response(character_trajectory=0),
        applicability={"character_trajectory": False, "camera_motion": False},
        required_event_ids=[],
    )
    high_character = score_v7(
        _response(character_trajectory=100),
        applicability={"character_trajectory": False, "camera_motion": False},
        required_event_ids=[],
    )

    assert low_character.semantic_core == high_character.semantic_core
    assert low_character.task_score == high_character.task_score
