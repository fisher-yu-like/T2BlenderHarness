from __future__ import annotations

from evaluator.schemas import VLMJudgeResponse
from evaluator.visual_primary import score_visual_review


def _response(**updates) -> VLMJudgeResponse:
    payload = {
        "prompt_compliance": 100,
        "physical_plausibility": 100,
        "camera_coverage": 100,
        "camera_innovation": 100,
        "character_trajectory": 100,
        "object_trajectory": 100,
        "event_timing": 100,
        "temporal_smoothness": 100,
        "visual_clarity": 100,
        "appearance_detail": 100,
        "physical_realism": 100,
        "spatial_consistency": 100,
        "motion_naturalness": 100,
        "visual_presentation": 100,
        "visible_evidence": ["frame 1 shows the actor and cup"],
        "weaknesses": [],
        "confidence": 1.0,
    }
    payload.update(updates)
    return VLMJudgeResponse.model_validate(payload)


def test_visual_primary_keeps_task_and_realism_separate() -> None:
    result = score_visual_review(
        _response(),
        artifact_gate_pass=True,
        source="gpt-5.6-luna",
    )

    assert result.status == "scored"
    assert result.task_score == 100.0
    assert result.realism_score == 100.0
    assert result.overall_vlm_score == 100.0
    assert result.deterministic_score is None


def test_missing_semantic_evidence_reduces_task_score_without_changing_realism() -> None:
    result = score_visual_review(
        _response(prompt_compliance=0),
        artifact_gate_pass=True,
        source="gpt-5.6-terra",
    )

    assert result.status == "scored"
    assert result.task_score < 100.0
    assert result.realism_score == 100.0
    assert result.overall_vlm_score < 100.0


def test_artifact_gate_failure_skips_visual_score() -> None:
    result = score_visual_review(
        _response(),
        artifact_gate_pass=False,
        source="gpt-5.6-luna",
    )

    assert result.status == "skipped"
    assert result.task_score is None
    assert result.realism_score is None
    assert result.overall_vlm_score is None


def test_frame_statistics_and_low_confidence_never_become_numeric_scores() -> None:
    frame_result = score_visual_review(
        _response(),
        artifact_gate_pass=True,
        source="frame_statistics",
    )
    low_confidence = score_visual_review(
        _response(confidence=0.4),
        artifact_gate_pass=True,
        source="gpt-5.6-luna",
    )

    assert frame_result.status == "unavailable"
    assert frame_result.overall_vlm_score is None
    assert low_confidence.status == "needs_human_review"
    assert low_confidence.task_score is None
