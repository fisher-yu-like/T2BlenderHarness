from __future__ import annotations

from evaluator.schemas import VLMJudgeResponse


def _response(**updates) -> VLMJudgeResponse:
    payload = {
        "prompt_compliance": 95,
        "physical_plausibility": 95,
        "camera_coverage": 95,
        "camera_innovation": 95,
        "character_trajectory": 95,
        "object_trajectory": 95,
        "event_timing": 95,
        "temporal_smoothness": 95,
        "visual_clarity": 95,
        "appearance_detail": 95,
        "physical_realism": 95,
        "spatial_consistency": 95,
        "motion_naturalness": 95,
        "visual_presentation": 95,
        "visible_evidence": ["frame 1", "frame 12"],
        "weaknesses": [],
        "confidence": 0.95,
    }
    payload.update(updates)
    return VLMJudgeResponse.model_validate(payload)


def _dimension_evidence():
    names = (
        "prompt_compliance",
        "physical_plausibility",
        "camera_coverage",
        "camera_innovation",
        "character_trajectory",
        "object_trajectory",
        "event_timing",
        "temporal_smoothness",
        "visual_clarity",
        "appearance_detail",
        "physical_realism",
        "spatial_consistency",
        "motion_naturalness",
        "visual_presentation",
    )
    return {
        name: {"confidence": 0.95, "evidence_completeness": 1.0, "evidence_refs": ["frame:1", "frame:12"]}
        for name in names
    }


def test_missing_dimension_evidence_has_no_fabricated_refs_and_wide_interval() -> None:
    from evaluator.evidence import build_dimension_evidence

    report = build_dimension_evidence(_response())

    item = report["dimensions"]["object_trajectory"]
    assert item["score"] == 95
    assert item["evidence_completeness"] == 0
    assert item["evidence_refs"] == []
    assert item["interval"][0] < 70
    assert item["interval"][1] == 100
    assert report["complete"] is False


def test_complete_dimension_evidence_preserves_high_score_without_a_hard_cap() -> None:
    from evaluator.evidence import build_dimension_evidence

    report = build_dimension_evidence(_response(dimension_evidence=_dimension_evidence()))

    assert report["complete"] is True
    assert report["mean_confidence"] == 0.95
    assert report["mean_evidence_completeness"] == 1.0
    assert report["dimensions"]["prompt_compliance"]["score"] == 95
    assert report["dimensions"]["prompt_compliance"]["evidence_refs"] == ["frame:1", "frame:12"]


def test_strict_visual_review_requires_dimension_evidence() -> None:
    from evaluator.visual_primary import score_visual_review

    result = score_visual_review(
        _response(),
        artifact_gate_pass=True,
        source="gpt-5.6-luna",
        strict_evidence=True,
    )

    assert result.status == "needs_human_review"
    assert result.reason == "dimension_evidence_incomplete"


def test_non_applicable_dimension_does_not_block_strict_evidence() -> None:
    from evaluator.visual_primary import score_visual_review

    evidence = _dimension_evidence()
    evidence.pop("character_trajectory")
    evidence.pop("camera_innovation")
    result = score_visual_review(
        _response(dimension_evidence=evidence),
        artifact_gate_pass=True,
        source="gpt-5.6-luna",
        applicability={"character_trajectory": False, "camera_motion": False},
        strict_evidence=True,
    )

    assert result.status == "scored"
    assert result.reason is None
    assert result.dimension_evidence["character_trajectory"]["applicability"] is False


def test_strict_evidence_reports_applicability_in_dimension_contract() -> None:
    from evaluator.evidence import build_dimension_evidence

    report = build_dimension_evidence(
        _response(dimension_evidence=_dimension_evidence()),
        applicability={"character_trajectory": False},
    )

    assert report["dimensions"]["character_trajectory"]["applicability"] is False
    assert report["complete"] is True


def test_confidence_is_a_gate_not_a_linear_score_weight() -> None:
    from evaluator.visual_primary import score_visual_review

    complete = _dimension_evidence()
    high = score_visual_review(
        _response(dimension_evidence=complete, confidence=0.9),
        artifact_gate_pass=True,
        source="gpt-5.6-luna",
        strict_evidence=True,
    )
    low = score_visual_review(
        _response(dimension_evidence=complete, confidence=0.5),
        artifact_gate_pass=True,
        source="gpt-5.6-luna",
        strict_evidence=True,
    )

    assert high.status == "scored"
    assert high.task_score == high.task_score
    assert low.status == "needs_human_review"
    assert low.task_score is None
