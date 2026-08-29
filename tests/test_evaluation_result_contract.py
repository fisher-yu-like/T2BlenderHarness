from __future__ import annotations


def test_missing_real_video_evidence_invalidates_execution_and_all_scores() -> None:
    from evaluator.result_contract import build_evaluation_result

    result = build_evaluation_result(
        artifact_status="incomplete",
        video_probe={"playable": False},
        runtime_observation_count=0,
        visual_status="scored",
        semantic_score=88,
        observability_score=90,
        presentation_score=80,
        task_score=88,
        realism_score=75,
    )

    assert result.execution_status == "invalid"
    assert result.semantic_status == "uncertain"
    assert result.quality_status == "unavailable"
    assert result.semantic_score is None
    assert result.observability_score is None
    assert result.presentation_score is None
    assert result.task_score is None
    assert result.realism_score is None


def test_valid_video_with_unobserved_required_event_is_uncertain_not_failed_or_zero() -> None:
    from evaluator.result_contract import build_evaluation_result

    result = build_evaluation_result(
        artifact_status="complete",
        video_probe={"playable": True},
        runtime_observation_count=24,
        visual_status="needs_human_review",
        semantic_score=92,
        observability_score=78,
        presentation_score=65,
        task_score=84,
        realism_score=70,
        required_event_scores={"handoff_01": None},
    )

    assert result.execution_status == "valid"
    assert result.semantic_status == "uncertain"
    assert result.semantic_score == 92
    assert result.task_score == 84
    assert result.quality_status == "unavailable"
    assert result.presentation_score is None


def test_required_event_failure_caps_task_without_removing_presentation_score() -> None:
    from evaluator.result_contract import build_evaluation_result

    result = build_evaluation_result(
        artifact_status="complete",
        video_probe={"playable": True},
        runtime_observation_count=24,
        visual_status="scored",
        semantic_score=61,
        observability_score=80,
        presentation_score=73,
        task_score=61,
        realism_score=58,
        required_event_scores={"handoff_01": 12},
    )

    assert result.execution_status == "valid"
    assert result.semantic_status == "failed_required_event"
    assert result.task_score == 49
    assert result.semantic_score == 49
    assert result.presentation_score == 73
    assert result.quality_status == "scored"


def test_valid_no_event_case_can_have_scored_quality_without_inventing_event_score() -> None:
    from evaluator.result_contract import build_evaluation_result

    result = build_evaluation_result(
        artifact_status="complete",
        video_probe={"playable": True},
        runtime_observation_count=24,
        visual_status="scored",
        semantic_score=90,
        observability_score=82,
        presentation_score=77,
        task_score=88,
        realism_score=80,
        required_event_scores={},
    )

    assert result.execution_status == "valid"
    assert result.semantic_status == "passed"
    assert result.quality_status == "scored"
    assert result.task_score == 88
    assert result.realism_score == 80


def test_result_contract_keeps_required_event_status_separate_from_quality() -> None:
    from evaluator.result_contract import build_evaluation_result

    result = build_evaluation_result(
        artifact_status="complete",
        video_probe={"playable": True},
        runtime_observation_count=24,
        visual_status="scored",
        semantic_score=40,
        observability_score=80,
        presentation_score=70,
        task_score=45,
        realism_score=65,
        required_event_scores={"event_01": 10},
    )

    payload = result.model_dump(mode="json")
    assert payload["execution_status"] == "valid"
    assert payload["semantic_status"] == "failed_required_event"
    assert payload["quality_status"] == "scored"
    assert payload["task_score"] == 45
    assert payload["presentation_score"] == 70
