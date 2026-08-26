from evaluator.deterministic import DeterministicReport
from evaluator.schemas import VLMJudgeResponse


def vlm_all(value=100):
    return VLMJudgeResponse(
        prompt_compliance=value,
        physical_plausibility=value,
        camera_coverage=value,
        camera_innovation=value,
        character_trajectory=value,
        object_trajectory=value,
        event_timing=value,
        temporal_smoothness=value,
        visual_clarity=value,
        visible_evidence=["test evidence"],
        weaknesses=[],
        confidence=1.0,
    )


def test_score_aggregation_uses_documented_weights_and_blends_sources():
    from evaluator.aggregate import aggregate_scores

    deterministic = DeterministicReport(
        terminal_status="pass",
        hard_gate_failed=False,
        score=80,
        findings=[],
    )
    aggregate = aggregate_scores(deterministic, vlm_all(100))

    assert aggregate.vlm_score == 100
    assert aggregate.uncapped_score == 96
    assert aggregate.final_score == 96
    assert aggregate.cap_reason is None


def test_hard_gate_caps_score_and_preserves_uncapped_value():
    from evaluator.aggregate import aggregate_scores

    deterministic = DeterministicReport(
        terminal_status="fail",
        hard_gate_failed=True,
        score=98,
        findings=[],
    )
    aggregate = aggregate_scores(deterministic, vlm_all(100))

    assert aggregate.uncapped_score == 99.6
    assert aggregate.final_score == 49
    assert aggregate.cap_reason == "hard_gate"


def test_grouped_geometric_vlm_score_penalizes_camera_trajectory_failure_without_clarity_dominance():
    from evaluator.aggregate import grouped_vlm_score

    balanced = vlm_all(80)
    camera_failure = balanced.model_copy(
        update={
            "camera_coverage": 20,
            "camera_innovation": 20,
            "character_trajectory": 40,
        }
    )
    clarity_failure = balanced.model_copy(update={"visual_clarity": 20})

    balanced_score = grouped_vlm_score(balanced)
    camera_score = grouped_vlm_score(camera_failure)
    clarity_score = grouped_vlm_score(clarity_failure)

    assert balanced_score == 80
    assert camera_score < 70
    assert clarity_score > camera_score
    assert clarity_score > 70
