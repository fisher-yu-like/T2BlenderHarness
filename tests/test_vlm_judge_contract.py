import pytest
from pydantic import ValidationError


def valid_response():
    return {
        "prompt_compliance": 90,
        "physical_plausibility": 80,
        "camera_coverage": 70,
        "camera_innovation": 65,
        "character_trajectory": 85,
        "object_trajectory": 80,
        "event_timing": 75,
        "temporal_smoothness": 60,
        "visual_clarity": 50,
        "visible_evidence": ["grasp visible in closeup"],
        "weaknesses": ["camera transition is abrupt"],
        "confidence": 0.8,
    }


def test_vlm_response_requires_all_scored_dimensions_and_evidence():
    from evaluator.schemas import VLMJudgeResponse

    response = VLMJudgeResponse.model_validate(valid_response())

    assert response.prompt_compliance == 90
    assert response.confidence == 0.8


@pytest.mark.parametrize("field", ["prompt_compliance", "physical_plausibility", "camera_coverage", "temporal_smoothness", "visual_clarity"])
def test_vlm_response_rejects_scores_outside_zero_to_hundred(field):
    from evaluator.schemas import VLMJudgeResponse

    payload = valid_response()
    payload[field] = 101

    with pytest.raises(ValidationError):
        VLMJudgeResponse.model_validate(payload)


def test_judge_input_contains_evidence_but_not_harness_identity():
    from evaluator.vlm_judge import VLMJudge

    prepared = VLMJudge.prepare_input(
        prompt="Walk to table and pick up cup.",
        scene_contract={"scene_id": "scene-1", "events": [{"id": "grasp"}]},
        selected_frames=["frame-001.png"],
        deterministic_findings=[{"failure_id": "none"}],
        harness_version="secret-harness-version",
    )

    assert prepared["prompt"] == "Walk to table and pick up cup."
    assert prepared["selected_frames"] == ["frame-001.png"]
    assert "harness_version" not in prepared
    assert "secret-harness-version" not in str(prepared)


def test_real_vlm_provider_defaults_to_project_model():
    from evaluator.openai_vlm import OpenAIVLMProvider, canonical_vlm_name, normalize_vlm_model

    assert OpenAIVLMProvider(api_key="test-key").model == "gpt-5.6-luna"
    assert normalize_vlm_model("gpt-5.6-Terra") == "gpt-5.6-terra"
    assert canonical_vlm_name("gpt-5.6-luna") == "gpt-5.6-luna"
    assert canonical_vlm_name("gpt-5.6-Terra") == "gpt-5.6-terra"
    assert OpenAIVLMProvider(api_key="test-key", model="gpt-5.6-luna").model_alias == "gpt-5.6-luna"
