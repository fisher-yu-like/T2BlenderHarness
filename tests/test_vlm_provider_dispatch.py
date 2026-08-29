from __future__ import annotations

import json
from pathlib import Path

import pytest


def _valid_review() -> dict[str, object]:
    return {
        "prompt_compliance": 71,
        "physical_plausibility": 63,
        "camera_coverage": 78,
        "camera_innovation": 66,
        "character_trajectory": 59,
        "object_trajectory": 61,
        "event_timing": 57,
        "temporal_smoothness": 74,
        "visual_clarity": 69,
        "appearance_detail": 42,
        "physical_realism": 51,
        "spatial_consistency": 64,
        "motion_naturalness": 55,
        "visual_presentation": 47,
        "visible_evidence": ["frame 1 shows both named entities"],
        "weaknesses": ["the proxy lacks fine material detail"],
        "confidence": 0.82,
    }


def test_two_local_model_adapters_build_multiframe_schema_payloads(tmp_path: Path):
    from evaluator.vlm_providers import PROVIDERS
    from evaluator.schemas import VLMJudgeResponse

    frames = []
    for index in range(3):
        path = tmp_path / f"frame_{index:03d}.png"
        path.write_bytes(f"png-{index}".encode("ascii"))
        frames.append(path)

    assert {"gpt-5.6-luna", "gpt-5.6-terra"}.issubset(PROVIDERS)
    for model_name in ("gpt-5.6-luna", "gpt-5.6-terra"):
        adapter = PROVIDERS[model_name]
        payload = adapter.build_payload(
            prompt="Priya hands the ceramic mug to Wei.",
            frame_paths=frames,
            scene_contract={"entities": ["Priya", "Wei", "ceramic mug"]},
            deterministic_findings=[{"failure_id": "none"}],
        )
        assert payload["model"] == model_name
        assert payload["messages"][0]["role"] == "system"
        content = payload["messages"][1]["content"]
        assert sum(item["type"] == "image_url" for item in content) == 3
        assert "json_schema" in json.dumps(payload)

        parsed = adapter.parse_response({"choices": [{"message": {"content": json.dumps(_valid_review())}}]})
        assert isinstance(parsed, VLMJudgeResponse)
        assert parsed.object_trajectory == 61


def test_provider_dispatch_is_fail_closed_on_transport_error(tmp_path: Path):
    from evaluator.vlm_providers import dispatch_vlm

    frame = tmp_path / "frame.png"
    frame.write_bytes(b"png")

    def failing_transport(*args, **kwargs):
        raise RuntimeError("endpoint unavailable")

    result = dispatch_vlm(
        model="gpt-5.6-luna",
        prompt="A person walks past a table.",
        frame_paths=[frame],
        scene_contract={},
        deterministic_findings=[],
        transport=failing_transport,
    )

    assert result["status"] == "unavailable"
    assert result["score"] is None
    assert result["reason"] == "transport_error"


def test_adapter_accepts_upstream_nested_scores_envelope():
    from evaluator.vlm_providers import PROVIDERS

    review = _valid_review()
    nested = {
        "scores": {name: review[name] for name in review if name not in {"visible_evidence", "weaknesses", "confidence"}},
        "visible_evidence": review["visible_evidence"],
        "weaknesses": review["weaknesses"],
        "confidence": review["confidence"],
    }

    parsed = PROVIDERS["gpt-5.6-luna"].parse_response(
        {"choices": [{"message": {"content": json.dumps(nested)}}]}
    )

    assert parsed.prompt_compliance == 71
    assert parsed.motion_naturalness == 55
    assert parsed.confidence == 0.82


def test_default_provider_honors_training_model_environment(monkeypatch):
    from evaluator.vlm_providers import OpenAICompatibleVLMProvider

    monkeypatch.setenv("OPENAI_VLM_MODEL", "gpt-5.6-terra")
    provider = OpenAICompatibleVLMProvider()

    assert provider.model == "gpt-5.6-terra"
    assert provider.model_alias == "gpt-5.6-terra"
