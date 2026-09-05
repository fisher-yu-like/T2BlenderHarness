from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_glm_structured_provider_falls_back_to_openai_on_transport_failure():
    from videoact.external_structured_provider import (
        FallbackStructuredProvider,
        GLMStructuredProvider,
        OpenAICompatibleStructuredProvider,
    )

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    def glm_opener(_request, **_kwargs):
        raise OSError("GLM endpoint unavailable")

    def openai_opener(_request, **_kwargs):
        return _Response({"choices": [{"message": {"content": '{"ok": true}'}}]})

    primary = GLMStructuredProvider(
        response_schema=schema,
        prompt_builder=lambda payload: json.dumps(payload),
        api_key="glm-key",
        opener=glm_opener,
    )
    fallback = OpenAICompatibleStructuredProvider(
        response_schema=schema,
        prompt_builder=lambda payload: json.dumps(payload),
        api_key="openai-key",
        base_url="https://fallback.example/v1",
        opener=openai_opener,
    )
    provider = FallbackStructuredProvider(primary=primary, fallback=fallback)

    assert provider({"case_id": "case-a"}) == {"ok": True}
    assert provider.fallback_used is True
    assert [item["provider_kind"] for item in provider.call_records] == [
        "zhipu_glm_openai_compatible",
        "external_openai_compatible",
    ]
    assert provider.last_call()["provider_kind"] == "external_openai_compatible"


def test_fallback_manifest_retains_primary_failure_and_secondary_success():
    from videoact.external_structured_provider import (
        FallbackStructuredProvider,
        GLMStructuredProvider,
        OpenAICompatibleStructuredProvider,
    )
    from videoact.provider_provenance import ProviderManifest

    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}

    def glm_opener(_request, **_kwargs):
        raise OSError("GLM endpoint unavailable")

    def openai_opener(_request, **_kwargs):
        return _Response({"choices": [{"message": {"content": '{"ok": true}'}}]})

    provider = FallbackStructuredProvider(
        primary=GLMStructuredProvider(
            response_schema=schema,
            prompt_builder=lambda payload: json.dumps(payload),
            api_key="glm-key",
            opener=glm_opener,
        ),
        fallback=OpenAICompatibleStructuredProvider(
            response_schema=schema,
            prompt_builder=lambda payload: json.dumps(payload),
            api_key="openai-key",
            base_url="https://fallback.example/v1",
            opener=openai_opener,
        ),
    )

    assert provider({"case_id": "case-a"}) == {"ok": True}
    manifest = ProviderManifest(
        case_id="case-a",
        prompt="A red ball rolls.",
        provider_mode="glm",
        harness_version="harness-rsi-test",
    )
    for record in provider.call_records:
        manifest.add_record(record)
    payload = manifest.as_dict()

    assert payload["fallback_used"] is True
    assert payload["stages"]["director"]["call_count"] == 2
    assert payload["stages"]["director"]["provider_kinds"] == [
        "external_openai_compatible",
        "zhipu_glm_openai_compatible",
    ]
    assert payload["stages"]["director"]["calls"][0]["error"]


def test_glm_agent_wiring_keeps_glm_primary_and_openai_fallback(monkeypatch):
    from scripts.train_real_harness import build_dynamic_codex_agents
    from videoact.external_structured_provider import FallbackStructuredProvider

    monkeypatch.setenv("GLM_API_KEY", "glm-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    director, code_agent = build_dynamic_codex_agents(provider_mode="glm")

    assert isinstance(director.interpreter.provider, FallbackStructuredProvider)
    assert isinstance(code_agent.provider, FallbackStructuredProvider)
    assert director.interpreter.provider.primary.provider_kind == "zhipu_glm_openai_compatible"
    assert director.interpreter.provider.fallback.provider_kind == "external_openai_compatible"
    assert code_agent.provider.primary.provider_kind == "zhipu_glm_openai_compatible"
    assert code_agent.provider.fallback.provider_kind == "external_openai_compatible"


def test_codex_visual_review_provider_uses_read_only_codex_and_local_frames(monkeypatch, tmp_path):
    from evaluator.codex_visual import CodexVisualReviewProvider

    frame = tmp_path / "frame_0001.png"
    frame.write_bytes(b"not-a-real-png-for-transport-test")
    dimensions = {
        "prompt_compliance": 80,
        "physical_plausibility": 80,
        "camera_coverage": 80,
        "camera_innovation": 70,
        "character_trajectory": 80,
        "object_trajectory": 80,
        "event_timing": 75,
        "temporal_smoothness": 80,
        "visual_clarity": 80,
        "appearance_detail": 70,
        "physical_realism": 70,
        "spatial_consistency": 75,
        "motion_naturalness": 70,
        "visual_presentation": 75,
    }
    response_payload = {
        **dimensions,
        "visible_evidence": ["frame_0001.png shows the requested subject"],
        "weaknesses": ["single-frame diagnostic"],
        "confidence": 0.8,
    }
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["input"] = kwargs["input"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(response_payload),
            stderr="",
        )

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)
    provider = CodexVisualReviewProvider(command="codex-test", timeout_s=17)

    response, _raw = provider.evaluate(
        prompt="A red ball rolls.",
        scene_contract={},
        frame_paths=[frame],
        deterministic_findings=[],
    )

    assert response.prompt_compliance == 80
    assert provider.review_source == "codex_local_visual_review"
    assert "--sandbox" in captured["command"]
    assert "read-only" in captured["command"]
    assert "--model" in captured["command"]
    assert "model_reasoning_effort=\"low\"" in captured["command"]
    assert "--image" in captured["command"]
    assert str(frame.resolve()) in captured["command"]
    assert frame.name in str(captured["input"])
    assert "dimension_evidence" in str(captured["input"])


def test_codex_visual_review_provider_preserves_visual_frame_budget_in_clone():
    from evaluator.codex_visual import CodexVisualReviewProvider

    provider = CodexVisualReviewProvider(visual_frame_budget=6, fallback_timeout_s=11)
    clone = provider.clone()

    assert provider.visual_frame_budget == 6
    assert clone.visual_frame_budget == 6
    assert clone.fallback_timeout_s == 11


def test_codex_visual_review_fallback_preserves_visual_frame_budget(monkeypatch):
    from evaluator.codex_visual import CodexVisualReviewProvider
    from evaluator.openai_vlm import VLMUnavailable

    seen: list[tuple[str, int]] = []
    sentinel = object()

    def fake_evaluate_once(self, **_kwargs):
        seen.append((self._codex_model, self.visual_frame_budget))
        if len(seen) == 1:
            raise VLMUnavailable("primary unavailable")
        return sentinel, {}

    monkeypatch.setattr(CodexVisualReviewProvider, "_evaluate_once", fake_evaluate_once)
    provider = CodexVisualReviewProvider(
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
        visual_frame_budget=3,
        fallback_timeout_s=5,
    )

    response, _raw = provider.evaluate(
        prompt="A red ball rolls.",
        scene_contract={},
        frame_paths=[],
        deterministic_findings=[],
    )

    assert response is sentinel
    assert seen == [("gpt-5.6-luna", 3), ("gpt-5.6-terra", 3)]


def test_codex_visual_review_failure_circuit_is_shared_by_case_clones(monkeypatch):
    from evaluator.codex_visual import CodexVisualReviewProvider
    from evaluator.openai_vlm import VLMUnavailable

    calls: list[str] = []

    def failing_evaluate_once(self, **_kwargs):
        calls.append(self._codex_model)
        raise VLMUnavailable("local visual provider timed out")

    monkeypatch.setattr(CodexVisualReviewProvider, "_evaluate_once", failing_evaluate_once)
    provider = CodexVisualReviewProvider(
        model="gpt-5.6-luna",
        fallback_model="gpt-5.6-terra",
        fallback_timeout_s=5,
    )
    clone = provider.clone()

    with pytest.raises(VLMUnavailable, match="local visual provider timed out"):
        provider.evaluate(prompt="A red ball rolls.", frame_paths=[])
    with pytest.raises(VLMUnavailable, match="circuit_open"):
        clone.evaluate(prompt="A blue ball rolls.", frame_paths=[])

    assert calls == ["gpt-5.6-luna", "gpt-5.6-terra"]


def test_glm_director_prompt_treats_explicit_camera_subject_as_resolvable():
    from videoact.external_structured_provider import GLMStructuredProvider

    provider = GLMStructuredProvider.for_director(api_key="glm-key")
    request = SimpleNamespace(
        prompt="Garden, zoom in.",
        scene_id="camera-only",
        duration_s=10.0,
        fps=12,
        obligations={},
    )

    prompt = provider.prompt_builder(request)

    assert "explicit subject followed by a camera cue" in prompt
    assert "do not mark the target unresolved" in prompt


def test_codex_visual_review_provider_converts_process_failure_to_vlm_unavailable(monkeypatch, tmp_path):
    from evaluator.codex_visual import CodexVisualReviewProvider
    from evaluator.openai_vlm import VLMUnavailable

    frame = tmp_path / "frame_0001.png"
    frame.write_bytes(b"not-a-real-png")

    def failing_run(_command, **_kwargs):
        raise OSError("codex executable missing")

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", failing_run)
    provider = CodexVisualReviewProvider(command="codex-missing", timeout_s=17)

    with pytest.raises(VLMUnavailable, match="codex_visual_review_unavailable"):
        provider.evaluate(
            prompt="A red ball rolls.",
            scene_contract={},
            frame_paths=[frame],
            deterministic_findings=[],
        )


def test_codex_visual_review_provider_retries_with_bounded_fallback_model(monkeypatch, tmp_path):
    from evaluator.codex_visual import CodexVisualReviewProvider

    frame = tmp_path / "frame_0001.png"
    frame.write_bytes(b"not-a-real-png")
    dimensions = {
        name: 80
        for name in (
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
    }
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:
            raise OSError("primary Codex session failed")
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                **dimensions,
                "visible_evidence": ["frame_0001.png"],
                "weaknesses": [],
                "confidence": 0.8,
            }),
            stderr="",
        )

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)
    provider = CodexVisualReviewProvider(
        command="codex-test",
        timeout_s=17,
        fallback_model="gpt-5.6-terra",
        fallback_timeout_s=5,
    )

    response, raw = provider.evaluate(
        prompt="A red ball rolls.",
        scene_contract={},
        frame_paths=[frame],
        deterministic_findings=[],
    )

    assert response.confidence == 0.8
    assert len(calls) == 2
    assert "gpt-5.6-terra" in calls[1]
    assert raw["fallback_from_model"] == "gpt-5.6-luna"


def test_codex_schema_requires_finite_dimension_evidence_and_preserves_maps():
    from evaluator.codex_visual import _codex_response_schema
    from videoact.codex_exec_provider import _normalize_strict_schema

    schema = _normalize_strict_schema(_codex_response_schema())
    dimensions = schema["properties"]["dimension_evidence"]

    assert dimensions["required"] == [
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
    ]
    assert dimensions["additionalProperties"] is False
    event_scores = schema["properties"]["event_scores"]["anyOf"][0]
    assert isinstance(event_scores["additionalProperties"], dict)


def test_codex_visual_prompt_includes_event_timeline_context_for_evidence_refs(tmp_path):
    from evaluator.codex_visual import CodexVisualReviewProvider

    frame = tmp_path / "frame_000001.png"
    prompt = CodexVisualReviewProvider._build_prompt(
        {
            "prompt": "Garden, zoom in.",
            "frame_paths": [frame],
            "frame_timeline": [
                {"frame": 1, "time_s": 0.0, "path": str(frame.resolve())},
            ],
            "required_events": [
                {"id": "zoom_in", "action": "zoom", "start_frame": 1, "end_frame": 120},
            ],
        }
    )

    assert "frame_timeline" in prompt
    assert "time_s" in prompt
    assert "zoom_in" in prompt
    assert "start_frame" in prompt
    assert "observe/full-scene" in prompt


def test_codex_visual_review_lock_is_shared_by_cases_in_one_stream(tmp_path):
    from evaluator.codex_visual import visual_review_lock_path

    frame = tmp_path / "stream" / "jobs" / "case-a" / "case-a" / "frames" / "animation" / "frame_000001.png"

    assert visual_review_lock_path([frame]) == tmp_path / "stream" / "codex_visual_review.lock"
