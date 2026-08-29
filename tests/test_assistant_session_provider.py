"""Tests for the assistant-session structured provider arm."""

from __future__ import annotations

import json

import pytest

from videoact.assistant_session_provider import (
    DEFAULT_MODEL_ID,
    PROVIDER_KIND,
    AssistantSessionProvider,
)
from videoact.director_contracts import DirectorRequest


def _director_request(tmp_path, scene_id="vbench2-train-99-01", prompt="One person passes a ball to another."):
    return DirectorRequest(
        prompt=prompt,
        scene_id=scene_id,
        duration_s=10.0,
        fps=12,
        provider="assistant-session-glm-flash",
        policy="director-v5-glm-structured",
        obligations={"required_entity_ids": [], "required_event_ids": [], "required_camera_event_ids": []},
    )


INTERPRETATION = {
    "entities": [
        {"id": "actor_a", "kind": "actor", "role": "participant", "label": "One person", "attributes": {}},
        {"id": "actor_b", "kind": "actor", "role": "participant", "label": "another", "attributes": {}},
        {"id": "prop_01_ball", "kind": "prop", "role": "target_object", "label": "ball", "attributes": {}},
    ],
    "directives": [
        {
            "id": "handoff_01",
            "action": "handoff",
            "actor_id": "actor_a",
            "prop_id": "prop_01_ball",
            "receiver_id": "actor_b",
            "evidence_id": "ev_action_01",
        }
    ],
    "camera_cues": [],
    "evidence": [
        {
            "id": "ev_action_01",
            "source": "prompt",
            "prompt_span": [3, 9],
            "quoted_text": "person",
            "claim": "a person participates in the handoff",
        }
    ],
    "assumptions": [],
    "uncertainties": [],
}


def _write_preauth(session_root, request, interpretation):
    preauth_dir = session_root / "preauth" / "director"
    preauth_dir.mkdir(parents=True, exist_ok=True)
    (preauth_dir / f"{request.scene_id}.json").write_text(
        json.dumps(
            {
                "scene_id": request.scene_id,
                "prompt": request.prompt,
                "response": dict(interpretation),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_provider_identity_and_preauth_exchange(tmp_path):
    session_root = tmp_path / "session"
    provider = AssistantSessionProvider.for_director(session_root=session_root, wait_timeout_s=0.2)
    assert provider.provider_kind == PROVIDER_KIND
    assert provider.model_id == DEFAULT_MODEL_ID
    assert provider.template_backed is False
    assert provider.llm_generated is True

    request = _director_request(tmp_path)
    _write_preauth(session_root, request, INTERPRETATION)
    result = provider(request)

    assert result["llm_call_id"].startswith("assistant:director:")
    assert result["generation_provenance"]["transport"] == "assistant_session_file_exchange"
    assert result["evidence"][0]["id"] == "ev_action_01"
    call = provider.last_call("director")
    assert call is not None
    assert call["error"] is None
    assert call["request_hash"] and call["response_hash"] and call["prompt_hash"]
    # The request and consumed preauth are materialized for provenance.
    assert list((session_root / "requests" / "director").glob("*.json"))
    assert list((session_root / "responses" / "director").glob("*.json"))


def test_missing_response_times_out_fail_closed(tmp_path):
    session_root = tmp_path / "session"
    provider = AssistantSessionProvider.for_director(
        session_root=session_root, wait_timeout_s=0.05, poll_interval_s=0.01
    )
    with pytest.raises(RuntimeError) as excinfo:
        provider(_director_request(tmp_path, scene_id="vbench2-train-99-02"))
    assert "assistant_response_timeout" in str(excinfo.value)
    call = provider.last_call("director")
    assert call is not None and call["error"]


def test_preauth_prompt_mismatch_is_not_consumed(tmp_path):
    session_root = tmp_path / "session"
    provider = AssistantSessionProvider.for_director(
        session_root=session_root, wait_timeout_s=0.05, poll_interval_s=0.01
    )
    request = _director_request(tmp_path, scene_id="vbench2-train-99-03")
    _write_preauth(session_root, request, INTERPRETATION)
    mismatched = _director_request(
        tmp_path, scene_id="vbench2-train-99-03", prompt="A different prompt entirely."
    )
    with pytest.raises(RuntimeError):
        provider(mismatched)


def test_codegen_exchange_binds_transport_provenance(tmp_path):
    session_root = tmp_path / "session"
    provider = AssistantSessionProvider.for_codegen(session_root=session_root, wait_timeout_s=0.2)
    payload = {
        "model": provider.model_id,
        "director_plan": {},
        "director_plan_hash": "a" * 64,
        "library_signatures": {},
        "context_examples": [],
        "harness_version": "test",
        "constraints": [],
    }
    responses_dir = session_root / "responses" / "blender_code"
    responses_dir.mkdir(parents=True, exist_ok=True)
    # Discover the pending request the exchange writes, then answer it from a
    # concurrent-responder perspective (what the driving agent session does).
    import threading

    def respond():
        requests_dir = session_root / "requests" / "blender_code"
        for _ in range(200):
            for request_path in sorted(requests_dir.glob("*.json")):
                response_path = responses_dir / request_path.name
                if not response_path.exists():
                    response_path.write_text(
                        json.dumps(
                            {
                                "status": "success",
                                "generated_code": "import bpy\nDIRECTOR_PLAN = {}",
                                "library_calls": [],
                            }
                        ),
                        encoding="utf-8",
                    )
                    return
            import time

            time.sleep(0.01)

    thread = threading.Thread(target=respond)
    thread.start()
    result = provider(payload)
    thread.join()
    assert result["status"] == "success"
    assert result["llm_call_id"].startswith("assistant:blender_code:")
    assert result["generation_provenance"]["response_file"]


def test_build_dynamic_codex_agents_assistant_arm(tmp_path, monkeypatch):
    from scripts.train_real_harness import build_dynamic_codex_agents

    monkeypatch.setenv("ASSISTANT_SESSION_ROOT", str(tmp_path / "session"))
    director, code_agent = build_dynamic_codex_agents(provider_mode="assistant")
    provider = director.interpreter.provider
    assert provider.provider_kind == PROVIDER_KIND
    assert provider.model_id == DEFAULT_MODEL_ID
    assert provider.template_backed is False
    assert provider.llm_generated is True
    assert code_agent.provider is not None
    assert code_agent.provider.provider_kind == PROVIDER_KIND
    assert code_agent.max_codegen_attempts == 2
