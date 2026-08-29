from __future__ import annotations

import pytest


def _request(prompt: str):
    from videoact.director_contracts import DirectorRequest

    return DirectorRequest(
        prompt=prompt,
        scene_id="self-provider-test",
        duration_s=4.0,
        fps=12,
        provider="codex-self",
        policy="director-v3-codex-self",
    )


def test_codex_self_director_keeps_raw_prompt_and_camera_cue_case_specific() -> None:
    from videoact.codex_self_provider import CodexSelfProvider
    from videoact.director import DirectorAgent

    provider = CodexSelfProvider()
    agent = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-self",
        policy="director-v3-codex-self",
    )

    zoom = agent.plan("Garden, zoom out.", scene_id="garden", duration_s=4.0, fps=12)
    pan = agent.plan("Laptop, pan right.", scene_id="laptop", duration_s=4.0, fps=12)

    assert zoom.director_plan.request.prompt == "Garden, zoom out."
    assert pan.director_plan.request.prompt == "Laptop, pan right."
    assert zoom.director_plan.content_hash() != pan.director_plan.content_hash()
    assert zoom.camera_plan.shots[0].trajectory_type == "dolly"
    assert zoom.camera_plan.shots[0].camera_cue == "zoom"
    assert pan.camera_plan.shots[0].trajectory_type == "follow"
    assert pan.camera_plan.shots[0].camera_direction == "right"
    assert zoom.director_plan.evidence


def test_codex_self_director_does_not_invent_a_person_for_object_only_prompt() -> None:
    from videoact.codex_self_provider import CodexLocalProvider
    from videoact.director import DirectorAgent

    provider = CodexLocalProvider()
    agent = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-local",
        policy="director-v3-codex-local",
    )

    result = agent.plan(
        "A water droplet slides down the edge of a smooth sheet of aluminum, maintaining its spherical form",
        scene_id="object-only-water-droplet",
        duration_s=4.0,
        fps=12,
    )

    assert not any(entity.kind == "actor" for entity in result.director_plan.entities)
    assert any(entity.label == "water droplet" for entity in result.director_plan.entities)
    assert any(entity.kind == "support" for entity in result.director_plan.entities)


def test_codex_self_director_preserves_compound_object_and_slide_motion() -> None:
    from videoact.codex_self_provider import CodexLocalProvider
    from videoact.director import DirectorAgent

    provider = CodexLocalProvider()
    agent = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-local",
        policy="director-v3-codex-local",
    )

    result = agent.plan(
        "A water droplet slides down the edge of a smooth sheet of aluminum, maintaining its spherical form",
        scene_id="object-only-slide",
        duration_s=4.0,
        fps=12,
    )

    assert any(entity.label == "water droplet" for entity in result.director_plan.entities)
    assert any(event.action == "move" for event in result.director_plan.events)
    assert not any(entity.kind == "actor" for entity in result.director_plan.entities)


def test_codex_self_director_expands_another_receiver_into_two_participants() -> None:
    from videoact.codex_self_provider import CodexLocalProvider
    from videoact.director import DirectorAgent

    provider = CodexLocalProvider()
    agent = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-local",
        policy="director-v3-codex-local",
    )

    result = agent.plan(
        "One person passes a ball to another.",
        scene_id="two-person-handoff",
        duration_s=10.0,
        fps=12,
    )

    actors = [entity for entity in result.director_plan.entities if entity.kind == "actor"]
    handoff = next(event for event in result.director_plan.events if event.action == "handoff")

    assert [entity.id for entity in actors] == ["actor_a", "actor_b"]
    assert handoff.participant_ids == ["actor_a", "actor_b"]
    assert any(shot.target_ids == ["actor_a", "actor_b", "prop_01_ball"] for shot in result.camera_plan.shots)
    assert result.director_trajectories.entities["actor_a"].states[-1].position != result.director_trajectories.entities["actor_b"].states[-1].position
    assert result.director_trajectories.entities["prop_01_ball"].states[-1].position != result.director_trajectories.entities["prop_01_ball"].states[0].position


def test_codex_self_codegen_is_case_specific_and_static_gate_clean() -> None:
    from blender.lib.__meta__ import collect_library_signatures
    from videoact.blender_code_agent import BlenderCodeAgent, validate_generated_source
    from videoact.codegen_contracts import CodegenRequest, FunctionSignature
    from videoact.codex_self_provider import CodexSelfProvider
    from videoact.director import DirectorAgent

    provider = CodexSelfProvider()
    director = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-self",
        policy="director-v3-codex-self",
    )
    signatures = {
        category: [FunctionSignature.model_validate(item) for item in entries]
        for category, entries in collect_library_signatures().items()
    }
    code_agent = BlenderCodeAgent(provider=provider.codegen, model="codex-self")
    sources = []
    for scene_id, prompt in (("garden", "Garden, zoom out."), ("laptop", "Laptop, pan right.")):
        plan = director.plan(prompt, scene_id=scene_id, duration_s=4.0, fps=12)
        request = CodegenRequest(
            director_plan=plan.director_plan.model_dump(mode="json"),
            library_signatures=signatures,
            harness_version="codex-self-test-v1",
            constraints=["preserve_prompt_entities_and_events", "emit_telemetry_and_sample_frames"],
        )
        response = code_agent.generate(request)
        assert response.status == "success"
        assert response.llm_call_id.startswith("codex-self:")
        assert not any(token in response.generated_code for token in ("compile_real_proxy_job", "direct_prompt_code", "real_proxy_job"))
        assert validate_generated_source(
            response.generated_code,
            allowed_library_calls=request.available_library_calls,
        ) == []
        sources.append(response.generated_code)
    assert sources[0] != sources[1]


def test_codex_self_provider_fails_closed_for_empty_or_uninterpretable_prompt() -> None:
    from videoact.codex_self_provider import CodexSelfProvider

    provider = CodexSelfProvider()
    with pytest.raises(ValueError, match="cannot derive executable scene subject"):
        provider.director(_request("..."))


def test_codex_local_codegen_stamps_local_provider_provenance() -> None:
    from videoact.codex_self_provider import CodexLocalProvider
    from videoact.director import DirectorAgent

    provider = CodexLocalProvider()
    director = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-local",
        policy="director-v3-codex-local",
    )
    plan = director.plan("A laptop slides right across a desk.", scene_id="local", duration_s=4.0, fps=12)
    response = provider.codegen(
        {
            "model": "codex-local",
            "director_plan": plan.director_plan.model_dump(mode="json"),
        }
    )

    assert response["llm_call_id"].startswith("codex-local:")
    assert "CODEX_PROVIDER = 'codex-local'" in response["generated_code"]
    assert '"provider": CODEX_PROVIDER' in response["generated_code"]
    assert '"provider_variant": CODEX_VARIANT' in response["generated_code"]


def test_codex_local_codegen_emits_case_specific_visual_profile_and_rig_strategy() -> None:
    from videoact.codex_self_provider import CodexLocalProvider
    from videoact.director import DirectorAgent

    provider = CodexLocalProvider()
    director = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-local",
        policy="director-v3-codex-local",
    )
    prompts = (
        "A person pours tea from a ceramic cup into a glass on a kitchen counter.",
        "A red apple rolls beside a wooden book on a garden table.",
    )
    sources = []
    profiles = []
    for index, prompt in enumerate(prompts):
        plan = director.plan(prompt, scene_id=f"profile-{index}", duration_s=4.0, fps=12)
        response = provider.codegen(
            {
                "model": "codex-local",
                "director_plan": plan.director_plan.model_dump(mode="json"),
            }
        )
        source = response["generated_code"]
        sources.append(source)
        profiles.append(source.split("CASE_SCENE_PROFILE = ", 1)[1].split("\n", 1)[0])

    assert sources[0] != sources[1]
    assert profiles[0] != profiles[1]
    assert "create_connected_humanoid_rig" in sources[0]
    assert "geometry_style" in sources[0]
    assert "pose_actor_for_event" in sources[0]
