from __future__ import annotations


def _interpret(prompt: str):
    from videoact.director_contracts import DirectorRequest
    from videoact.director_prompt import DeterministicPromptInterpreter

    request = DirectorRequest(
        prompt=prompt,
        scene_id="schedule-test",
        duration_s=12.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )
    return request, DeterministicPromptInterpreter().interpret(request)


def test_scheduler_builds_sequential_transfer_graph_and_lifecycle():
    from videoact.director_schedule import EventScheduler

    request, interpretation = _interpret(
        "Alice carries the red cube, then Alice hands the red cube to Bob, "
        "then Bob places the red cube."
    )
    schedule = EventScheduler().schedule(request, interpretation)

    events = {event.id: event for event in schedule.events}
    assert ["carry_actor_a_red_cube", "handoff_actor_a_actor_b_red_cube", "place_actor_b_red_cube"] == [
        event.id for event in schedule.events
    ]
    assert events["handoff_actor_a_actor_b_red_cube"].depends_on == ["carry_actor_a_red_cube"]
    assert events["place_actor_b_red_cube"].depends_on == ["handoff_actor_a_actor_b_red_cube"]
    assert all(0 <= event.start <= event.end <= request.duration_s for event in schedule.events)

    lifecycle = schedule.interactions[0]
    assert lifecycle.prop_id == "red_cube"
    assert lifecycle.giver_id == "actor_a"
    assert lifecycle.receiver_id == "actor_b"
    assert lifecycle.attach_event_id == "carry_actor_a_red_cube"
    assert lifecycle.transfer_event_id == "handoff_actor_a_actor_b_red_cube"
    assert lifecycle.detach_event_id == "place_actor_b_red_cube"
    assert lifecycle.final_owner_id == "actor_b"


def test_scheduler_groups_concurrent_independent_carrying_and_bounds_timing():
    from videoact.director_schedule import EventScheduler

    request, interpretation = _interpret(
        "Alice carries the red cube while Bob carries the blue cube, then Alice hands the red cube to Bob."
    )
    schedule = EventScheduler().schedule(request, interpretation)

    carries = [event for event in schedule.events if event.action == "carry"]
    assert {event.concurrency_group for event in carries} == {"while_01"}
    assert carries[0].start == carries[1].start
    assert carries[0].end == carries[1].end
    handoff = next(event for event in schedule.events if event.action == "handoff")
    assert set(handoff.depends_on) == {event.id for event in carries}


def test_scheduler_supports_pause_resume_return_and_acyclic_dependencies():
    from videoact.director_schedule import EventScheduler

    request, interpretation = _interpret(
        "Alice takes the red cube to Bob, pauses, then Bob returns the red cube to Alice."
    )
    schedule = EventScheduler().schedule(request, interpretation)

    actions = [event.action for event in schedule.events]
    assert actions == ["carry", "pause", "handoff", "return"]
    pause = next(event for event in schedule.events if event.action == "pause")
    return_event = next(event for event in schedule.events if event.action == "return")
    assert return_event.participant_ids == ["actor_b"]
    assert return_event.target_ids == ["red_cube"]
    assert return_event.depends_on == ["handoff_actor_a_actor_b_red_cube"]
    assert pause.depends_on == ["carry_actor_a_red_cube"]

    graph = {event.id: set(event.depends_on) for event in schedule.events}
    assert all(event_id not in dependencies for event_id, dependencies in graph.items())


def test_scheduler_honors_provider_supplied_event_ids_for_dataset_traceability():
    from videoact.director_contracts import DirectorDecisionEvidence, DirectorEntity, DirectorRequest
    from videoact.director_prompt import DirectorActionDirective, PromptInterpretation
    from videoact.director_schedule import EventScheduler

    request = DirectorRequest(
        prompt="Alice attaches the red cup.",
        scene_id="explicit-event-id",
        duration_s=8,
        fps=24,
        provider="codex-local",
        policy="director-v2",
    )
    interpretation = PromptInterpretation(
        request=request,
        entities=[
            DirectorEntity(id="actor_a", kind="actor", role="participant", label="Alice"),
            DirectorEntity(id="red_cup", kind="prop", role="target_object", label="red cup"),
        ],
        directives=[
            DirectorActionDirective(
                id="attach_01",
                action="attach",
                actor_id="actor_a",
                prop_id="red_cup",
                evidence_id="ev_action",
            )
        ],
        evidence=[DirectorDecisionEvidence(id="ev_action", source="prompt", claim="attach")],
    )

    schedule = EventScheduler().schedule(request, interpretation)

    assert [event.id for event in schedule.events] == ["attach_01"]


def test_scheduler_accepts_prop_only_benchmark_actions_as_self_participant():
    """A physical prop action must remain executable without inventing an actor."""

    from videoact.director_contracts import DirectorDecisionEvidence, DirectorEntity, DirectorRequest
    from videoact.director_prompt import DirectorActionDirective, PromptInterpretation
    from videoact.director_schedule import EventScheduler

    request = DirectorRequest(
        prompt="A rubber ball bounces on the hard floor.",
        scene_id="prop-only-event",
        duration_s=8,
        fps=24,
        provider="assistant-diagnostic",
        policy="director-v5",
    )
    interpretation = PromptInterpretation(
        request=request,
        entities=[
            DirectorEntity(id="ball_1", kind="prop", role="bouncing object", label="rubber ball"),
            DirectorEntity(id="floor_1", kind="support", role="contact surface", label="hard floor"),
        ],
        directives=[
            DirectorActionDirective(
                id="bounce_01",
                action="bounce",
                prop_id="ball_1",
                evidence_id="ev_bounce",
            )
        ],
        evidence=[DirectorDecisionEvidence(id="ev_bounce", source="prompt", claim="the ball bounces")],
    )

    schedule = EventScheduler().schedule(request, interpretation)

    assert len(schedule.events) == 1
    assert schedule.events[0].participant_ids == ["ball_1"]
    assert schedule.events[0].target_ids == ["ball_1"]


def test_scheduler_materializes_camera_only_prompt_as_observable_subject_event():
    from videoact.director_contracts import DirectorDecisionEvidence, DirectorEntity, DirectorRequest
    from videoact.director_prompt import DirectorCameraCue, PromptInterpretation
    from videoact.director_schedule import EventScheduler

    request = DirectorRequest(
        prompt="Vase, tilt down.",
        scene_id="camera-only-vase",
        duration_s=10.0,
        fps=12,
        provider="external-glm",
        policy="director-v5-glm-structured",
    )
    interpretation = PromptInterpretation(
        request=request,
        entities=[DirectorEntity(id="vase", kind="prop", role="subject", label="Vase")],
        directives=[],
        camera_cues=[
            DirectorCameraCue(
                id="cam_tilt_down",
                action="tilt",
                direction="down",
                evidence_id="ev_tilt_down",
            )
        ],
        evidence=[
            DirectorDecisionEvidence(
                id="ev_vase",
                source="prompt",
                prompt_span=(0, 4),
                quoted_text="Vase",
                claim="the vase is the visible subject",
            ),
            DirectorDecisionEvidence(
                id="ev_tilt_down",
                source="prompt",
                prompt_span=(6, 15),
                quoted_text="tilt down",
                claim="the camera tilts down",
            ),
        ],
    )

    schedule = EventScheduler().schedule(request, interpretation)

    assert len(schedule.events) == 1
    event = schedule.events[0]
    assert event.id == "camera_observe_cam_tilt_down"
    assert event.action == "observe"
    assert event.participant_ids == ["vase"]
    assert event.target_ids == ["vase"]
    assert event.start == 0.0
    assert event.end == 10.0
