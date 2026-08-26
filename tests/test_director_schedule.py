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
