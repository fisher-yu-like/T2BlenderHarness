from __future__ import annotations

import math

import pytest


def _scheduled_scene(prompt: str):
    from videoact.director_contracts import DirectorRequest
    from videoact.director_prompt import DeterministicPromptInterpreter
    from videoact.director_schedule import EventScheduler

    request = DirectorRequest(
        prompt=prompt,
        scene_id="trajectory-test",
        duration_s=12.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )
    interpretation = DeterministicPromptInterpreter().interpret(request)
    schedule = EventScheduler().schedule(request, interpretation)
    return request, interpretation, schedule


def _distance_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def test_composer_builds_per_entity_paths_lanes_and_prop_owner_lifecycle():
    from videoact.director_trajectory import MultiEntityTrajectoryComposer

    request, interpretation, schedule = _scheduled_scene(
        "Alice carries the red cube while Bob carries the blue cube, "
        "then Alice hands the red cube to Bob, then Bob places the red cube."
    )
    trajectories = MultiEntityTrajectoryComposer().compose(request, interpretation, schedule)

    assert set(trajectories.entities) == {"actor_a", "actor_b", "red_cube", "blue_cube"}
    actor_a = trajectories.entities["actor_a"]
    actor_b = trajectories.entities["actor_b"]
    red_cube = trajectories.entities["red_cube"]

    assert _distance_xy(actor_a.states[0].position, actor_b.states[0].position) >= 1.5
    assert all(primitive.type in {"arc", "s_curve", "zigzag", "bezier"} for primitive in actor_a.motion_primitives)
    # The handoff boundary releases the prop from the giver (detach) while
    # transferring ownership to the receiver, and the later place detaches it
    # onto the support surface; the interaction evaluator requires the full
    # attach/transfer/detach lifecycle.
    assert [event.action for event in red_cube.attachment_events] == ["attach", "transfer", "detach", "detach"]
    assert red_cube.attachment_events[0].object_id == "actor_a"
    assert red_cube.attachment_events[1].object_id == "actor_b"
    assert red_cube.attachment_events[2].object_id == "actor_a"
    assert red_cube.attachment_events[-1].object_id == "support_surface"
    assert red_cube.attachment_events[0].constraint_type == "child_of"
    assert red_cube.attachment_events[0].subtarget == "hand.R"
    assert red_cube.attachment_events[1].constraint_type == "child_of"
    assert red_cube.attachment_events[2].constraint_type == "child_of"
    assert red_cube.attachment_events[-1].constraint_type == "support_surface"
    assert trajectories.current_owner_by_event["carry_actor_a_red_cube:red_cube"] == "actor_a"
    assert trajectories.current_owner_by_event["handoff_actor_a_actor_b_red_cube:red_cube"] == "actor_b"
    assert trajectories.final_support_by_prop["red_cube"] == "support_surface"


def test_composer_fails_closed_when_actor_lanes_would_collide():
    from videoact.director_trajectory import MultiEntityTrajectoryComposer

    request, interpretation, schedule = _scheduled_scene(
        "Alice carries the red cube while Bob carries the blue cube."
    )
    with pytest.raises(ValueError, match="lane collision"):
        MultiEntityTrajectoryComposer(lane_spacing=0.0).compose(request, interpretation, schedule)
