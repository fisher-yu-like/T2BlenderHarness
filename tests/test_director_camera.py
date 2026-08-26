from __future__ import annotations


def _director_parts(prompt: str):
    from videoact.director_contracts import DirectorRequest
    from videoact.director_prompt import DeterministicPromptInterpreter
    from videoact.director_schedule import EventScheduler
    from videoact.director_trajectory import MultiEntityTrajectoryComposer

    request = DirectorRequest(
        prompt=prompt,
        scene_id="camera-test",
        duration_s=12.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )
    interpretation = DeterministicPromptInterpreter().interpret(request)
    schedule = EventScheduler().schedule(request, interpretation)
    trajectories = MultiEntityTrajectoryComposer().compose(request, interpretation, schedule)
    return request, interpretation, schedule, trajectories


def test_camera_composes_handoff_two_shot_with_visibility_and_axis_continuity():
    from videoact.director_camera import MultiTargetCameraChoreographer

    request, interpretation, schedule, trajectories = _director_parts(
        "Alice carries the red cube while Bob carries the blue cube, "
        "then Alice hands the red cube to Bob, then Bob places the red cube."
    )
    camera = MultiTargetCameraChoreographer().compose(request, interpretation, schedule, trajectories)

    handoff_shot = next(shot for shot in camera.shots if "handoff_actor_a_actor_b_red_cube" in shot.required_event_ids)
    assert set(handoff_shot.target_ids) >= {"actor_a", "actor_b", "red_cube"}
    assert handoff_shot.max_occlusion <= 0.25
    assert handoff_shot.visibility_predicates["actor_a"] == "visible"
    assert handoff_shot.visibility_predicates["actor_b"] == "visible"
    assert handoff_shot.visibility_predicates["red_cube"] == "visible"
    assert handoff_shot.continuity_group == "axis_a"
    assert "handoff two-shot" in handoff_shot.intent
    assert handoff_shot.innovation_intent_evidence_id.startswith("ev_camera_")


def test_camera_covers_concurrent_actions_and_all_must_show_events():
    from videoact.director_camera import MultiTargetCameraChoreographer

    request, interpretation, schedule, trajectories = _director_parts(
        "Alice carries the red cube while Bob carries the blue cube, then Alice hands the red cube to Bob."
    )
    camera = MultiTargetCameraChoreographer().compose(request, interpretation, schedule, trajectories)

    covered_events = {event_id for shot in camera.shots for event_id in shot.required_event_ids}
    assert covered_events == {event.id for event in schedule.events}
    concurrent_shot = camera.shots[0]
    assert set(concurrent_shot.required_event_ids) == {"carry_actor_a_red_cube", "carry_actor_b_blue_cube"}
    assert set(concurrent_shot.target_ids) == {"actor_a", "actor_b", "red_cube", "blue_cube"}
    assert concurrent_shot.trajectory_type in {"follow", "orbit", "dolly"}
    assert all(shot.continuity_group == "axis_a" for shot in camera.shots)
