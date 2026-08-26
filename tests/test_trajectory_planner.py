import pytest


def pickup_contract():
    from videoact.scene_contract import SceneContractBuilder

    return SceneContractBuilder().build(
        "A character walks to the table, picks up the red cup, and shows the grasp closeup."
    )


def test_trajectory_planner_creates_frame_indexed_entity_states():
    from videoact.trajectory import TrajectoryPlanner

    plan = TrajectoryPlanner().plan(pickup_contract())

    assert plan.timebase.fps == 24
    assert plan.timebase.frame_start == 1
    assert plan.timebase.frame_end == 240
    assert set(plan.entities) == {"character", "table", "red_cup"}
    assert [state.frame for state in plan.entities["character"].states] == sorted(
        state.frame for state in plan.entities["character"].states
    )


def test_trajectory_planner_attaches_target_at_grasp_event():
    from videoact.trajectory import TrajectoryPlanner

    plan = TrajectoryPlanner().plan(pickup_contract())
    attachments = plan.entities["character"].attachment_events

    assert [(event.subject_id, event.object_id, event.action) for event in attachments] == [
        ("red_cup", "character", "attach")
    ]
    assert attachments[0].frame == 145


def test_camera_plan_covers_every_required_event_with_observability_predicates():
    from videoact.camera import CameraPlanner
    from videoact.trajectory import TrajectoryPlanner

    contract = pickup_contract()
    trajectory = TrajectoryPlanner().plan_entities(contract)
    camera = CameraPlanner().plan(contract, trajectory)

    covered = {
        event_id
        for shot in camera.shots
        for event_id in shot.required_event_ids
    }
    assert set(contract.must_show) <= covered
    assert all(shot.target_ids for shot in camera.shots)


def test_interpolation_helpers_are_continuous_and_bounded():
    from videoact.trajectory import interpolate_ease_in_out, interpolate_linear

    start = (0.0, 0.0, 0.0)
    end = (10.0, 5.0, 2.0)
    assert interpolate_linear(start, end, 0.0) == start
    assert interpolate_linear(start, end, 1.0) == end
    assert interpolate_ease_in_out(start, end, 0.5) == (5.0, 2.5, 1.0)
    with pytest.raises(ValueError, match="progress"):
        interpolate_linear(start, end, 1.1)


def test_discontinuous_motion_is_rejected():
    from videoact.trajectory import validate_continuity

    with pytest.raises(ValueError, match="discontinuity"):
        validate_continuity([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)], max_step=1.0)


def test_trajectory_planner_keeps_initial_state_when_events_are_absent():
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    plan = TrajectoryPlanner().plan(SceneContractBuilder().build("Observe a table."))

    assert plan.entities["character"].states[0].position == (0.0, 0.0, 0.0)
