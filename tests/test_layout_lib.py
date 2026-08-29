from __future__ import annotations

from blender.lib.layout import (
    avoid_penetration,
    handoff_constraint_sequence,
    lane_separated_paths,
    place_on_surface,
)


def test_lane_separated_paths_preserves_x_and_z_and_separates_y() -> None:
    paths = {
        "a": [(1, (0.0, 0.0, 1.0)), (2, (1.0, 0.0, 1.0))],
        "b": [(1, (0.0, 0.0, 1.0)), (2, (1.0, 0.0, 1.0))],
    }

    adjusted = lane_separated_paths(paths, min_distance=1.5)

    assert adjusted["a"][0][1][0] == 0.0
    assert adjusted["a"][0][1][2] == 1.0
    assert abs(adjusted["a"][0][1][1] - adjusted["b"][0][1][1]) >= 1.5


def test_place_on_surface_and_handoff_sequence() -> None:
    assert place_on_surface(((-1, -1, -0.5), (1, 1, 0.5)), surface_z=0.0) == (0.0, 0.0, 0.5)
    constraints = handoff_constraint_sequence("cup", "giver.hand.R", "receiver.hand.L", 20, transition_frames=3)

    assert len(constraints) == 2
    assert constraints[0].influence_keyframes[-1] == (20, 0.0)
    assert constraints[1].influence_keyframes[0] == (17, 0.0)
    assert constraints[1].influence_keyframes[-1] == (20, 1.0)


def test_avoid_penetration_adjusts_overlapping_path_without_looping() -> None:
    adjusted = avoid_penetration(
        "actor",
        ["wall"],
        [(1, (0.0, 0.0, 0.0)), (2, (1.0, 0.0, 0.0))],
        safety_margin=0.2,
        obstacle_bounds={"wall": ((-0.5, -0.5, -1.0), (0.5, 0.5, 1.0))},
    )

    assert adjusted
    assert adjusted[0][1][1] >= 0.7
    assert adjusted[1][1][1] == 0.0
