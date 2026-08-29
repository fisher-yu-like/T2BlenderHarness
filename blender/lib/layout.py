"""Pure layout and interaction helpers."""

from __future__ import annotations

from typing import Mapping

from . import register_primitive
from .constraints import ConstraintSpec, child_of_constraint


Vec3 = tuple[float, float, float]


@register_primitive(category="layout", tags=["lanes", "collision"], cost_estimate="low", example_usage="lane_separated_paths(paths, 1.5)")
def lane_separated_paths(
    trajectories: Mapping[str, list[tuple[int, Vec3]]],
    min_distance: float = 1.5,
) -> dict[str, list[tuple[int, Vec3]]]:
    """Adjust only Y so same-frame paths have at least ``min_distance`` separation."""

    if min_distance <= 0:
        raise ValueError("min_distance must be positive")
    names = sorted(trajectories)
    adjusted: dict[str, list[tuple[int, Vec3]]] = {}
    for index, name in enumerate(names):
        states = trajectories[name]
        if any(states[position][0] > states[position + 1][0] for position in range(len(states) - 1)):
            raise ValueError(f"trajectory frames must be ordered: {name}")
        offset = (index - (len(names) - 1) / 2.0) * min_distance
        adjusted[name] = [
            (frame, (float(position[0]), float(position[1]) + offset, float(position[2])))
            for frame, position in states
        ]
    return adjusted


@register_primitive(category="layout", tags=["surface", "contact"], cost_estimate="low", example_usage="place_on_surface(((-1, -1, -0.5), (1, 1, 0.5)), 0)")
def place_on_surface(obj_bounds: tuple[Vec3, Vec3], surface_z: float, margin: float = 0.0) -> Vec3:
    """Return a translation that places the object's lower bound on a surface."""

    low, _high = obj_bounds
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return (0.0, 0.0, float(surface_z) - float(low[2]) + float(margin))


@register_primitive(category="layout", tags=["handoff", "attachment"], cost_estimate="low", example_usage="handoff_constraint_sequence('cup', 'giver.hand.R', 'receiver.hand.L', 20)")
def handoff_constraint_sequence(
    prop_id: str,
    giver_bone: str,
    receiver_bone: str,
    handoff_frame: int,
    *,
    transition_frames: int = 3,
) -> list[ConstraintSpec]:
    """Generate synchronized giver/receiver Child Of influence curves."""

    if not prop_id or "." not in giver_bone or "." not in receiver_bone:
        raise ValueError("bone references must look like 'rig.hand.R'")
    if handoff_frame <= 0 or transition_frames <= 0 or handoff_frame < transition_frames:
        raise ValueError("handoff frame and transition must be positive and ordered")
    giver_target, giver_subtarget = giver_bone.split(".", 1)
    receiver_target, receiver_subtarget = receiver_bone.split(".", 1)
    start = handoff_frame - transition_frames
    return [
        child_of_constraint(prop_id, giver_target, giver_subtarget, [(start, 1.0), (handoff_frame, 0.0)]),
        child_of_constraint(prop_id, receiver_target, receiver_subtarget, [(start, 0.0), (handoff_frame, 1.0)]),
    ]


@register_primitive(category="layout", tags=["collision", "obstacle"], cost_estimate="medium", example_usage="avoid_penetration('actor', ['wall'], trajectory, obstacle_bounds=bounds)")
def avoid_penetration(
    moving_obj: str,
    static_obstacles: list[str],
    trajectory: list[tuple[int, Vec3]],
    safety_margin: float = 0.2,
    *,
    obstacle_bounds: Mapping[str, tuple[Vec3, Vec3]] | None = None,
) -> list[tuple[int, Vec3]]:
    """Move points that enter an obstacle to its positive-Y safe side."""

    del moving_obj
    if safety_margin < 0:
        raise ValueError("safety_margin must be non-negative")
    bounds = obstacle_bounds or {}
    result: list[tuple[int, Vec3]] = []
    for frame, position in trajectory:
        x, y, z = (float(item) for item in position)
        for obstacle in static_obstacles:
            low, high = bounds.get(obstacle, ((float("-inf"), float("-inf"), float("-inf")), (float("inf"), float("inf"), float("inf"))))
            if low[0] <= x <= high[0] and low[1] <= y <= high[1] and low[2] <= z <= high[2]:
                y = high[1] + safety_margin
        result.append((int(frame), (x, y, z)))
    return result


__all__ = ["avoid_penetration", "handoff_constraint_sequence", "lane_separated_paths", "place_on_surface"]

