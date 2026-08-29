"""Pure camera choreography primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from . import register_primitive


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraKeyframe:
    frame: int
    location: Vec3
    target: Vec3


def _lerp(left: Vec3, right: Vec3, amount: float) -> Vec3:
    return tuple(left[index] + (right[index] - left[index]) * amount for index in range(3))  # type: ignore[return-value]


@register_primitive(category="camera", tags=["orbit", "arc", "handoff"], cost_estimate="low", example_usage="orbit_camera((0, 0, 1), 5, 0, 180, 2, (1, 24))")
def orbit_camera(
    center: Vec3,
    radius: float,
    start_angle_deg: float,
    end_angle_deg: float,
    height: float,
    frames: tuple[int, int],
    *,
    num_keyframes: int = 8,
) -> list[CameraKeyframe]:
    """Generate a circular arc with at least eight keyframes and a fixed target."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    if frames[1] <= frames[0]:
        raise ValueError("camera frame interval must increase")
    if num_keyframes < 8:
        raise ValueError("orbit requires at least 8 keyframes")
    cx, cy, cz = center
    result: list[CameraKeyframe] = []
    for index in range(num_keyframes):
        amount = index / (num_keyframes - 1)
        angle = math.radians(start_angle_deg + (end_angle_deg - start_angle_deg) * amount)
        frame = round(frames[0] + (frames[1] - frames[0]) * amount)
        result.append(
            CameraKeyframe(
                frame=frame,
                location=(cx + radius * math.cos(angle), cy + radius * math.sin(angle), cz + height),
                target=center,
            )
        )
    return result


@register_primitive(category="camera", tags=["follow", "tracking"], cost_estimate="low", example_usage="follow_camera([(1, (0, 0, 1))], (0, -5, 2))")
def follow_camera(target_trajectory: list[tuple[int, Vec3]], offset: Vec3, *, use_track_to: bool = True) -> list[CameraKeyframe]:
    """Follow a target trajectory with a constant camera offset."""

    del use_track_to  # The compiler chooses the concrete Blender constraint.
    if not target_trajectory:
        raise ValueError("target trajectory cannot be empty")
    if any(target_trajectory[index][0] > target_trajectory[index + 1][0] for index in range(len(target_trajectory) - 1)):
        raise ValueError("target frames must be ordered")
    return [
        CameraKeyframe(
            frame=int(frame),
            location=tuple(target[index] + offset[index] for index in range(3)),  # type: ignore[return-value]
            target=tuple(float(value) for value in target),
        )
        for frame, target in target_trajectory
    ]


@register_primitive(category="camera", tags=["dolly", "push-in"], cost_estimate="low", example_usage="dolly_camera((0, -8, 3), (0, -4, 2), (0, 0, 1), (1, 24))")
def dolly_camera(start: Vec3, end: Vec3, look_at: Vec3, frames: tuple[int, int]) -> list[CameraKeyframe]:
    """Move the camera linearly between two locations while tracking a target."""

    if frames[1] <= frames[0]:
        raise ValueError("camera frame interval must increase")
    return [CameraKeyframe(frames[0], start, look_at), CameraKeyframe(frames[1], end, look_at)]


@register_primitive(category="camera", tags=["reveal", "occlusion"], cost_estimate="low", example_usage="reveal_from_occluder(bounds, (3, 0, 1), (1, 24))")
def reveal_from_occluder(
    occluder_bounds: tuple[Vec3, Vec3],
    target: Vec3,
    frames: tuple[int, int],
    *,
    approach_direction: Vec3 = (1.0, 0.0, 0.0),
) -> list[CameraKeyframe]:
    """Start behind an occluder and move toward the target-facing side."""

    if frames[1] <= frames[0]:
        raise ValueError("camera frame interval must increase")
    dx, dy, dz = approach_direction
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        raise ValueError("approach direction must be non-zero")
    direction = (dx / length, dy / length, dz / length)
    low, high = occluder_bounds
    extent = math.sqrt(sum((high[index] - low[index]) ** 2 for index in range(3)))
    start = tuple(target[index] - direction[index] * max(1.0, extent) for index in range(3))  # type: ignore[return-value]
    end = tuple(target[index] - direction[index] * 0.45 for index in range(3))  # type: ignore[return-value]
    return [CameraKeyframe(frames[0], start, target), CameraKeyframe(frames[1], end, target)]


def _segment_intersects_box(start: Vec3, end: Vec3, low: Vec3, high: Vec3) -> bool:
    direction = tuple(end[index] - start[index] for index in range(3))
    t_min, t_max = 0.0, 1.0
    for index in range(3):
        if abs(direction[index]) < 1e-9:
            if start[index] < low[index] or start[index] > high[index]:
                return False
            continue
        inverse = 1.0 / direction[index]
        near = (low[index] - start[index]) * inverse
        far = (high[index] - start[index]) * inverse
        if near > far:
            near, far = far, near
        t_min, t_max = max(t_min, near), min(t_max, far)
        if t_min > t_max:
            return False
    return True


def check_visibility(camera_loc: Vec3, target: Vec3, occluders: list[dict[str, Any]], threshold: float = 0.8) -> bool:
    """Return false when the camera-to-target segment intersects too much occluder."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    hits = 0
    for occluder in occluders:
        bounds = occluder.get("bounds") if isinstance(occluder, dict) else None
        if bounds and _segment_intersects_box(camera_loc, target, bounds[0], bounds[1]):
            hits += 1
    return hits / max(1, len(occluders)) <= (1.0 - threshold)


def compute_framing_distance(targets: list[Vec3], fov_deg: float, margin: float = 1.2) -> float:
    """Compute a conservative distance that frames every target in a symmetric FOV."""

    if not targets:
        raise ValueError("at least one target is required")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError("fov must be between 0 and 180 degrees")
    if margin < 1.0:
        raise ValueError("margin must be >= 1")
    center = tuple(sum(target[index] for target in targets) / len(targets) for index in range(3))
    radius = max(math.dist(target, center) for target in targets)
    return margin * radius / math.tan(math.radians(fov_deg) / 2.0) + 0.1


__all__ = [
    "CameraKeyframe",
    "check_visibility",
    "compute_framing_distance",
    "dolly_camera",
    "follow_camera",
    "orbit_camera",
    "reveal_from_occluder",
]
