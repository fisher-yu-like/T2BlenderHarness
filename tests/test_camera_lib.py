from __future__ import annotations

import math

import pytest

from blender.lib.camera import (
    CameraKeyframe,
    check_visibility,
    compute_framing_distance,
    dolly_camera,
    follow_camera,
    orbit_camera,
    reveal_from_occluder,
)


def test_orbit_camera_has_at_least_eight_arc_keyframes() -> None:
    keyframes = orbit_camera((0, 0, 1), 5.0, 0.0, 180.0, 2.0, (1, 24))

    assert len(keyframes) >= 8
    radii = [math.hypot(item.location[0], item.location[1]) for item in keyframes]
    assert max(radii) - min(radii) < 1e-6
    assert keyframes[0].frame == 1
    assert keyframes[-1].frame == 24


def test_follow_and_dolly_preserve_frame_order() -> None:
    target = [(1, (0, 0, 1)), (12, (2, 0, 1)), (24, (4, 0, 1))]
    follow = follow_camera(target, offset=(0, -5, 2))
    dolly = dolly_camera((0, -8, 3), (0, -4, 2), (0, 0, 1), (1, 24))

    assert [item.frame for item in follow] == [1, 12, 24]
    assert [item.frame for item in dolly] == [1, 24]
    assert follow[-1].target == (4, 0, 1)


def test_reveal_and_framing_are_bounded() -> None:
    reveal = reveal_from_occluder(((-1, -1, 0), (1, 1, 3)), (3, 0, 1), (1, 24))

    assert len(reveal) == 2
    assert compute_framing_distance([(0, 0, 0), (2, 0, 0)], 50.0) > 0
    with pytest.raises(ValueError):
        compute_framing_distance([], 50.0)


def test_visibility_checks_axis_aligned_occluder() -> None:
    assert not check_visibility((0, 0, 1), (3, 0, 1), [{"bounds": ((1, -1, 0), (2, 1, 2))}])
    assert check_visibility((0, 0, 1), (0, 3, 1), [{"bounds": ((1, -1, 0), (2, 1, 2))}])

