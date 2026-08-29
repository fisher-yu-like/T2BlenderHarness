from __future__ import annotations

import math


def test_orbit_dsl_generates_arc_with_intermediate_keyframes():
    from blender.camera_dsl import orbit_points

    points = orbit_points(center=(0.0, 0.0, 1.0), radius=5.0, start_angle=0.0, end_angle=180.0, height=3.0, frame_count=10)

    assert len(points) == 10
    assert all(abs(math.dist((point[0], point[1], 1.0), (0.0, 0.0, 1.0)) - 5.0) < 1e-6 for point in points)
    assert points[0] != points[1]
    assert points[1] != points[-2]
    assert points[0][1] < 1e-6 and points[-1][1] > -1e-6


def test_camera_visibility_and_continuity_return_named_findings():
    from blender.camera_dsl import audit_camera_constraints

    findings = audit_camera_constraints(
        {"target": {"max_occlusion": 0.2}, "continuity_group": "axis-a"},
        {"target": {"occlusion": 0.7}, "continuity_group": "axis-b"},
    )

    assert {finding["failure_id"] for finding in findings} == {"camera_occlusion_exceeded", "camera_continuity_violation"}

