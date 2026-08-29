"""Pure CameraPlan primitives and host-side constraint diagnostics."""

from __future__ import annotations

import math
from typing import Any


def orbit_points(
    *,
    center: tuple[float, float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    height: float,
    frame_count: int,
) -> list[tuple[float, float, float]]:
    if radius <= 0 or frame_count < 2:
        raise ValueError("orbit requires positive radius and at least two frames")
    step = (end_angle - start_angle) / (frame_count - 1)
    return [
        (
            center[0] + radius * math.cos(math.radians(start_angle + step * index)),
            center[1] + radius * math.sin(math.radians(start_angle + step * index)),
            height,
        )
        for index in range(frame_count)
    ]


def audit_camera_constraints(required: dict[str, Any], observed: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for target, requirement in required.items():
        if target == "continuity_group" or not isinstance(requirement, dict):
            continue
        observed_target = observed.get(target, {})
        maximum = float(requirement.get("max_occlusion", 1.0))
        actual = float(observed_target.get("occlusion", 1.0))
        if actual > maximum:
            findings.append({
                "failure_id": "camera_occlusion_exceeded",
                "owner": "director_camera",
                "category": "camera_coverage",
                "severity": "error",
                "message": f"target {target} occlusion {actual:.3f} exceeds {maximum:.3f}",
                "evidence": [target],
            })
    required_group = required.get("continuity_group")
    observed_group = observed.get("continuity_group")
    if required_group and observed_group and required_group != observed_group:
        findings.append({
            "failure_id": "camera_continuity_violation",
            "owner": "director_camera",
            "category": "camera_continuity",
            "severity": "error",
            "message": f"continuity group changed from {required_group} to {observed_group}",
            "evidence": [str(required_group), str(observed_group)],
        })
    return findings
