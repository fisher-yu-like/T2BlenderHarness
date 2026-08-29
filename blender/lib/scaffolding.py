"""Blender job runtime contract helpers.

These helpers are Blender-free.  Generated jobs use the contract values to
write telemetry and artifact manifests, while the host validates the actual
files with ``RealArtifactGate``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from . import register_primitive


RUNTIME_CONTRACT_VERSION = "runtime-contract-v1"
REQUIRED_REAL_ARTIFACTS = (
    "run_manifest.json",
    "scene_contract.json",
    "trajectory.json",
    "camera_plan.json",
    "blender_job.py",
    "proxy.blend",
    "proxy.mp4",
    "telemetry.json",
    "frames/index.json",
)


@register_primitive(
    category="scaffolding",
    tags=["runtime", "traceability", "telemetry"],
    cost_estimate="low",
    example_usage="build_runtime_contract('a' * 64, ['actor_a'], ['carry_01'], ['carry_01'])",
)
def build_runtime_contract(
    director_plan_hash: str,
    required_entities: list[str],
    required_events: list[str],
    required_camera_events: list[str],
) -> dict[str, Any]:
    """Build the immutable traceability payload expected by a generated job."""

    return {
        "contract_version": RUNTIME_CONTRACT_VERSION,
        "director_plan_hash": str(director_plan_hash),
        "required_entities": list(dict.fromkeys(str(item) for item in required_entities)),
        "required_events": list(dict.fromkeys(str(item) for item in required_events)),
        "required_camera_events": list(dict.fromkeys(str(item) for item in required_camera_events)),
        "required_artifacts": list(REQUIRED_REAL_ARTIFACTS),
    }


@register_primitive(
    category="scaffolding",
    tags=["runtime", "validation"],
    cost_estimate="low",
    example_usage="validate_runtime_contract(runtime_contract)",
)
def validate_runtime_contract(contract: Mapping[str, Any]) -> list[str]:
    """Return deterministic failures for an incomplete runtime contract."""

    failures: list[str] = []
    plan_hash = str(contract.get("director_plan_hash") or "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", plan_hash):
        failures.append("missing_director_plan_hash")
    artifacts = {str(item) for item in contract.get("required_artifacts", []) or []}
    for artifact in REQUIRED_REAL_ARTIFACTS:
        if artifact not in artifacts:
            failures.append(f"missing_required_artifact:{artifact}")
    for field in ("required_entities", "required_events", "required_camera_events"):
        values = contract.get(field)
        if not isinstance(values, list) or not values or any(not str(item).strip() for item in values):
            failures.append(f"missing_{field}")
    return list(dict.fromkeys(failures))


__all__ = [
    "REQUIRED_REAL_ARTIFACTS",
    "RUNTIME_CONTRACT_VERSION",
    "build_runtime_contract",
    "validate_runtime_contract",
]
