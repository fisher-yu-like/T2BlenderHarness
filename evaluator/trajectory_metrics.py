"""Deterministic checks for phase-aligned character and object trajectories."""

from __future__ import annotations

from videoact.contracts import Finding, SceneContract, TrajectoryPlan


def _finding(
    failure_id: str,
    message: str,
    evidence: list[str],
    *,
    severity: str = "error",
    root_cause_id: str | None = None,
) -> Finding:
    return Finding(
        failure_id=failure_id,
        owner="trajectory_planner",
        category="trajectory_planning",
        severity=severity,
        message=message,
        evidence=evidence,
        root_cause_id=root_cause_id or failure_id,
        repair_route="trajectory_repair",
    )


def check_trajectory_phase_alignment(contract: SceneContract, plan: TrajectoryPlan) -> list[Finding]:
    """Check event phases, camera-coupled motion, and attachment lifecycle alignment."""
    findings: list[Finding] = []
    event_ids = {event.id for event in contract.events}
    for requirement in contract.trajectory_requirements:
        trajectory = plan.entities.get(requirement.entity_id)
        root = f"trajectory_requirement:{requirement.entity_id}"
        if trajectory is None:
            findings.append(
                _finding(
                    "trajectory_entity_missing",
                    f"declared trajectory for {requirement.entity_id} is missing",
                    [requirement.entity_id],
                    severity="hard",
                    root_cause_id=root,
                )
            )
            continue
        if len(trajectory.states) < requirement.minimum_states:
            findings.append(
                _finding(
                    "trajectory_state_evidence_insufficient",
                    f"{requirement.entity_id} has {len(trajectory.states)} states; contract requires {requirement.minimum_states}",
                    [requirement.entity_id, str(requirement.minimum_states)],
                    root_cause_id=root,
                )
            )
        if requirement.require_phase_primitives:
            primitive_phases = {
                primitive.parameters.get("phase")
                for primitive in trajectory.motion_primitives
                if primitive.parameters.get("phase")
            }
            missing_phases = sorted(set(requirement.required_event_ids) - primitive_phases)
            if missing_phases:
                findings.append(
                    _finding(
                        "trajectory_phase_primitive_missing",
                        f"{requirement.entity_id} has no declared primitive for phases {missing_phases}",
                        [requirement.entity_id, *missing_phases],
                        root_cause_id=root,
                    )
                )
        actual_actions = [event.action for event in trajectory.attachment_events]
        if requirement.required_attachment_actions and actual_actions != requirement.required_attachment_actions:
            findings.append(
                _finding(
                    "trajectory_attachment_lifecycle_mismatch",
                    f"{requirement.entity_id} attachment actions {actual_actions} do not match {requirement.required_attachment_actions}",
                    [requirement.entity_id, *requirement.required_attachment_actions],
                    severity="hard",
                    root_cause_id=f"attachment_lifecycle:{requirement.entity_id}",
                )
            )

    character = plan.entities.get("character")
    if character is None:
        return findings

    if "camera_orbit" in contract.camera_constraints and "carry" in event_ids:
        if not any(
            primitive.type == "orbit" and primitive.parameters.get("camera_coupled")
            for primitive in character.motion_primitives
        ):
            findings.append(
                _finding(
                    "trajectory_camera_coupled_orbit_missing",
                    "carry phase requires a camera-coupled orbit primitive",
                    ["carry", "camera_orbit"],
                    severity="warning",
                    root_cause_id="camera_orbit_execution",
                )
            )
    if "camera_dolly" in contract.camera_constraints and "release" in event_ids:
        if not any(
            primitive.type == "dolly" and primitive.parameters.get("camera_coupled")
            for primitive in character.motion_primitives
        ):
            findings.append(
                _finding(
                    "trajectory_camera_coupled_dolly_missing",
                    "release phase requires a camera-coupled dolly primitive",
                    ["release", "camera_dolly"],
                    severity="warning",
                    root_cause_id="camera_dolly_execution",
                )
            )

    return findings
