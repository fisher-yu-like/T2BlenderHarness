"""Deterministic camera coverage predicates."""

from __future__ import annotations

from videoact.contracts import Finding, SceneContract, TrajectoryPlan


def check_camera_coverage(contract: SceneContract, plan: TrajectoryPlan) -> list[Finding]:
    covered = {
        event_id
        for shot in plan.camera.shots
        for event_id in shot.required_event_ids
    }
    findings: list[Finding] = []
    for event_id in contract.must_show:
        if event_id not in covered:
            findings.append(
                Finding(
                    failure_id="camera_event_uncovered",
                    owner="camera_planner",
                    category="camera_coverage",
                    severity="hard",
                    root_cause_id="camera_required_event_coverage",
                    message=f"camera has no shot for required event {event_id}",
                    evidence=[event_id],
                    repair_route="camera_repair",
                )
            )
    return findings


def check_camera_motion_intent(contract: SceneContract, plan: TrajectoryPlan) -> list[Finding]:
    """Verify that prompt-declared camera choreography becomes executable shots."""
    findings: list[Finding] = []
    shot_types = {shot.trajectory_type for shot in plan.camera.shots}
    required = set(contract.camera_constraints)
    requirements = {
        "camera_follow": ("follow", "camera_follow_intent_missing"),
        "camera_orbit": ("orbit", "camera_orbit_intent_missing"),
        "camera_dolly": ("dolly", "camera_dolly_intent_missing"),
    }
    for constraint, (trajectory_type, failure_id) in requirements.items():
        if constraint in required and trajectory_type not in shot_types:
            findings.append(
                Finding(
                    failure_id=failure_id,
                    owner="camera_planner",
                    category="camera_innovation",
                    severity="error",
                    root_cause_id=f"camera_motion_intent:{trajectory_type}",
                    message=f"prompt requires {trajectory_type} choreography but no such camera shot exists",
                    evidence=[constraint, trajectory_type, str(sorted(shot_types))],
                    repair_route="camera_repair",
                )
            )
    if "grasp_in_closeup" in required and not any(
        "grasp" in shot.required_event_ids and "closeup" in shot.shot_id for shot in plan.camera.shots
    ):
        findings.append(
            Finding(
                failure_id="camera_grasp_closeup_missing",
                owner="camera_planner",
                category="camera_coverage",
                severity="error",
                root_cause_id="camera_required_closeup",
                message="grasp_in_closeup is declared but no grasp close-up shot is planned",
                evidence=["grasp_in_closeup"],
                repair_route="camera_repair",
            )
        )
    if "occlusion_reveal" in required and not any(
        "reveal" in shot.required_event_ids for shot in plan.camera.shots
    ):
        findings.append(
            Finding(
                failure_id="camera_reveal_shot_missing",
                owner="camera_planner",
                category="camera_coverage",
                severity="error",
                root_cause_id="camera_required_reveal",
                message="occlusion_reveal is declared but no reveal shot observes it",
                evidence=["occlusion_reveal"],
                repair_route="camera_repair",
            )
        )
    return findings
