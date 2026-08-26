"""Contract-level predicates expressed as structured Findings."""

from __future__ import annotations

from videoact.contracts import Finding, SceneContract, TrajectoryPlan


def check_event_order(contract: SceneContract, plan: TrajectoryPlan) -> list[Finding]:
    observed = [item.event_id for item in plan.event_observability]
    expected = list(contract.must_show)
    if observed and observed != expected:
        return [
            Finding(
                failure_id="event_order",
                owner="scene_parser",
                category="prompt_compliance",
                severity="hard",
                root_cause_id="prompt_event_order",
                message="required event observability is not in contract order",
                repair_route="scene_contract_repair",
            )
        ]
    return []


def check_required_event_coverage(contract: SceneContract, plan: TrajectoryPlan) -> list[Finding]:
    covered = {
        event_id
        for shot in plan.camera.shots
        for event_id in shot.required_event_ids
    }
    missing = sorted(set(contract.must_show) - covered)
    if not missing:
        return []
    return [
        Finding(
            failure_id="missing_required_event",
            owner="camera_planner",
            category="camera_coverage",
            severity="hard",
            root_cause_id="camera_required_event_coverage",
            message=f"required events are not covered by any camera shot: {missing}",
            evidence=missing,
            repair_route="camera_repair",
        )
    ]
