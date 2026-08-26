"""Compare generated plans with dataset-authored expectations.

This is intentionally separate from the normal deterministic evaluator.  The
normal evaluator checks whether a plan is internally valid; this evaluator
checks whether it implements the requested task and proxy scene contract.
"""

from __future__ import annotations

from typing import Any

from videoact.contracts import Finding, SceneContract, TrajectoryPlan


def _finding(
    failure_id: str,
    owner: str,
    message: str,
    evidence: list[str],
    *,
    severity: str = "hard",
    root_cause_id: str | None = None,
) -> Finding:
    routes = {
        "scene_parser": "scene_contract_repair",
        "trajectory_planner": "trajectory_repair",
        "camera_planner": "camera_repair",
        "proxy_renderer": "runtime_repair",
    }
    return Finding(
        failure_id=failure_id,
        owner=owner,
        category="independent_oracle",
        severity=severity,
        root_cause_id=root_cause_id or failure_id,
        message=message,
        evidence=evidence,
        repair_route=routes.get(owner, "candidate_recovery"),
    )


def evaluate_independent_oracle(
    record: dict[str, Any],
    contract: SceneContract,
    plan: TrajectoryPlan,
) -> list[Finding]:
    """Return hard findings for mismatches against authored expectations."""
    oracle = record.get("oracle_expectations")
    if not oracle:
        return []
    evidence = [f"dataset/{record.get('case_id', 'unknown')}/oracle_expectations"]
    findings: list[Finding] = []
    actual_events = [event.id for event in contract.events]
    expected_events = list(oracle.get("event_order", []))
    if actual_events != expected_events:
        missing = [event for event in expected_events if event not in actual_events]
        extra = [event for event in actual_events if event not in expected_events]
        findings.append(
            _finding(
                "oracle_event_order_mismatch",
                "scene_parser",
                f"authored event order differs; missing={missing}, extra={extra}, actual={actual_events}, expected={expected_events}",
                evidence,
                root_cause_id="prompt_event_order",
            )
        )

    expected_constraints = set(oracle.get("required_camera_constraints", []))
    actual_constraints = set(contract.camera_constraints) | set(contract.physics_constraints)
    missing_constraints = sorted(expected_constraints - actual_constraints)
    if missing_constraints:
        findings.append(
            _finding(
                "oracle_constraint_missing",
                "scene_parser",
                f"authored constraints are not represented: {missing_constraints}",
                evidence,
                root_cause_id="prompt_required_constraints",
            )
        )

    actual_camera_types = {shot.trajectory_type for shot in plan.camera.shots}
    required_camera_types = set(oracle.get("required_camera_types", []))
    missing_camera_types = sorted(required_camera_types - actual_camera_types)
    if missing_camera_types:
        findings.append(
            _finding(
                "oracle_camera_intent_missing",
                "camera_planner",
                f"required camera trajectory types are missing: {missing_camera_types}; actual={sorted(actual_camera_types)}",
                evidence,
                severity="error",
                root_cause_id="camera_authored_intent",
            )
        )

    character = plan.entities.get("character")
    actual_primitives = {primitive.type for primitive in character.motion_primitives} if character else set()
    required_primitives = set(oracle.get("required_motion_primitives", []))
    missing_primitives = sorted(required_primitives - actual_primitives)
    if missing_primitives:
        findings.append(
            _finding(
                "oracle_motion_primitive_missing",
                "trajectory_planner",
                f"required motion primitives are missing: {missing_primitives}; actual={sorted(actual_primitives)}",
                evidence,
                severity="error",
                root_cause_id="trajectory_authored_primitives",
            )
        )

    actual_attachments = [event.action for event in character.attachment_events] if character else []
    required_attachments = list(oracle.get("required_attachment_actions", []))
    if actual_attachments != required_attachments:
        findings.append(
            _finding(
                "oracle_attachment_lifecycle_mismatch",
                "trajectory_planner",
                f"attachment lifecycle differs; actual={actual_attachments}, expected={required_attachments}",
                evidence,
                root_cause_id="attachment_lifecycle:character",
            )
        )

    actual_entities = {entity.id: entity.kind for entity in contract.entities}
    required_entities = oracle.get("required_entity_kinds", {})
    missing_entities = sorted(set(required_entities) - set(actual_entities))
    wrong_kinds = sorted(
        entity_id
        for entity_id, kind in required_entities.items()
        if entity_id in actual_entities and actual_entities[entity_id] != kind
    )
    if missing_entities or wrong_kinds:
        findings.append(
            _finding(
                "oracle_proxy_entity_mismatch",
                "scene_parser",
                f"proxy entity contract mismatch; missing={missing_entities}, wrong_kinds={wrong_kinds}",
                evidence,
                root_cause_id="scene_authored_entities",
            )
        )
    return findings
