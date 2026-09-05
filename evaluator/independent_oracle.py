"""Compare generated plans with dataset-authored expectations.

This is intentionally separate from the normal deterministic evaluator.  The
normal evaluator checks whether a plan is internally valid; this evaluator
checks whether it implements the requested task and proxy scene contract.
"""

from __future__ import annotations

from typing import Any

from videoact.contracts import Finding, SceneContract, TrajectoryPlan
from .physics_oracle import evaluate_physics_oracle


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


def _entity_value(entity: Any, field: str, default: Any = None) -> Any:
    if isinstance(entity, dict):
        return entity.get(field, default)
    return getattr(entity, field, default)


def _actor_ids(contract: SceneContract, plan: TrajectoryPlan) -> list[str]:
    """Return actor IDs without assuming the legacy ``character`` name."""

    ids = [
        str(entity.id)
        for entity in contract.entities
        if str(entity.kind).lower() in {"actor", "character"}
        and str(entity.id) in plan.entities
    ]
    if ids:
        return ids
    return sorted(
        entity_id
        for entity_id in plan.entities
        if entity_id.startswith("actor") or entity_id == "character"
    )


def _trajectory_targets(
    oracle: dict[str, Any], contract: SceneContract, plan: TrajectoryPlan
) -> list[str]:
    explicit = oracle.get("trajectory_entity_ids")
    if isinstance(explicit, list) and explicit:
        return [entity_id for entity_id in explicit if entity_id in plan.entities]
    explicit_id = oracle.get("trajectory_entity_id")
    if isinstance(explicit_id, str) and explicit_id in plan.entities:
        return [explicit_id]
    if "character" in plan.entities:
        return ["character"]
    return _actor_ids(contract, plan)


def _state_axis_at(trajectory: Any, frame: int, axis: int = 1) -> float | None:
    states = sorted(getattr(trajectory, "states", []), key=lambda state: state.frame)
    if not states:
        return None
    if frame <= states[0].frame:
        return float(states[0].position[axis])
    if frame >= states[-1].frame:
        return float(states[-1].position[axis])
    for left, right in zip(states, states[1:]):
        if left.frame <= frame <= right.frame:
            if right.frame == left.frame:
                return float(left.position[axis])
            ratio = (frame - left.frame) / (right.frame - left.frame)
            return float(left.position[axis]) + ratio * (
                float(right.position[axis]) - float(left.position[axis])
            )
    return None


def _allowed_crossing_intervals(contract: SceneContract, oracle: dict[str, Any]) -> list[tuple[int, int]]:
    allowed_ids = {str(value) for value in oracle.get("allowed_crossing_event_ids", []) or []}
    intervals: list[tuple[int, int]] = []
    for event in contract.events:
        description = str(event.description).lower()
        if event.id in allowed_ids or any(token in description for token in ("handoff", "transfer", "cross")):
            intervals.append((round(event.start * contract.fps) + 1, round(event.end * contract.fps) + 1))
    return intervals


def _find_unplanned_actor_crossings(
    oracle: dict[str, Any], contract: SceneContract, plan: TrajectoryPlan
) -> list[str]:
    """Detect lane-order inversions using authored actor lanes and plan states.

    This is a negative-constraint check, not a visual-quality score.  Handoff
    or explicitly authored crossing windows are exempt; an unplanned inversion
    is evidence that the trajectory planner violated the case contract.
    """

    actor_ids = _actor_ids(contract, plan)
    allowed_intervals = _allowed_crossing_intervals(contract, oracle)
    frames = sorted(
        {
            state.frame
            for actor_id in actor_ids
            for state in plan.entities[actor_id].states
        }
    )
    crossings: list[str] = []
    for index, left_id in enumerate(actor_ids):
        for right_id in actor_ids[index + 1 :]:
            previous_frame: int | None = None
            previous_delta: float | None = None
            for frame in frames:
                left_y = _state_axis_at(plan.entities[left_id], frame)
                right_y = _state_axis_at(plan.entities[right_id], frame)
                if left_y is None or right_y is None:
                    continue
                delta = left_y - right_y
                if previous_delta is not None and previous_frame is not None:
                    inverted = previous_delta * delta < 0
                    if inverted:
                        midpoint = round((previous_frame + frame) / 2)
                        allowed = any(start <= midpoint <= end for start, end in allowed_intervals)
                        if not allowed:
                            crossings.append(f"{left_id}:{right_id}:{previous_frame}-{frame}")
                            break
                previous_frame = frame
                previous_delta = delta
    return crossings


def evaluate_independent_oracle(
    record: dict[str, Any],
    contract: SceneContract,
    plan: TrajectoryPlan,
    telemetry: dict[str, Any] | None = None,
    *,
    physics_report: dict[str, Any] | None = None,
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

    expected_entity_ids = set(oracle.get("required_entity_ids", []))
    actual_entity_ids = {entity.id for entity in contract.entities}
    missing_entities = sorted(expected_entity_ids - actual_entity_ids)
    if missing_entities:
        findings.append(
            _finding(
                "oracle_entity_missing",
                "scene_parser",
                f"authored entities are absent from the generated contract: {missing_entities}",
                evidence + missing_entities,
                root_cause_id="prompt_required_entity_coverage",
            )
        )

    expected_camera_events = set(oracle.get("required_camera_events", []))
    actual_camera_events = {
        event_id
        for shot in plan.camera.shots
        for event_id in shot.required_event_ids
    }
    missing_camera_events = sorted(expected_camera_events - actual_camera_events)
    if missing_camera_events:
        findings.append(
            _finding(
                "oracle_camera_event_missing",
                "camera_planner",
                f"authored required camera events are not covered: {missing_camera_events}",
                evidence + missing_camera_events,
                root_cause_id="prompt_required_camera_coverage",
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

    trajectory_targets = _trajectory_targets(oracle, contract, plan)
    actual_primitives = {
        primitive.type
        for entity_id in trajectory_targets
        for primitive in plan.entities[entity_id].motion_primitives
    }
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

    actual_attachments = [
        (event.frame, event.action)
        for entity_id in sorted(plan.entities)
        for event in plan.entities[entity_id].attachment_events
    ]
    actual_attachments = [action for _frame, action in sorted(actual_attachments)]
    required_attachments = list(oracle.get("required_attachment_actions", []))
    actual_iter = iter(actual_attachments)
    attachments_match = all(
        any(action == required for action in actual_iter)
        for required in required_attachments
    )
    if not attachments_match:
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

    negative_constraints = set(oracle.get("required_negative_constraints", []))
    if "no_unplanned_actor_crossing" in negative_constraints:
        trajectory_crossings = _find_unplanned_actor_crossings(oracle, contract, plan)
        if trajectory_crossings:
            findings.append(
                _finding(
                    "oracle_negative_constraint_violated",
                    "proxy_renderer",
                    "trajectory evidence contains an unplanned actor lane crossing",
                    evidence + trajectory_crossings,
                    root_cause_id="negative_constraint:no_unplanned_actor_crossing",
                )
            )

    # Runtime negative constraints are derived from the trusted observer's raw
    # transforms, bounds, and pose bones.  Generated fields such as
    # ``attachment_penetration`` and ``transfer_constraints`` are deliberately
    # ignored at this boundary.
    # Keep plan-only callers separate from the real-video runtime boundary.
    # ``telemetry=None`` means that no runtime claim was supplied; it must not
    # manufacture a physical failure while checking authored plan semantics.
    # The real-video path passes telemetry (including an empty mapping when
    # the artifact is missing) or an explicit physics report, so that path
    # remains fail-closed through ``evaluate_physics_oracle``.
    if physics_report is not None:
        physics = physics_report
    elif telemetry is not None:
        physics = evaluate_physics_oracle(
            record,
            telemetry,
            contract=contract,
            trajectory_plan=plan,
        )
    else:
        physics = None
    for item in (physics or {}).get("findings", []) or []:
        if isinstance(item, dict):
            findings.append(Finding.model_validate(item))
    return findings
