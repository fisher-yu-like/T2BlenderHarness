"""Independent deterministic checks for DirectorPlan integrity and coverage."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from videoact.contracts import Finding, TrajectoryPlan
from videoact.director_contracts import DirectorPlan

from .findings import deduplicate_findings, score_findings


ALLOWED_OWNERS = {
    "director_prompt_interpreter",
    "director_event_scheduler",
    "director_trajectory",
    "director_camera",
    "blender_code_agent",
    "blender_executor",
    "proxy_renderer",
    "evaluator",
}


class DirectorEvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    director_plan_score: float = Field(ge=0, le=100)
    findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


def _finding(
    failure_id: str,
    message: str,
    *,
    owner: str,
    severity: str = "hard",
    evidence: list[str] | None = None,
    root_cause_id: str | None = None,
) -> Finding:
    if owner not in ALLOWED_OWNERS:
        raise ValueError(f"Director finding owner is not allowed: {owner}")
    route = {
        "director_prompt_interpreter": "scene_contract_repair",
        "director_event_scheduler": "scene_contract_repair",
        "director_trajectory": "trajectory_repair",
        "director_camera": "camera_repair",
        "blender_code_agent": "runtime_repair",
        "blender_executor": "runtime_repair",
        "proxy_renderer": "runtime_repair",
        "evaluator": "candidate_recovery",
    }[owner]
    return Finding(
        failure_id=failure_id,
        owner=owner,
        category="director_plan",
        severity=severity,
        message=message,
        root_cause_id=root_cause_id or failure_id,
        evidence=evidence or [],
        repair_route=route,
    )


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _dependency_findings(plan: DirectorPlan) -> list[Finding]:
    events = list(plan.events)
    event_ids = {_value(event, "id") for event in events}
    findings: list[Finding] = []
    graph: dict[str, list[str]] = {}
    for event in events:
        event_id = _value(event, "id")
        dependencies = list(_value(event, "depends_on", []) or [])
        graph[event_id] = dependencies
        unknown = sorted(set(dependencies) - event_ids)
        if unknown:
            findings.append(
                _finding(
                    "director_dependency_mismatch",
                    f"event {event_id} depends on unknown events {unknown}",
                    owner="director_event_scheduler",
                    evidence=[event_id, *unknown],
                    root_cause_id=f"director_dependency:{event_id}",
                )
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(event_id: str) -> None:
        if event_id in visited or event_id not in graph:
            return
        if event_id in visiting:
            findings.append(
                _finding(
                    "director_dependency_cycle",
                    f"event dependency cycle includes {event_id}",
                    owner="director_event_scheduler",
                    evidence=[event_id],
                    root_cause_id="director_dependency_cycle",
                )
            )
            return
        visiting.add(event_id)
        for dependency in graph[event_id]:
            visit(dependency)
        visiting.remove(event_id)
        visited.add(event_id)

    for event_id in graph:
        visit(event_id)
    return findings


def _identity_findings(trajectory: TrajectoryPlan, telemetry: dict[str, Any] | None) -> list[Finding]:
    if not telemetry:
        return []
    findings: list[Finding] = []
    for entity_id, observed in (telemetry.get("objects") or {}).items():
        source_id = observed.get("source_entity_id") or observed.get("entity_id")
        if source_id and source_id != entity_id and entity_id in trajectory.entities:
            findings.append(
                _finding(
                    "director_identity_swap",
                    f"rendered entity {entity_id} reports source identity {source_id}",
                    owner="blender_executor",
                    evidence=[entity_id, str(source_id)],
                    root_cause_id=f"director_identity:{entity_id}",
                )
            )
    return findings


def _collision_findings(plan: DirectorPlan, trajectory: TrajectoryPlan) -> list[Finding]:
    actor_ids = {
        _value(entity, "id")
        for entity in plan.entities
        if _value(entity, "kind") == "actor"
    }
    findings: list[Finding] = []
    handoff_windows = [
        (
            round(event.start * trajectory.timebase.fps) + 1,
            round(event.end * trajectory.timebase.fps) + 1,
            set(event.participant_ids),
        )
        for event in plan.events
        if event.action == "handoff" and len(event.participant_ids) > 1
    ]
    actor_items = [
        (entity_id, trajectory.entities[entity_id])
        for entity_id in sorted(actor_ids)
        if entity_id in trajectory.entities
    ]
    for index, (left_id, left) in enumerate(actor_items):
        for right_id, right in actor_items[index + 1 :]:
            right_by_frame = {state.frame: state for state in right.states}
            for state in left.states:
                other = right_by_frame.get(state.frame)
                if other is not None and _distance(state.position, other.position) < 1.0:
                    in_handoff = any(
                        start <= state.frame <= end
                        and {left_id, right_id}.issubset(participants)
                        for start, end, participants in handoff_windows
                    )
                    if in_handoff:
                        continue
                    findings.append(
                        _finding(
                            "director_path_collision",
                            f"actor lanes {left_id} and {right_id} collide at frame {state.frame}",
                            owner="director_trajectory",
                            evidence=[left_id, right_id, str(state.frame)],
                            root_cause_id=f"director_path_collision:{left_id}:{right_id}",
                        )
                    )
                    break
    return findings


def _camera_findings(plan: DirectorPlan, trajectory: TrajectoryPlan) -> list[Finding]:
    findings: list[Finding] = []
    shots = trajectory.camera.shots
    for event in plan.events:
        event_id = event.id
        required_targets = [*event.participant_ids, *event.target_ids]
        matching = [shot for shot in shots if event_id in shot.required_event_ids]
        if not matching:
            findings.append(
                _finding(
                    "director_target_invisible",
                    f"event {event_id} has no camera coverage",
                    owner="director_camera",
                    evidence=[event_id],
                    root_cause_id=f"director_camera_coverage:{event_id}",
                )
            )
            continue
        for target_id in required_targets:
            if not any(
                target_id in shot.target_ids
                and shot.visibility_predicates.get(target_id) == "visible"
                and shot.max_occlusion <= 0.5
                for shot in matching
            ):
                findings.append(
                    _finding(
                        "director_target_invisible",
                        f"target {target_id} is not explicitly visible during {event_id}",
                        owner="director_camera",
                        evidence=[event_id, target_id],
                        root_cause_id=f"director_visibility:{event_id}:{target_id}",
                    )
                )
    return findings


def evaluate_director_plan(
    director_plan: DirectorPlan,
    trajectory_plan: TrajectoryPlan,
    *,
    telemetry: dict[str, Any] | None = None,
) -> DirectorEvaluationReport:
    findings: list[Finding] = []
    evidence_ids = {
        _value(evidence, "id") for evidence in (director_plan.evidence or [])
    }
    if director_plan.events and not evidence_ids:
        findings.append(
            _finding(
                "director_evidence_missing",
                "DirectorPlan has events but no decision evidence",
                owner="director_prompt_interpreter",
                evidence=["events", "evidence"],
                root_cause_id="director_evidence_coverage",
            )
        )
    for assumption in director_plan.assumptions or []:
        supported_by = set(_value(assumption, "supported_by_evidence_ids", []) or [])
        missing = sorted(supported_by - evidence_ids)
        if not supported_by or missing:
            findings.append(
                _finding(
                    "director_unsupported_assumption",
                    f"assumption {_value(assumption, 'id')} is not supported by known evidence",
                    owner="director_prompt_interpreter",
                    evidence=[_value(assumption, "id"), *missing],
                    root_cause_id=f"director_assumption:{_value(assumption, 'id')}",
                )
            )
    entity_ids = [_value(entity, "id") for entity in director_plan.entities]
    if len(entity_ids) != len(set(entity_ids)):
        findings.append(
            _finding(
                "director_entity_identity_duplicate",
                "DirectorPlan contains duplicate stable entity IDs",
                owner="director_prompt_interpreter",
                evidence=entity_ids,
                root_cause_id="director_entity_identity",
            )
        )
    trajectory_ids = set(trajectory_plan.entities)
    for entity_id in set(entity_ids) - trajectory_ids:
        findings.append(
            _finding(
                "director_trajectory_missing",
                f"DirectorPlan entity {entity_id} has no trajectory",
                owner="director_trajectory",
                evidence=[entity_id],
                root_cause_id=f"director_trajectory:{entity_id}",
            )
        )
    findings.extend(_dependency_findings(director_plan))
    findings.extend(_identity_findings(trajectory_plan, telemetry))
    findings.extend(_collision_findings(director_plan, trajectory_plan))
    findings.extend(_camera_findings(director_plan, trajectory_plan))
    findings = deduplicate_findings(findings)
    return DirectorEvaluationReport(
        director_plan_score=score_findings(findings),
        findings=findings,
        metrics={
            "finding_count": float(len(findings)),
            "hard_count": float(sum(finding.severity == "hard" for finding in findings)),
            "event_count": float(len(director_plan.events)),
        },
    )
