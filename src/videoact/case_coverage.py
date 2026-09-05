"""Fail-closed coverage checks between a case, its plan, and generated code.

The renderer must not be allowed to turn an incomplete DirectorPlan into a
generic proxy scene.  This module is intentionally independent of Blender and
only inspects the structured artifacts available before rendering.
"""

from __future__ import annotations

import hashlib
import ast
from typing import Any, Mapping

from pydantic import Field

from .director_contracts import ContractModel
from .obligations import ObligationCompilation, ObligationRecord, validate_obligation_completeness


class CoverageReport(ContractModel):
    """Serializable pre-render coverage decision for one dataset case."""

    case_id: str = Field(min_length=1)
    status: str
    hard_failures: list[str] = Field(default_factory=list)
    required_entities: list[str] = Field(default_factory=list)
    required_events: list[str] = Field(default_factory=list)
    required_camera_events: list[str] = Field(default_factory=list)
    covered_entities: list[str] = Field(default_factory=list)
    covered_events: list[str] = Field(default_factory=list)
    covered_trajectory_events: list[str] = Field(default_factory=list)
    covered_camera_events: list[str] = Field(default_factory=list)
    source_hash: str = Field(min_length=64, max_length=64)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    # Optional T03 trace anchors.  Legacy benchmark coverage remains valid
    # when no obligation compilation is supplied.
    obligation_ids: list[str] = Field(default_factory=list)
    covered_obligation_ids: list[str] = Field(default_factory=list)
    missing_obligation_ids: list[str] = Field(default_factory=list)


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _ids(value: Any, key: str = "id") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        values = value.keys() if key == "__keys__" else value.values()
        for item in values:
            if key == "__keys__":
                candidate = item
            elif isinstance(item, Mapping):
                candidate = item.get(key)
            else:
                candidate = getattr(item, key, None)
            if candidate is not None and str(candidate).strip():
                result.append(str(candidate))
    else:
        for item in _items(value):
            if isinstance(item, (str, int)):
                candidate = item
            elif isinstance(item, Mapping):
                candidate = item.get(key)
            else:
                candidate = getattr(item, key, None)
            if candidate is not None and str(candidate).strip():
                result.append(str(candidate))
    return list(dict.fromkeys(result))


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _required_case_ids(record: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    oracle = record.get("oracle_expectations") or {}
    proxy_scene = record.get("proxy_scene") or {}
    scene_entities = proxy_scene.get("entities") or []
    required_entities = _ids(oracle.get("required_entity_ids"))
    if not required_entities:
        required_entities = _ids(scene_entities)

    required_events = _ids(record.get("required_events"))
    if not required_events:
        required_events = _ids(oracle.get("event_order"))
    if not required_events:
        required_events = _ids(record.get("event_graph"))

    required_camera_events = _ids(oracle.get("required_camera_events"))
    if not required_camera_events:
        camera = proxy_scene.get("camera") or {}
        required_camera_events = _ids(camera.get("must_show_events"))
    return required_entities, required_events, required_camera_events


def _source_hash(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _trajectory_event_ids(trajectories: Any) -> set[str]:
    """Collect event IDs that are consumed by motion/attachment trajectories."""

    entities = _field(trajectories, "entities", {})
    if not isinstance(entities, Mapping):
        return set()
    event_ids: set[str] = set()
    for trajectory in entities.values():
        for primitive in _items(_field(trajectory, "motion_primitives")):
            parameters = _field(primitive, "parameters", {})
            event_id = _field(parameters, "event_id")
            if event_id is not None and str(event_id).strip():
                event_ids.add(str(event_id))
        for attachment in _items(_field(trajectory, "attachment_events")):
            event_id = _field(attachment, "event_id")
            if event_id is not None and str(event_id).strip():
                event_ids.add(str(event_id))
    return event_ids


def _source_tokens(source: str) -> set[str]:
    """Return executable/name and string-literal tokens, excluding comments."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            tokens.add(node.value)
    return tokens


def _existing_hashes(existing_code_hashes: Mapping[str, str] | None) -> set[str]:
    hashes: set[str] = set()
    for source_or_hash in (existing_code_hashes or {}).values():
        value = str(source_or_hash)
        hashes.add(value)
        if len(value) != 64:
            hashes.add(_source_hash(value))
    return hashes


def validate_case_coverage(
    *,
    record: Mapping[str, Any],
    director_plan: Any,
    director_trajectories: Any,
    director_camera: Any,
    generated_code: str,
    existing_code_hashes: Mapping[str, str] | None = None,
    obligations: ObligationCompilation | list[ObligationRecord] | None = None,
    source_obligation_ids: list[str] | None = None,
) -> CoverageReport:
    """Reject a case unless every required identity and event reaches source.

    ``existing_code_hashes`` is a registry of previously frozen sources.  A
    duplicate source is hard-failed here; the caller may only reuse a frozen
    source after an explicit cache lookup for the same plan/version.
    """

    case_id = str(record.get("case_id") or "unknown-case")
    required_entities, required_events, required_camera_events = _required_case_ids(record)
    plan_entity_ids = _ids(_field(director_plan, "entities"))
    plan_event_ids = _ids(_field(director_plan, "events"))
    plan_entities = set(plan_entity_ids)
    plan_events = set(plan_event_ids)
    trajectory_events = _trajectory_event_ids(director_trajectories)
    trajectory_entities = set(_ids(_field(director_trajectories, "entities"), "__keys__"))
    camera_events: set[str] = set()
    for shot in _items(_field(director_camera, "shots")):
        camera_events.update(_ids(_field(shot, "required_event_ids")))

    obligation_ids: list[str] = []
    covered_obligation_ids: list[str] = []
    missing_obligation_ids: list[str] = []
    if obligations is not None:
        # This is a schema gate, not a score.  A truncated compilation is
        # rejected before source coverage can be reported as successful.
        validate_obligation_completeness(obligations)
        if isinstance(obligations, ObligationCompilation):
            obligation_ids = list(obligations.obligation_ids)
        else:
            obligation_ids = [item.obligation_id for item in obligations]
        plan_obligation_ids = set(_ids(_field(director_plan, "obligation_ids")))
        missing_obligation_ids = [item for item in obligation_ids if item not in plan_obligation_ids]
        if missing_obligation_ids:
            hard_failures = [
                f"missing_plan_obligations:{item}" for item in missing_obligation_ids
            ]
        else:
            hard_failures = []
        # The report itself is the source-coverage handoff.  If a caller
        # supplies an explicit source list, use it as the stricter check.
        covered_obligation_ids = list(
            dict.fromkeys(source_obligation_ids if source_obligation_ids is not None else obligation_ids)
        )
        if source_obligation_ids is not None:
            missing_source_ids = [item for item in obligation_ids if item not in set(source_obligation_ids)]
            hard_failures.extend(f"missing_source_obligations:{item}" for item in missing_source_ids)
    else:
        hard_failures = []

    # Raw benchmark records intentionally carry no locally authored entity,
    # event, or camera labels.  Their executable DirectorPlan is therefore
    # the *internal* coverage contract: it must still reach trajectories,
    # camera cues, and generated source, but this gate must not invent a
    # semantic oracle that was absent from the benchmark.
    if record.get("benchmark_prompt_only") is True:
        if not required_entities:
            required_entities = sorted(plan_entity_ids)
        if not required_events:
            required_events = list(plan_event_ids)
        if not required_camera_events:
            required_camera_events = sorted(camera_events)

    source = str(generated_code or "")
    source_hash = _source_hash(source)
    hard_failures = list(hard_failures)
    if not source.strip():
        hard_failures.append("empty_generated_source")

    missing_plan_entities = sorted(set(required_entities) - plan_entities)
    missing_trajectory_entities = sorted(set(required_entities) - trajectory_entities)
    missing_plan_events = sorted(set(required_events) - plan_events)
    missing_camera_events = sorted(set(required_camera_events) - camera_events)
    if missing_plan_entities:
        hard_failures.extend(f"missing_plan_entities:{item}" for item in missing_plan_entities)
    if missing_trajectory_entities:
        hard_failures.extend(f"missing_trajectory_entities:{item}" for item in missing_trajectory_entities)
    if missing_plan_events:
        hard_failures.extend(f"missing_plan_events:{item}" for item in missing_plan_events)
    missing_trajectory_events = sorted(set(required_events) - trajectory_events)
    if missing_trajectory_events:
        hard_failures.extend(f"missing_trajectory_events:{item}" for item in missing_trajectory_events)
    if missing_camera_events:
        hard_failures.extend(f"missing_camera_events:{item}" for item in missing_camera_events)

    source_requirements = list(dict.fromkeys([*required_entities, *required_events, *required_camera_events]))
    source_tokens = _source_tokens(source)
    hard_failures.extend(
        f"missing_source_tokens:{item}" for item in source_requirements if item not in source_tokens
    )
    if source_hash in _existing_hashes(existing_code_hashes):
        hard_failures.append(f"duplicate_source_hash:{source_hash}")

    all_requirements = [
        (required_entities, plan_entities & set(required_entities) & trajectory_entities),
        (required_events, plan_events & set(required_events)),
        (required_events, trajectory_events & set(required_events)),
        (required_camera_events, camera_events & set(required_camera_events)),
        (source_requirements, {item for item in source_requirements if item in source_tokens}),
    ]
    total = sum(len(required) for required, _covered in all_requirements)
    covered = sum(len(covered) for _required, covered in all_requirements)
    ratio = covered / total if total else 0.0
    if total == 0:
        hard_failures.append("empty_case_requirements")

    return CoverageReport(
        case_id=case_id,
        status="pass" if not hard_failures else "coverage_failed",
        hard_failures=list(dict.fromkeys(hard_failures)),
        required_entities=required_entities,
        required_events=required_events,
        required_camera_events=required_camera_events,
        covered_entities=sorted(plan_entities & set(required_entities) & trajectory_entities),
        covered_events=sorted(plan_events & set(required_events)),
        covered_trajectory_events=sorted(trajectory_events & set(required_events)),
        covered_camera_events=sorted(camera_events & set(required_camera_events)),
        source_hash=source_hash,
        coverage_ratio=round(ratio, 6),
        obligation_ids=obligation_ids,
        covered_obligation_ids=covered_obligation_ids,
        missing_obligation_ids=missing_obligation_ids,
    )
