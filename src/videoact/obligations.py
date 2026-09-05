"""Typed, traceable obligations compiled from a case and its plan.

An obligation is deliberately smaller than a score.  It states one thing the
case requires, how to check it, and where evidence for that check may be
attached.  The compiler only uses structured case/plan contracts for stable
identity; prompt text is used for explicit camera cues, but never as the sole
identity of an obligation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any
from typing_extensions import Literal

from pydantic import Field, model_validator

from .director_contracts import ContractModel


OBLIGATION_SCHEMA_VERSION = "obligation-compiler-v1"

ObligationKind = Literal[
    "entity",
    "event",
    "event_order",
    "event_timing",
    "participant",
    "target",
    "ownership_transition",
    "transfer_window",
    "trajectory",
    "contact_support",
    "camera_coverage",
    "camera_visibility",
    "camera_motion",
    "camera_innovation",
    "artifact",
    "provenance",
]
ObligationStatus = Literal[
    "pending",
    "satisfied",
    "failed",
    "unavailable",
    "not_applicable",
]


class ObligationRecord(ContractModel):
    """One independently checkable requirement in a case."""

    obligation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    kind: ObligationKind
    required: bool
    applicable: bool
    expected: Any = None
    pass_rule: str = Field(min_length=1)
    # Both spellings are accepted at the boundary.  ``evidence_sources`` is
    # the canonical spelling used by the compiler; ``evidence_source`` keeps
    # the model compatible with the design document's singular wording.
    evidence_sources: list[str] = Field(default_factory=list)
    evidence_source: list[str] | None = None
    evidence_refs: list[str] = Field(default_factory=list)
    planned_status: ObligationStatus = "pending"
    implemented_status: ObligationStatus = "pending"
    executed_status: ObligationStatus = "pending"
    visible_status: ObligationStatus = "pending"
    judged_status: ObligationStatus = "pending"
    # These are reserved, deterministic anchors.  They do not claim that an
    # artifact already exists; later stages replace the corresponding
    # evidence reference when they write an actual artifact.
    traceability: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_obligation(self) -> "ObligationRecord":
        sources = list(dict.fromkeys([*self.evidence_sources, *(self.evidence_source or [])]))
        object.__setattr__(self, "evidence_sources", sources)
        object.__setattr__(self, "evidence_source", list(sources))
        if not sources:
            raise ValueError("obligation requires at least one evidence source")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("obligation evidence_refs must be unique")
        if not self.applicable and self.required:
            raise ValueError("non-applicable obligation cannot be required")
        if not self.applicable:
            statuses = (
                self.planned_status,
                self.implemented_status,
                self.executed_status,
                self.visible_status,
                self.judged_status,
            )
            if any(status not in {"pending", "not_applicable"} for status in statuses):
                raise ValueError("non-applicable obligation cannot have a scored status")
        return self

    @property
    def is_required(self) -> bool:
        """Whether this record participates in a fail-closed gate."""

        return self.required and self.applicable


class ObligationCompilation(ContractModel):
    """Versioned compiler output and its cross-stage trace anchors."""

    schema_version: str = OBLIGATION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=64, max_length=64)
    director_plan_id: str | None = None
    director_plan_hash: str | None = Field(default=None, min_length=64, max_length=64)
    obligations: list[ObligationRecord] = Field(default_factory=list)
    obligation_ids: list[str] = Field(default_factory=list)
    na_dimensions: list[str] = Field(default_factory=list)
    traceability: dict[str, list[str]] = Field(default_factory=dict)
    fingerprint: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_compilation(self) -> "ObligationCompilation":
        record_ids = [item.obligation_id for item in self.obligations]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("obligation IDs must be unique")
        ids = list(self.obligation_ids or record_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("obligation_ids must be unique")
        if set(ids) != set(record_ids):
            raise ValueError("obligation_ids must match obligation records")
        object.__setattr__(self, "obligation_ids", ids)
        trace = dict(self.traceability)
        for stage in ("plan", "source_coverage", "trusted_observer", "telemetry", "evaluator"):
            trace[stage] = list(dict.fromkeys(str(item) for item in trace.get(stage, ids)))
        object.__setattr__(self, "traceability", trace)
        return self

    @property
    def records(self) -> list[ObligationRecord]:
        """Alias useful to callers that call the output a record set."""

        return self.obligations

    @property
    def required_obligation_ids(self) -> list[str]:
        return [item.obligation_id for item in self.obligations if item.is_required]


class ObligationGateReport(ContractModel):
    """Fail-closed presence check for a compiled obligation set."""

    status: Literal["pass", "failed"]
    required_ids: list[str] = Field(default_factory=list)
    present_ids: list[str] = Field(default_factory=list)
    missing_ids: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


def _dump(value: Any) -> Any:
    if value is None:
        return None
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_dump(child) for child in value]
    if isinstance(value, tuple):
        return [_dump(child) for child in value]
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _hash(value: Any) -> str:
    encoded = json.dumps(_dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text[:48] or "item"


def _stable_id(case_id: str, kind: str, key: str) -> str:
    digest = hashlib.sha256(f"{case_id}|{kind}|{key}".encode("utf-8")).hexdigest()[:10]
    return f"{_slug(kind)}.{_slug(key)}.{digest}"


def _normalise_entity_kind(value: Any) -> str:
    normalized = str(value or "environment").casefold()
    if normalized in {"actor", "character", "person", "human"}:
        return "actor"
    if normalized in {"prop", "object", "item"}:
        return "prop"
    if normalized in {"support", "surface", "platform", "table"}:
        return "support"
    return "environment"


def _entity_kind(entity: Any) -> str:
    return _normalise_entity_kind(_field(entity, "kind", _field(entity, "entity_kind", "environment")))


def _entity_map(record: Mapping[str, Any], plan: Mapping[str, Any], scene: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    specs: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []

    def add(value: Any, *, source: str) -> None:
        if isinstance(value, Mapping):
            entity_id = _field(value, "id", _field(value, "entity_id"))
            if entity_id is None:
                return
            entity_id = str(entity_id)
            specs.setdefault(entity_id, {"id": entity_id, "source": source})
            specs[entity_id].update({str(key): _dump(child) for key, child in value.items()})
        else:
            entity_id = str(value)
            specs.setdefault(entity_id, {"id": entity_id, "source": source})
        if entity_id not in ordered:
            ordered.append(entity_id)

    oracle = record.get("oracle_expectations") or {}
    required_ids = _strings(record.get("required_entity_ids")) or _strings(oracle.get("required_entity_ids"))
    required_kinds = oracle.get("required_entity_kinds") or record.get("required_entity_kinds") or {}
    for entity_id in required_ids:
        add({"id": entity_id, "kind": required_kinds.get(entity_id, "environment")}, source="dataset")

    proxy_entities = _field(_field(record, "proxy_scene", {}), "entities", [])
    for entity in _items(proxy_entities):
        add(entity, source="dataset")
    for entity in _items(_field(scene, "entities", [])):
        add(entity, source="scene_contract")
    for entity in _items(_field(plan, "entities", [])):
        add(entity, source="director_plan")

    return ordered, specs


def _event_map(record: Mapping[str, Any], plan: Mapping[str, Any], scene: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    events: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []

    def add(value: Any, *, source: str) -> None:
        if isinstance(value, Mapping):
            event_id = _field(value, "id", _field(value, "event_id"))
            if event_id is None:
                return
            event_id = str(event_id)
            item = events.setdefault(event_id, {"id": event_id, "source": source})
            item.update({str(key): _dump(child) for key, child in value.items()})
        else:
            event_id = str(value)
            events.setdefault(event_id, {"id": event_id, "source": source})
        if event_id not in ordered:
            ordered.append(event_id)

    oracle = record.get("oracle_expectations") or {}
    explicit = _strings(record.get("required_events"))
    if not explicit:
        explicit = _strings(oracle.get("event_order"))
    if not explicit:
        explicit = _strings(record.get("event_order"))
    for event_id in explicit:
        add(event_id, source="dataset")
    for event in _items(_field(scene, "events", [])):
        add(event, source="scene_contract")
    for event in _items(_field(plan, "events", [])):
        add(event, source="director_plan")
    return ordered, events


def _plan_hash(plan: Any) -> str | None:
    if plan is None:
        return None
    content_hash = getattr(plan, "content_hash", None)
    if callable(content_hash):
        return str(content_hash())
    return _hash(plan)


def _prompt_cues(prompt: str) -> list[str]:
    text = prompt.casefold()
    patterns = (
        ("orbit", r"\borbit(?:s|ed|ing)?\b|\bcircle(?:s|d|ing)?\b|\barc(?:s|ed|ing)?\b"),
        ("dolly", r"\bdolly(?:s|ed|ing)?\b|\bzoom(?:s|ed|ing)?\b|\bpush(?:es|ed|ing)?\s+in\b"),
        ("follow", r"\bfollow(?:s|ed|ing)?\b|\btracking\b|\btrack(?:s|ed|ing)?\b"),
        ("pan", r"\bpan(?:s|ned|ning)?\b"),
        ("tilt", r"\btilt(?:s|ed|ing)?\b"),
        ("crane", r"\bcrane(?:s|d|ing)?\b"),
    )
    matches = [(re.search(pattern, text).start(), cue) for cue, pattern in patterns if re.search(pattern, text)]
    return [cue for _position, cue in sorted(matches)]


def _status(value: bool | None) -> ObligationStatus:
    if value is None:
        return "pending"
    return "satisfied" if value else "failed"


class ObligationCompiler:
    """Compile structured case and plan inputs into deterministic records."""

    def compile(
        self,
        record: Mapping[str, Any] | None = None,
        director_plan: Any | None = None,
        scene_contract: Any | None = None,
        *,
        case_id: str | None = None,
        prompt: str | None = None,
        required_artifacts: Sequence[str] | None = None,
    ) -> ObligationCompilation:
        case = dict(record or {})
        plan = _dump(director_plan) or {}
        scene = _dump(scene_contract) or {}
        request = _field(plan, "request", {}) or {}
        prompt_text = str(prompt or case.get("prompt") or _field(request, "prompt", ""))
        resolved_case_id = str(case_id or case.get("case_id") or _field(request, "scene_id") or _field(scene, "scene_id") or "case")
        plan_hash = _plan_hash(director_plan)
        plan_id = _field(plan, "id")
        entity_ids, entities = _entity_map(case, plan, scene)
        event_ids, events = _event_map(case, plan, scene)
        records: list[ObligationRecord] = []
        seen_keys: set[tuple[str, str]] = set()

        def add(
            kind: ObligationKind,
            key: str,
            expected: Any,
            pass_rule: str,
            *,
            sources: Sequence[str],
            required: bool = True,
            applicable: bool = True,
            planned: ObligationStatus = "pending",
            refs: Sequence[str] = (),
        ) -> ObligationRecord | None:
            unique_key = (str(kind), str(key))
            if unique_key in seen_keys:
                return None
            seen_keys.add(unique_key)
            obligation_id = _stable_id(resolved_case_id, str(kind), str(key))
            status = "not_applicable" if not applicable else planned
            record_item = ObligationRecord(
                obligation_id=obligation_id,
                case_id=resolved_case_id,
                kind=kind,
                required=required if applicable else False,
                applicable=applicable,
                expected=expected,
                pass_rule=pass_rule,
                evidence_sources=list(dict.fromkeys(str(item) for item in sources if str(item).strip())),
                evidence_refs=list(dict.fromkeys(str(item) for item in refs if str(item).strip())),
                planned_status=status,  # type: ignore[arg-type]
                implemented_status=status if not applicable else "pending",
                executed_status=status if not applicable else "pending",
                visible_status=status if not applicable else "pending",
                judged_status=status if not applicable else "pending",
                traceability={
                    "plan": f"plan:{plan_id or 'unbound'}:{obligation_id}",
                    "source_coverage": f"coverage:{obligation_id}",
                    "trusted_observer": f"trusted_observer:{obligation_id}",
                    "telemetry": f"telemetry:{obligation_id}",
                    "evaluator": f"evaluation:{obligation_id}",
                },
            )
            records.append(record_item)
            return record_item

        # 1. Entity presence is always derived from structured entities.  A
        # prompt mentioning a person cannot silently invent a character when
        # the actual plan is object-only.
        plan_entity_ids = {str(_field(item, "id")) for item in _items(_field(plan, "entities", [])) if _field(item, "id") is not None}
        for entity_id in entity_ids:
            spec = entities.get(entity_id, {})
            add(
                "entity",
                entity_id,
                {
                    "entity_id": entity_id,
                    "entity_kind": _entity_kind(spec),
                    "role": spec.get("role"),
                    "label": spec.get("label", entity_id),
                },
                "trusted scene observation contains the required entity identity",
                sources=[str(spec.get("source", "dataset")), "director_plan" if entity_id in plan_entity_ids else "scene_contract"],
                planned=_status(entity_id in plan_entity_ids),
            )

        # 2. Events, order, and timing.
        plan_event_ids = {str(_field(item, "id")) for item in _items(_field(plan, "events", [])) if _field(item, "id") is not None}
        for index, event_id in enumerate(event_ids):
            event = events.get(event_id, {})
            has_interval = _field(event, "start") is not None and _field(event, "end") is not None
            add(
                "event",
                event_id,
                {
                    "event_id": event_id,
                    "action": event.get("action"),
                    "depends_on": list(_strings(event.get("depends_on"))),
                    "order_index": index,
                },
                "trusted observer and blind visual evidence establish the event occurrence",
                sources=[str(event.get("source", "dataset")), "director_plan" if event_id in plan_event_ids else "scene_contract"],
                planned=_status(event_id in plan_event_ids),
            )
            if has_interval:
                add(
                    "event_timing",
                    event_id,
                    {"event_id": event_id, "start": float(event["start"]), "end": float(event["end"])},
                    "observed event evidence falls within the declared start/end interval",
                    sources=["director_plan" if event_id in plan_event_ids else str(event.get("source", "scene_contract"))],
                    planned=_status(event_id in plan_event_ids),
                )
        if len(event_ids) > 1:
            add(
                "event_order",
                "sequence",
                {"event_order": event_ids},
                "all required events occur in the declared order; concurrency is explicit in the plan",
                sources=["dataset" if case.get("oracle_expectations", {}).get("event_order") else "director_plan"],
                planned=_status(bool(plan_event_ids) and set(event_ids).issubset(plan_event_ids)),
            )

        # 3. Participant and target bindings.
        for event_id in event_ids:
            event = events.get(event_id, {})
            for role, field_name, kind in (
                ("participant", "participant_ids", "participant"),
                ("target", "target_ids", "target"),
            ):
                for entity_id in _strings(event.get(field_name)):
                    add(
                        kind,  # type: ignore[arg-type]
                        f"{event_id}.{entity_id}",
                        {"event_id": event_id, "entity_id": entity_id, "role": role},
                        f"event {role} binding references an observed entity with the declared identity",
                        sources=["director_plan" if event_id in plan_event_ids else str(event.get("source", "scene_contract"))],
                        planned=_status(event_id in plan_event_ids and entity_id in entity_ids),
                    )

        # 4. Ownership and transfer windows.  Use explicit lifecycle records
        # first, and only infer a lifecycle from a structured handoff event.
        interactions = _items(_field(plan, "interactions", []))
        for event_id in event_ids:
            event = events.get(event_id, {})
            action = str(event.get("action") or "").casefold()
            if action not in {"handoff", "transfer"}:
                continue
            participants = _strings(event.get("participant_ids"))
            targets = _strings(event.get("target_ids"))
            props = [item for item in targets if _entity_kind(entities.get(item, {})) == "prop"]
            if participants and props:
                interactions.append(
                    {
                        "id": f"inferred-{event_id}",
                        "prop_id": props[0],
                        "giver_id": participants[0],
                        "receiver_id": participants[-1],
                        "transfer_event_id": event_id,
                        "attach_event_id": None,
                        "detach_event_id": None,
                        "final_owner_id": participants[-1],
                        "source": "director_plan",
                    }
                )
        seen_interactions: set[str] = set()
        for lifecycle in interactions:
            lifecycle_id = str(_field(lifecycle, "id") or _field(lifecycle, "transfer_event_id") or "interaction")
            if lifecycle_id in seen_interactions:
                continue
            seen_interactions.add(lifecycle_id)
            prop_id = _field(lifecycle, "prop_id")
            transfer_event_id = _field(lifecycle, "transfer_event_id")
            transfer_event = events.get(str(transfer_event_id), {}) if transfer_event_id else {}
            expected = {
                "interaction_id": lifecycle_id,
                "prop_id": prop_id,
                "giver_id": _field(lifecycle, "giver_id"),
                "receiver_id": _field(lifecycle, "receiver_id"),
                "attach_event_id": _field(lifecycle, "attach_event_id"),
                "transfer_event_id": transfer_event_id,
                "detach_event_id": _field(lifecycle, "detach_event_id"),
                "final_owner_id": _field(lifecycle, "final_owner_id"),
                "final_support_id": _field(lifecycle, "final_support_id"),
            }
            add(
                "ownership_transition",
                lifecycle_id,
                expected,
                "trusted observer ownership/contact evidence matches giver, receiver, and final owner",
                sources=["director_plan"],
                planned=_status(bool(plan.get("interactions")) or lifecycle_id.startswith("inferred-")),
            )
            if transfer_event_id:
                add(
                    "transfer_window",
                    lifecycle_id,
                    {
                        "event_id": str(transfer_event_id),
                        "prop_id": prop_id,
                        "giver_id": _field(lifecycle, "giver_id"),
                        "receiver_id": _field(lifecycle, "receiver_id"),
                        "start": _field(transfer_event, "start"),
                        "end": _field(transfer_event, "end"),
                    },
                    "the prop has one bounded transfer interval with giver and receiver contact",
                    sources=["director_plan"],
                    planned=_status(str(transfer_event_id) in plan_event_ids),
                )
            add(
                "contact_support",
                lifecycle_id,
                {
                    "prop_id": prop_id,
                    "attach_event_id": _field(lifecycle, "attach_event_id"),
                    "detach_event_id": _field(lifecycle, "detach_event_id"),
                    "final_support_id": _field(lifecycle, "final_support_id"),
                },
                "attachment and final support contact are observed without penetration or floating",
                sources=["director_plan", "trusted_observer"],
                planned=_status(bool(plan.get("interactions")) or lifecycle_id.startswith("inferred-")),
            )

        # Structured scene relations are also support obligations, including
        # object-only cases that have no character or handoff.
        for index, relation in enumerate(_items(_field(scene, "relations", []))):
            add(
                "contact_support",
                f"relation.{index}.{_field(relation, 'type', 'relation')}",
                {
                    "relation_type": _field(relation, "type"),
                    "subject": _field(relation, "subject"),
                    "object": _field(relation, "object"),
                },
                "trusted observer geometry confirms the declared support/contact relation",
                sources=["scene_contract", "trusted_observer"],
                planned="satisfied",
            )

        # 5. Actor/object trajectories.  Requirements are sourced from the
        # plan or SceneContract; no actor trajectory is invented for objects.
        trajectory_summary = _field(plan, "trajectory_summary", {}) or {}
        trajectory_entities = _field(trajectory_summary, "entities", {}) or {}
        trajectory_requirements = _items(_field(scene, "trajectory_requirements", []))
        requirement_by_entity: dict[str, dict[str, Any]] = {}
        for requirement in trajectory_requirements:
            entity_id = _field(requirement, "entity_id")
            if entity_id is not None:
                requirement_by_entity[str(entity_id)] = requirement
        for entity_id in entity_ids:
            kind = _entity_kind(entities.get(entity_id, {}))
            requirement = requirement_by_entity.get(entity_id, {})
            summary = _field(trajectory_entities, entity_id, {}) if isinstance(trajectory_entities, Mapping) else {}
            required_event_ids = _strings(_field(requirement, "required_event_ids"))
            if not required_event_ids:
                required_event_ids = [
                    event_id
                    for event_id in event_ids
                    if entity_id in _strings(events.get(event_id, {}).get("participant_ids"))
                    or entity_id in _strings(events.get(event_id, {}).get("target_ids"))
                ]
            if not required_event_ids and not summary:
                continue
            add(
                "trajectory",
                entity_id,
                {
                    "entity_id": entity_id,
                    "entity_kind": kind,
                    "required_event_ids": required_event_ids,
                    "minimum_states": _field(requirement, "minimum_states", 1),
                    "required_attachment_actions": _strings(_field(requirement, "required_attachment_actions")),
                },
                "trusted observer states/primitives cover every declared trajectory event",
                sources=["director_plan" if entity_id in trajectory_entities else "scene_contract"],
                planned=_status(entity_id in trajectory_entities or bool(requirement)),
            )

        # 6. Camera coverage and visibility.  Coverage is a requirement only
        # when the case/plan explicitly names an event or shot.
        oracle = case.get("oracle_expectations") or {}
        required_camera_events = _strings(case.get("required_camera_events")) or _strings(oracle.get("required_camera_events"))
        camera = _field(_field(case, "proxy_scene", {}), "camera", {}) or {}
        required_camera_events = required_camera_events or _strings(_field(camera, "must_show_events"))
        camera_plan = _field(plan, "camera_plan", {}) or {}
        shots = _items(_field(camera_plan, "shots", []))
        if not shots and isinstance(camera, Mapping):
            shots = _items(camera.get("shots"))
        transfer_expectations = [
            item.expected
            for item in records
            if item.kind == "ownership_transition" and isinstance(item.expected, Mapping)
        ]
        for index, shot in enumerate(shots):
            shot_event_ids = _strings(_field(shot, "required_event_ids"))
            target_ids = _strings(_field(shot, "target_ids"))
            is_required = bool(shot_event_ids or target_ids or set(shot_event_ids) & set(required_camera_events))
            if not is_required:
                continue
            shot_id = str(_field(shot, "shot_id") or f"shot_{index + 1}")
            coverage = add(
                "camera_coverage",
                shot_id,
                {
                    "shot_id": shot_id,
                    "event_ids": shot_event_ids,
                    "target_ids": target_ids,
                    "trajectory_type": _field(shot, "trajectory_type", "hold"),
                },
                "sampled frames cover every required event and declared target in this shot",
                sources=["director_plan"],
                required=True,
                planned="satisfied",
            )
            add(
                "camera_visibility",
                shot_id,
                {
                    "shot_id": shot_id,
                    "event_ids": shot_event_ids,
                    "target_ids": target_ids,
                    "visibility_predicates": _field(shot, "visibility_predicates", {}) or {},
                },
                "all required targets remain visibly identifiable in the sampled frames",
                sources=["director_plan", "trusted_observer"],
                required=True,
                planned="satisfied" if target_ids else "pending",
            )
            for transfer in transfer_expectations:
                transfer_event_id = transfer.get("transfer_event_id")
                three_way = [transfer.get("giver_id"), transfer.get("receiver_id"), transfer.get("prop_id")]
                if transfer_event_id and transfer_event_id in shot_event_ids:
                    expected_targets = [str(item) for item in three_way if item]
                    covered = set(expected_targets).issubset(set(target_ids))
                    add(
                        "camera_visibility",
                        f"{shot_id}.three_way.{transfer_event_id}",
                        {
                            "shot_id": shot_id,
                            "event_id": transfer_event_id,
                            "target_ids": expected_targets,
                            "minimum_targets": 3,
                        },
                        "giver, receiver, and prop are simultaneously identifiable during transfer",
                        sources=["director_plan", "trusted_observer"],
                        required=True,
                        planned=_status(covered),
                    )
        if not shots:
            for event_id in required_camera_events:
                add(
                    "camera_coverage",
                    f"event.{event_id}",
                    {"event_ids": [event_id], "target_ids": []},
                    "sampled frames cover the required event",
                    sources=["dataset"],
                    required=True,
                    planned="failed",
                )

        # 7. Only prompt-explicit camera motion becomes a required motion
        # obligation.  Static-camera prompts therefore have no innovation
        # obligation and are recorded as an N/A dimension, never as 100.
        cues = _prompt_cues(prompt_text)
        if cues:
            planned_cues = {
                str(_field(shot, "camera_cue") or _field(shot, "trajectory_type"))
                for shot in shots
            }
            for cue in cues:
                add(
                    "camera_motion",
                    cue,
                    {"cue": cue},
                    "camera telemetry and frames show the prompt-explicit camera motion cue",
                    sources=["prompt", "director_plan"],
                    planned=_status(cue in planned_cues or any(cue == str(_field(shot, "trajectory_type")) for shot in shots)),
                )
        else:
            # Kept outside the record list so downstream scoring cannot turn a
            # non-applicable dimension into a numeric perfect score.
            na_dimensions = ["camera_motion"]

        # 8. Artifact and provenance obligations are intentionally explicit;
        # an unavailable artifact is not represented by a score of zero or
        # one hundred.
        artifacts = list(required_artifacts or _strings(case.get("required_artifacts")))
        if not artifacts:
            artifacts = ["candidate.blend", "proxy.mp4", "telemetry.json"]
        for artifact in dict.fromkeys(artifacts):
            add(
                "artifact",
                artifact,
                {"artifact": artifact, "case_id": resolved_case_id},
                "artifact exists, is readable, and is bound to this case's run manifest",
                sources=["dataset", "artifact_manifest"],
                planned="pending",
            )
        provenance_expected = {
            "case_id": resolved_case_id,
            "prompt_hash": _hash(prompt_text),
            "plan_hash": plan_hash,
            "provider_fingerprint": _field(plan, "provider_fingerprint") or _field(request, "provider"),
            "policy_fingerprint": _field(plan, "policy_fingerprint") or _field(request, "policy"),
        }
        add(
            "provenance",
            "run_identity",
            provenance_expected,
            "run manifest, observer manifest, source hash, and plan hash agree with the frozen case identity",
            sources=["director_plan", "artifact_manifest", "trusted_observer"],
            planned=_status(plan is not None and bool(plan_hash)),
        )

        obligation_ids = [item.obligation_id for item in records]
        traceability = {
            "plan": list(obligation_ids) if plan else [],
            "source_coverage": list(obligation_ids),
            "trusted_observer": list(obligation_ids),
            "telemetry": list(obligation_ids),
            "evaluator": list(obligation_ids),
        }
        compilation_payload = {
            "schema_version": OBLIGATION_SCHEMA_VERSION,
            "case_id": resolved_case_id,
            "prompt_hash": _hash(prompt_text),
            "director_plan_id": plan_id,
            "director_plan_hash": plan_hash,
            "obligations": [item.model_dump(mode="json") for item in records],
            "obligation_ids": obligation_ids,
            "na_dimensions": locals().get("na_dimensions", []),
            "traceability": traceability,
        }
        return ObligationCompilation(
            schema_version=OBLIGATION_SCHEMA_VERSION,
            case_id=resolved_case_id,
            prompt_hash=_hash(prompt_text),
            director_plan_id=str(plan_id) if plan_id is not None else None,
            director_plan_hash=plan_hash,
            obligations=records,
            obligation_ids=obligation_ids,
            na_dimensions=list(locals().get("na_dimensions", [])),
            traceability=traceability,
            fingerprint=_hash(compilation_payload),
        )


def compile_obligations(
    record: Mapping[str, Any] | None = None,
    director_plan: Any | None = None,
    scene_contract: Any | None = None,
    *,
    case_id: str | None = None,
    prompt: str | None = None,
    required_artifacts: Sequence[str] | None = None,
) -> ObligationCompilation:
    """Convenience wrapper around :class:`ObligationCompiler`."""

    return ObligationCompiler().compile(
        record,
        director_plan,
        scene_contract,
        case_id=case_id,
        prompt=prompt,
        required_artifacts=required_artifacts,
    )


def validate_obligation_completeness(
    value: ObligationCompilation | Sequence[ObligationRecord],
    *,
    expected_ids: Sequence[str] | None = None,
) -> ObligationGateReport:
    """Require every expected applicable obligation to be present.

    The function raises on a missing required ID instead of returning a
    numeric partial score.  This is the schema/gate used before coverage or
    runtime evidence is accepted.
    """

    if isinstance(value, ObligationCompilation):
        records = value.obligations
        required = list(expected_ids or value.required_obligation_ids)
    else:
        records = list(value)
        required = list(expected_ids or [item.obligation_id for item in records if item.is_required])
    present = [item.obligation_id for item in records]
    duplicate_ids = sorted({item for item in present if present.count(item) > 1})
    missing = [item for item in required if item not in set(present)]
    failures = [f"duplicate obligation:{item}" for item in duplicate_ids]
    failures.extend(f"missing required obligation:{item}" for item in missing)
    report = ObligationGateReport(
        status="pass" if not failures else "failed",
        required_ids=list(dict.fromkeys(required)),
        present_ids=list(dict.fromkeys(present)),
        missing_ids=list(dict.fromkeys(missing)),
        failures=failures,
    )
    if failures:
        raise ValueError("; ".join(failures))
    return report


def validate_required_obligations(
    value: ObligationCompilation | Sequence[ObligationRecord],
    *,
    expected_ids: Sequence[str] | None = None,
) -> ObligationGateReport:
    """Compatibility alias for the fail-closed completeness gate."""

    return validate_obligation_completeness(value, expected_ids=expected_ids)


def bind_obligations_to_plan(plan: Any, compilation: ObligationCompilation) -> Any:
    """Return a plan copy carrying the compiler IDs in its own contract."""

    if not hasattr(plan, "model_copy"):
        raise TypeError("plan must be a DirectorPlan-like model with model_copy")
    request = getattr(plan, "request", None)
    if request is not None and hasattr(request, "model_copy"):
        obligations = dict(getattr(request, "obligations", {}) or {})
        obligations["obligation_ids"] = list(compilation.obligation_ids)
        request = request.model_copy(update={"obligations": obligations})
    return plan.model_copy(update={"request": request, "obligation_ids": list(compilation.obligation_ids)})


def attach_obligation_ids(
    payload: Mapping[str, Any] | Any,
    compilation: ObligationCompilation,
    *,
    stage: str,
) -> dict[str, Any]:
    """Attach identity-only trace anchors to telemetry/evaluator artifacts.

    Callers must invoke this after the blind Judge payload is built; the
    helper is intentionally not used by the Judge adapters, so these IDs do
    not leak plan semantics into the primary visual judgement.
    """

    if hasattr(payload, "model_dump"):
        base = payload.model_dump(mode="json")
    elif isinstance(payload, Mapping):
        base = dict(payload)
    else:
        raise TypeError("payload must be a mapping or Pydantic model")
    base["obligation_ids"] = list(compilation.obligation_ids)
    base["obligation_trace"] = {
        "schema_version": OBLIGATION_SCHEMA_VERSION,
        "stage": str(stage),
        "compilation_fingerprint": compilation.fingerprint,
        "refs": {str(item): f"{stage}:{item}" for item in compilation.obligation_ids},
    }
    return base


__all__ = [
    "OBLIGATION_SCHEMA_VERSION",
    "ObligationCompiler",
    "ObligationCompilation",
    "ObligationGateReport",
    "ObligationKind",
    "ObligationRecord",
    "ObligationStatus",
    "attach_obligation_ids",
    "bind_obligations_to_plan",
    "compile_obligations",
    "validate_obligation_completeness",
    "validate_required_obligations",
]
