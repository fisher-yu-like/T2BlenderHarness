"""Strict contracts for DirectorAgent planning.

DirectorPlan is the typed boundary between prompt interpretation, event
scheduling, trajectory/camera composition, Blender projection, and evaluation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import CameraPlan


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


EntityKind = Literal["actor", "prop", "support", "environment"]
UncertaintySeverity = Literal["soft", "hard"]


class DirectorRequest(ContractModel):
    prompt: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    duration_s: float = Field(gt=0)
    fps: int = Field(gt=0)
    provider: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    # Stable case obligations constrain identifier traceability only.  They do
    # not replace the exact prompt or supply an implementation template.
    obligations: dict[str, list[str]] = Field(default_factory=dict)


class DirectorEntity(ContractModel):
    id: str = Field(min_length=1)
    kind: EntityKind
    role: str = Field(min_length=1)
    label: str = Field(min_length=1)
    attributes: dict[str, Any] = Field(default_factory=dict)


class DirectorEvent(ContractModel):
    id: str = Field(min_length=1)
    action: str = Field(min_length=1)
    participant_ids: list[str]
    target_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    concurrency_group: str | None = None
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_event(self) -> "DirectorEvent":
        if not self.participant_ids:
            raise ValueError("event must have at least one participant")
        if len(self.participant_ids) != len(set(self.participant_ids)):
            raise ValueError("event participant IDs must be unique")
        if len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError("event target IDs must be unique")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("event dependencies must be unique")
        if self.end < self.start:
            raise ValueError("event end must be greater than or equal to start")
        return self


class InteractionLifecycle(ContractModel):
    id: str = Field(min_length=1)
    prop_id: str = Field(min_length=1)
    giver_id: str | None = None
    receiver_id: str | None = None
    attach_event_id: str
    transfer_event_id: str | None = None
    detach_event_id: str
    final_owner_id: str | None = None
    final_support_id: str | None = None


class DirectorDecisionEvidence(ContractModel):
    id: str = Field(min_length=1)
    source: Literal["prompt", "provider", "dataset", "policy", "critic"]
    prompt_span: tuple[int, int] | None = None
    quoted_text: str | None = None
    claim: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prompt_span(self) -> "DirectorDecisionEvidence":
        if self.prompt_span is not None and self.prompt_span[1] < self.prompt_span[0]:
            raise ValueError("evidence prompt_span must be ordered")
        return self


class DirectorAssumption(ContractModel):
    id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    supported_by_evidence_ids: list[str]

    @model_validator(mode="after")
    def validate_support(self) -> "DirectorAssumption":
        if not self.supported_by_evidence_ids:
            raise ValueError("unsupported assumption")
        if len(self.supported_by_evidence_ids) != len(set(self.supported_by_evidence_ids)):
            raise ValueError("assumption evidence IDs must be unique")
        return self


class DirectorUncertainty(ContractModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: UncertaintySeverity
    resolved: bool


class DirectorPlan(ContractModel):
    id: str = Field(min_length=1)
    request: DirectorRequest
    entities: list[DirectorEntity]
    events: list[DirectorEvent]
    interactions: list[InteractionLifecycle] = Field(default_factory=list)
    assumptions: list[DirectorAssumption] = Field(default_factory=list)
    uncertainties: list[DirectorUncertainty] = Field(default_factory=list)
    evidence: list[DirectorDecisionEvidence] = Field(default_factory=list)
    # These fields make the DirectorPlan the single handoff contract.  The
    # legacy projection still exposes the same objects for compatibility.
    trajectory_summary: dict[str, Any] = Field(default_factory=dict)
    camera_plan: CameraPlan | None = None
    coverage_obligations: list[str] = Field(default_factory=list)
    provider_fingerprint: str = Field(min_length=1)
    policy_fingerprint: str = Field(min_length=1)

    @property
    def entity_ids(self) -> set[str]:
        return {entity.id for entity in self.entities}

    @property
    def event_ids(self) -> set[str]:
        return {event.id for event in self.events}

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # ensure_ascii=False matches the codegen payload hash and the
        # provenance canonical hash; with the default escaping, any
        # non-ASCII prompt (curly quotes and similar) produced a different
        # hash on the codegen boundary and failed the plan-binding audit.
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @model_validator(mode="after")
    def validate_plan(self) -> "DirectorPlan":
        entity_ids = [entity.id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique")
        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")
        evidence_ids = [item.id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence IDs must be unique")

        entity_set = set(entity_ids)
        event_set = set(event_ids)
        evidence_set = set(evidence_ids)
        actor_ids = {entity.id for entity in self.entities if entity.kind == "actor"}
        prop_ids = {entity.id for entity in self.entities if entity.kind == "prop"}

        for event in self.events:
            unknown_participants = set(event.participant_ids) - entity_set
            if unknown_participants:
                raise ValueError(f"unknown participant IDs: {sorted(unknown_participants)}")
            unknown_targets = set(event.target_ids) - entity_set
            if unknown_targets:
                raise ValueError(f"unknown target IDs: {sorted(unknown_targets)}")
            unknown_dependencies = set(event.depends_on) - event_set
            if unknown_dependencies:
                raise ValueError(f"unknown dependency IDs: {sorted(unknown_dependencies)}")
            if event.end > self.request.duration_s:
                raise ValueError(f"event {event.id} exceeds request duration")
        self._validate_acyclic_events()

        for lifecycle in self.interactions:
            if lifecycle.prop_id not in prop_ids:
                raise ValueError(f"interaction prop references unknown prop: {lifecycle.prop_id}")
            for role_name in ("giver_id", "receiver_id", "final_owner_id"):
                actor_id = getattr(lifecycle, role_name)
                if actor_id is not None and actor_id not in actor_ids:
                    label = role_name.replace("_", " ")
                    raise ValueError(f"interaction {label} must reference an actor: {actor_id}")
            for event_field in ("attach_event_id", "transfer_event_id", "detach_event_id"):
                lifecycle_event = getattr(lifecycle, event_field)
                if lifecycle_event is not None and lifecycle_event not in event_set:
                    raise ValueError(f"interaction {event_field} references unknown event: {lifecycle_event}")

        for assumption in self.assumptions:
            missing = set(assumption.supported_by_evidence_ids) - evidence_set
            if missing:
                raise ValueError(f"unsupported assumption evidence IDs: {sorted(missing)}")
        for evidence in self.evidence:
            if evidence.prompt_span is not None:
                start, end = evidence.prompt_span
                if start < 0 or end > len(self.request.prompt):
                    raise ValueError(f"evidence {evidence.id} prompt_span exceeds prompt bounds")
        return self

    def _validate_acyclic_events(self) -> None:
        dependencies = {event.id: set(event.depends_on) for event in self.events}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(event_id: str) -> None:
            if event_id in visited:
                return
            if event_id in visiting:
                raise ValueError("dependency cycle detected")
            visiting.add(event_id)
            for dependency_id in dependencies[event_id]:
                visit(dependency_id)
            visiting.remove(event_id)
            visited.add(event_id)

        for event_id in dependencies:
            visit(event_id)


class DirectorResult(ContractModel):
    director_plan: DirectorPlan

    @property
    def scene_id(self) -> str:
        return self.director_plan.request.scene_id

    @property
    def director_plan_hash(self) -> str:
        return self.director_plan.content_hash()
