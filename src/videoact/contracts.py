"""Versioned runtime contracts shared by planning, execution, and evaluation."""

from __future__ import annotations

import hashlib
import json
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EntitySpec(ContractModel):
    id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    role: str = Field(min_length=1)


class EventSpec(ContractModel):
    id: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    description: str = ""
    target_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_interval(self) -> "EventSpec":
        if self.end < self.start:
            raise ValueError("event end must be greater than or equal to start")
        return self


class RelationSpec(ContractModel):
    type: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    object: str = Field(min_length=1)


class TrajectoryRequirement(ContractModel):
    """Prompt-derived trajectory evidence required for one entity.

    The evaluator consumes these declarations instead of assuming that every
    scene follows a fixed walk/grasp/carry/release template.
    """

    entity_id: str = Field(min_length=1)
    required_event_ids: list[str] = Field(default_factory=list)
    minimum_states: int = Field(default=1, ge=1)
    require_phase_primitives: bool = False
    required_attachment_actions: list[Literal["attach", "detach"]] = Field(default_factory=list)


class SceneContract(ContractModel):
    scene_id: str = Field(min_length=1)
    duration_s: float = Field(gt=0)
    fps: int = Field(gt=0)
    entities: list[EntitySpec]
    events: list[EventSpec]
    relations: list[RelationSpec] = Field(default_factory=list)
    must_show: list[str] = Field(default_factory=list)
    physics_constraints: list[str] = Field(default_factory=list)
    camera_constraints: list[str] = Field(default_factory=list)
    trajectory_requirements: list[TrajectoryRequirement] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references_and_bounds(self) -> "SceneContract":
        entity_ids = [entity.id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique")

        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")

        if any(event.end > self.duration_s for event in self.events):
            raise ValueError("event times must be within scene duration")

        if any(
            earlier.start > later.start
            for earlier, later in zip(self.events, self.events[1:])
        ):
            raise ValueError("events must be ordered by start time")

        unknown_required = set(self.must_show) - set(event_ids)
        if unknown_required:
            raise ValueError(f"must_show references unknown events: {sorted(unknown_required)}")

        entity_set = set(entity_ids)
        for relation in self.relations:
            if relation.subject not in entity_set or relation.object not in entity_set:
                raise ValueError(
                    f"unknown entity in relation: {relation.subject}->{relation.object}"
                )

        for event in self.events:
            unknown_targets = set(event.target_ids) - entity_set
            if unknown_targets:
                raise ValueError(
                    f"event {event.id} references unknown entity: {sorted(unknown_targets)}"
                )
        for requirement in self.trajectory_requirements:
            if requirement.entity_id not in entity_set:
                raise ValueError(f"trajectory requirement references unknown entity: {requirement.entity_id}")
            unknown_events = set(requirement.required_event_ids) - set(event_ids)
            if unknown_events:
                raise ValueError(
                    f"trajectory requirement references unknown events: {sorted(unknown_events)}"
                )
        return self


class Timebase(ContractModel):
    fps: int = Field(gt=0)
    frame_start: int = Field(ge=1)
    frame_end: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_frame_range(self) -> "Timebase":
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        return self


class EntityState(ContractModel):
    frame: int = Field(ge=1)
    position: tuple[float, float, float]
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: bool = True


class MotionPrimitive(ContractModel):
    type: Literal[
        "linear",
        "ease_in_out",
        "hold",
        "look_at",
        "follow",
        "orbit",
        "dolly",
        "arc",
        "s_curve",
        "zigzag",
        "bezier",
    ]
    start_frame: int = Field(ge=1)
    end_frame: int = Field(ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interval(self) -> "MotionPrimitive":
        if self.end_frame < self.start_frame:
            raise ValueError("motion primitive end_frame must be >= start_frame")
        return self


class AttachmentEvent(ContractModel):
    frame: int = Field(ge=1)
    subject_id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    action: Literal["attach", "transfer", "detach"]


class EntityTrajectory(ContractModel):
    states: list[EntityState] = Field(default_factory=list)
    motion_primitives: list[MotionPrimitive] = Field(default_factory=list)
    attachment_events: list[AttachmentEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_monotonic_states(self) -> "EntityTrajectory":
        frames = [state.frame for state in self.states]
        if frames != sorted(set(frames)):
            raise ValueError("entity state frames must be strictly increasing")
        return self


class CameraShot(ContractModel):
    shot_id: str = Field(min_length=1)
    start_frame: int = Field(ge=1)
    end_frame: int = Field(ge=1)
    target_ids: list[str] = Field(default_factory=list)
    intent: str = Field(min_length=1)
    lens_mm: float = Field(gt=0)
    distance_range: tuple[float, float]
    required_event_ids: list[str] = Field(default_factory=list)
    trajectory_type: Literal["hold", "follow", "orbit", "dolly"] = "hold"
    visibility_predicates: dict[str, str] = Field(default_factory=dict)
    max_occlusion: float = Field(default=1.0, ge=0, le=1)
    continuity_group: str | None = None
    innovation_intent_evidence_id: str | None = None

    @model_validator(mode="after")
    def validate_interval_and_distance(self) -> "CameraShot":
        if self.end_frame < self.start_frame:
            raise ValueError("camera shot end_frame must be >= start_frame")
        if self.distance_range[0] <= 0 or self.distance_range[1] < self.distance_range[0]:
            raise ValueError("camera distance_range must be positive and ordered")
        return self


class CameraPlan(ContractModel):
    shots: list[CameraShot] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_shots(self) -> "CameraPlan":
        ids = [shot.shot_id for shot in self.shots]
        if len(ids) != len(set(ids)):
            raise ValueError("camera shot IDs must be unique")
        return self


class EventObservability(ContractModel):
    event_id: str = Field(min_length=1)
    covered_by_shots: list[str] = Field(default_factory=list)
    target_visible_predicate: str = Field(min_length=1)


class TrajectoryPlan(ContractModel):
    timebase: Timebase
    entities: dict[str, EntityTrajectory]
    camera: CameraPlan
    event_observability: list[EventObservability] = Field(default_factory=list)
    validation_intents: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_ranges(self) -> "TrajectoryPlan":
        start, end = self.timebase.frame_start, self.timebase.frame_end
        for entity_id, trajectory in self.entities.items():
            for state in trajectory.states:
                if not start <= state.frame <= end:
                    raise ValueError(f"entity {entity_id} state frame is outside timebase")
            for primitive in trajectory.motion_primitives:
                if primitive.start_frame < start or primitive.end_frame > end:
                    raise ValueError(f"entity {entity_id} primitive is outside timebase")
            for attachment in trajectory.attachment_events:
                if not start <= attachment.frame <= end:
                    raise ValueError(f"entity {entity_id} attachment frame is outside timebase")
        for shot in self.camera.shots:
            if shot.start_frame < start or shot.end_frame > end:
                raise ValueError(f"camera shot {shot.shot_id} is outside timebase")
        return self


class ExecutionResult(ContractModel):
    status: Literal["success", "failed", "timeout"]
    backend: Literal["fake", "cli", "mcp"]
    command: list[str] = Field(default_factory=list)
    request: dict[str, Any] | None = None
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    artifact_paths: dict[str, str] = Field(default_factory=dict)
    duration_s: float = Field(default=0.0, ge=0)
    fallback_used: bool = False
    error: str | None = None


class Finding(ContractModel):
    failure_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    category: str = Field(min_length=1)
    severity: Literal["info", "warning", "error", "hard"]
    message: str = Field(min_length=1)
    root_cause_id: str | None = None
    evidence: list[str] = Field(default_factory=list)
    repair_route: Literal[
        "scene_contract_repair",
        "trajectory_repair",
        "camera_repair",
        "runtime_repair",
        "candidate_recovery",
    ]


class RunManifest(ContractModel):
    run_id: str = Field(min_length=1)
    scene_id: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    plan_hash: str = Field(min_length=1)
    backend: Literal["fake", "cli", "mcp"]
    blender_version: str | None = None
    frame_start: int = Field(ge=1)
    frame_end: int = Field(ge=1)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_frames(self) -> "RunManifest":
        if self.frame_end < self.frame_start:
            raise ValueError("frame_end must be greater than or equal to frame_start")
        return self

    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class RunResult(ContractModel):
    run_id: str = Field(min_length=1)
    status: Literal["success", "failed", "exhausted"]
    selected_attempt: int | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    final_score: float | None = Field(default=None, ge=0, le=100)
