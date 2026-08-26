"""Collision-aware multi-entity trajectory composition for DirectorAgent."""

from __future__ import annotations

import math

from pydantic import Field, model_validator

from .contracts import AttachmentEvent, EntityState, EntityTrajectory, MotionPrimitive, Timebase
from .director_contracts import ContractModel, DirectorRequest
from .director_prompt import PromptInterpretation
from .director_schedule import DirectorSchedule

Point3 = tuple[float, float, float]


class DirectorTrajectories(ContractModel):
    timebase: Timebase
    entities: dict[str, EntityTrajectory]
    current_owner_by_event: dict[str, str] = Field(default_factory=dict)
    final_support_by_prop: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_entities(self) -> "DirectorTrajectories":
        if not self.entities:
            raise ValueError("director trajectories require at least one entity")
        return self


class MultiEntityTrajectoryComposer:
    _HAND_OFFSET: Point3 = (0.65, -0.05, 1.35)

    def __init__(self, *, lane_spacing: float = 2.0, minimum_lane_distance: float = 1.5) -> None:
        self.lane_spacing = lane_spacing
        self.minimum_lane_distance = minimum_lane_distance

    def compose(
        self,
        request: DirectorRequest,
        interpretation: PromptInterpretation,
        schedule: DirectorSchedule,
    ) -> DirectorTrajectories:
        actor_ids = [entity.id for entity in interpretation.entities if entity.kind == "actor"]
        prop_ids = [entity.id for entity in interpretation.entities if entity.kind == "prop"]
        lanes = self._lanes(actor_ids)
        self._validate_lanes(lanes)

        frame_end = max(1, round(request.duration_s * request.fps))
        frame = lambda seconds: max(1, min(frame_end, round(seconds * request.fps) + 1))
        entities: dict[str, EntityTrajectory] = {}
        current_owner_by_event: dict[str, str] = {}
        final_support_by_prop: dict[str, str] = {}

        for actor_id in actor_ids:
            states = [EntityState(frame=1, position=lanes[actor_id])]
            primitives: list[MotionPrimitive] = []
            previous_frame = 1
            for event in schedule.events:
                if actor_id not in event.participant_ids:
                    continue
                position = self._actor_position(actor_id, event.action, lanes, event.id)
                event_frame = frame(event.end)
                states.append(EntityState(frame=event_frame, position=position))
                primitives.append(
                    MotionPrimitive(
                        type=self._primitive_for(event.action),
                        start_frame=previous_frame,
                        end_frame=event_frame,
                        parameters={"event_id": event.id, "actor_id": actor_id},
                    )
                )
                previous_frame = event_frame
            entities[actor_id] = EntityTrajectory(
                states=self._unique_states(states, frame_end),
                motion_primitives=primitives,
            )

        for prop_id in prop_ids:
            states = [EntityState(frame=1, position=self._prop_start(prop_id))]
            attachments: list[AttachmentEvent] = []
            owner: str | None = None
            for event in schedule.events:
                if prop_id not in event.target_ids:
                    continue
                event_frame = frame(event.end)
                if event.action == "carry":
                    owner = event.participant_ids[0]
                    states.append(EntityState(frame=event_frame, position=self._hand_position(lanes[owner])))
                    attachments.append(
                        AttachmentEvent(
                            frame=frame(event.start),
                            subject_id=prop_id,
                            object_id=owner,
                            action="attach",
                        )
                    )
                elif event.action == "handoff":
                    owner = event.participant_ids[-1]
                    states.append(EntityState(frame=event_frame, position=self._hand_position(lanes[owner])))
                    attachments.append(
                        AttachmentEvent(
                            frame=frame(event.start),
                            subject_id=prop_id,
                            object_id=owner,
                            action="transfer",
                        )
                    )
                elif event.action == "return":
                    owner = event.participant_ids[-1] if len(event.participant_ids) > 1 else event.participant_ids[0]
                    states.append(EntityState(frame=event_frame, position=self._hand_position(lanes[owner])))
                elif event.action == "place":
                    owner = None
                    states.append(EntityState(frame=event_frame, position=self._support_position(prop_id)))
                    attachments.append(
                        AttachmentEvent(
                            frame=frame(event.end),
                            subject_id=prop_id,
                            object_id="support_surface",
                            action="detach",
                        )
                    )
                    final_support_by_prop[prop_id] = "support_surface"
                if owner:
                    current_owner_by_event[f"{event.id}:{prop_id}"] = owner
            entities[prop_id] = EntityTrajectory(
                states=self._unique_states(states, frame_end),
                attachment_events=attachments,
            )

        return DirectorTrajectories(
            timebase=Timebase(fps=request.fps, frame_start=1, frame_end=frame_end),
            entities=entities,
            current_owner_by_event=current_owner_by_event,
            final_support_by_prop=final_support_by_prop,
        )

    def _lanes(self, actor_ids: list[str]) -> dict[str, Point3]:
        offset = (len(actor_ids) - 1) * self.lane_spacing / 2.0
        return {
            actor_id: (-3.0, index * self.lane_spacing - offset, 0.0)
            for index, actor_id in enumerate(actor_ids)
        }

    def _validate_lanes(self, lanes: dict[str, Point3]) -> None:
        positions = list(lanes.items())
        for index, (left_id, left) in enumerate(positions):
            for right_id, right in positions[index + 1 :]:
                distance = math.hypot(left[0] - right[0], left[1] - right[1])
                if distance < self.minimum_lane_distance:
                    raise ValueError(f"lane collision between {left_id} and {right_id}: {distance:.3f}")

    @staticmethod
    def _actor_position(actor_id: str, action: str, lanes: dict[str, Point3], event_id: str) -> Point3:
        start = lanes[actor_id]
        if action in {"handoff", "return"}:
            if "actor_a_actor_b" in event_id:
                return (-0.35 if actor_id == "actor_a" else 0.35, 0.0, 0.0)
            if "actor_b_actor_a" in event_id:
                return (0.35 if actor_id == "actor_a" else -0.35, 0.0, 0.0)
            return (0.0, start[1], 0.0)
        if action == "place":
            return (2.5, start[1], 0.0)
        if action == "pause":
            return (-0.75, start[1], 0.0)
        return (-1.0, start[1], 0.0)

    @classmethod
    def _hand_position(cls, actor_position: Point3) -> Point3:
        return tuple(
            coordinate + offset for coordinate, offset in zip(actor_position, cls._HAND_OFFSET)
        )  # type: ignore[return-value]

    @staticmethod
    def _prop_start(prop_id: str) -> Point3:
        return (-3.0, -1.0 if prop_id.startswith("red") else 1.0, 0.8)

    @staticmethod
    def _support_position(prop_id: str) -> Point3:
        return (2.5, -1.0 if prop_id.startswith("red") else 1.0, 0.8)

    @staticmethod
    def _primitive_for(action: str) -> str:
        return {
            "carry": "s_curve",
            "handoff": "arc",
            "return": "bezier",
            "place": "zigzag",
            "pause": "bezier",
        }.get(action, "bezier")

    @staticmethod
    def _unique_states(states: list[EntityState], frame_end: int) -> list[EntityState]:
        unique: dict[int, EntityState] = {}
        for state in states:
            unique[min(frame_end, state.frame)] = state
        return [unique[frame] for frame in sorted(unique)]
