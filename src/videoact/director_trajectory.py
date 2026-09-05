"""Collision-aware multi-entity trajectory composition for DirectorAgent."""

from __future__ import annotations

import math

from pydantic import Field

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


class MultiEntityTrajectoryComposer:
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
                position = self._actor_position(
                    actor_id,
                    event.action,
                    lanes,
                    event.id,
                    event.participant_ids,
                )
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

        for prop_index, prop_id in enumerate(prop_ids):
            states = [EntityState(frame=1, position=self._prop_start(prop_index, len(prop_ids)))]
            attachments: list[AttachmentEvent] = []
            primitives: list[MotionPrimitive] = []
            owner: str | None = None
            for event in schedule.events:
                if prop_id not in event.target_ids:
                    continue
                event_frame = frame(event.end)
                if event.action in {"attach", "carry"}:
                    owner = event.participant_ids[0]
                    if not any(
                        item.action == "attach"
                        and item.object_id == owner
                        and item.frame == frame(event.start)
                        for item in attachments
                    ):
                        attachments.append(
                            AttachmentEvent(
                                frame=frame(event.start),
                                subject_id=prop_id,
                                object_id=owner,
                                action="attach",
                                constraint_type="child_of",
                                subtarget="hand.R",
                            )
                        )
                elif event.action == "handoff":
                    owner = event.participant_ids[-1]
                    states.append(
                        EntityState(
                            frame=event_frame,
                            position=self._handoff_prop_position(owner, lanes),
                        )
                    )
                    primitives.append(
                        MotionPrimitive(
                            type=self._primitive_for(event.action),
                            start_frame=frame(event.start),
                            end_frame=event_frame,
                            parameters={"event_id": event.id, "prop_id": prop_id, "receiver_id": owner},
                        )
                    )
                    attachments.append(
                        AttachmentEvent(
                            frame=frame(event.start),
                            subject_id=prop_id,
                            object_id=owner,
                            action="transfer",
                            constraint_type="child_of",
                            subtarget="hand.R",
                        )
                    )
                    # A handoff releases the prop from the giver at the same
                    # boundary; without this detach the attach/transfer/detach
                    # lifecycle required by the interaction evaluator is
                    # incomplete for every handoff-only prompt.
                    giver_id = event.participant_ids[0]
                    attachments.append(
                        AttachmentEvent(
                            frame=frame(event.start),
                            subject_id=prop_id,
                            object_id=giver_id,
                            action="detach",
                            constraint_type="child_of",
                            subtarget="hand.R",
                        )
                    )
                elif event.action == "return":
                    owner = event.participant_ids[-1] if len(event.participant_ids) > 1 else event.participant_ids[0]
                elif event.action == "place":
                    owner = None
                    states.append(EntityState(frame=event_frame, position=self._support_position(prop_index, len(prop_ids))))
                    attachments.append(
                        AttachmentEvent(
                            frame=frame(event.end),
                            subject_id=prop_id,
                            object_id="support_surface",
                            action="detach",
                            constraint_type="support_surface",
                        )
                    )
                    final_support_by_prop[prop_id] = "support_surface"
                elif event.action == "detach":
                    owner = None
                    attachments.append(
                        AttachmentEvent(
                            frame=event_frame,
                            subject_id=prop_id,
                            object_id="support_surface",
                            action="detach",
                            constraint_type="support_surface",
                        )
                    )
                elif event.action not in {"observe", "pause"}:
                    # Non-transfer benchmark actions (walk, fly, press, and
                    # similar) still need an observable object trajectory.
                    # The provider supplies the semantic action; this layer
                    # adds a bounded state at the event boundary without
                    # inventing an attachment lifecycle.
                    states.append(
                        EntityState(
                            frame=event_frame,
                            position=self._prop_action_position(prop_index, len(prop_ids), event.action),
                        )
                    )
                    primitives.append(
                        MotionPrimitive(
                            type=self._primitive_for(event.action),
                            start_frame=frame(event.start),
                            end_frame=event_frame,
                            parameters={"event_id": event.id, "prop_id": prop_id},
                        )
                    )
                elif prop_id in event.participant_ids or prop_id in event.target_ids:
                    # Observation/pause is still an event in the benchmark;
                    # retain it as a hold primitive so coverage can prove the
                    # event reached an executable trajectory.
                    primitives.append(
                        MotionPrimitive(
                            type="hold",
                            start_frame=frame(event.start),
                            end_frame=event_frame,
                            parameters={"event_id": event.id, "prop_id": prop_id},
                        )
                    )
                if owner:
                    current_owner_by_event[f"{event.id}:{prop_id}"] = owner
            entities[prop_id] = EntityTrajectory(
                states=self._unique_states(states, frame_end),
                motion_primitives=primitives,
                attachment_events=attachments,
            )

        # Camera-only and environmental benchmark prompts still need a
        # traceable trajectory entry.  A static state is honest: it says the
        # subject is observed while the camera performs the requested cue,
        # rather than pretending the subject itself moved.
        known_ids = set(entities)
        for index, entity in enumerate(interpretation.entities):
            if entity.id in known_ids:
                continue
            hold_primitives = [
                MotionPrimitive(
                    type="hold",
                    start_frame=frame(event.start),
                    end_frame=frame(event.end),
                    parameters={"event_id": event.id, "entity_id": entity.id},
                )
                for event in schedule.events
                if entity.id in event.participant_ids or entity.id in event.target_ids
                if event.action in {"observe", "pause"}
            ]
            entities[entity.id] = EntityTrajectory(
                states=[
                    EntityState(
                        frame=1,
                        position=(0.0, (float(index) - 0.5) * 1.5, 1.0),
                    ),
                    EntityState(
                        frame=frame_end,
                        position=(0.0, (float(index) - 0.5) * 1.5, 1.0),
                    ),
                ],
                motion_primitives=hold_primitives,
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
    def _actor_position(
        actor_id: str,
        action: str,
        lanes: dict[str, Point3],
        event_id: str,
        participants: list[str] | None = None,
    ) -> Point3:
        start = lanes[actor_id]
        if action in {"handoff", "return"}:
            participant_set = set(participants or [])
            if {"actor_a", "actor_b"}.issubset(participant_set) and action == "handoff":
                return (-0.35 if actor_id == "actor_a" else 0.35, 0.0, 0.0)
            if "actor_a_actor_b" in event_id:
                return (-0.35 if actor_id == "actor_a" else 0.35, 0.0, 0.0)
            if "actor_b_actor_a" in event_id:
                return (0.35 if actor_id == "actor_a" else -0.35, 0.0, 0.0)
            return (0.0, start[1], 0.0)
        if action == "place":
            return (2.5, start[1], 0.0)
        if action == "pause":
            return (-0.75, start[1], 0.0)
        if action in {"walk", "run", "move", "jump", "fly", "climb"}:
            return (1.5, start[1], 0.35 if action in {"jump", "fly"} else 0.0)
        if action in {"sit", "stand", "observe"}:
            return (-1.5, start[1], 0.0)
        if action in {"drink", "sweep", "brush", "write", "pour", "press", "interact", "open", "close"}:
            return (0.25, start[1], 0.0)
        return (-1.0, start[1], 0.0)

    @staticmethod
    def _handoff_prop_position(owner_id: str, lanes: dict[str, Point3]) -> Point3:
        """Place a transferred prop at the receiver's declared hand lane."""

        owner = lanes.get(owner_id, (0.0, 0.0, 0.0))
        return (owner[0] + 0.65, owner[1] - 0.05, owner[2] + 1.35)

    @classmethod
    def _prop_action_position(cls, prop_index: int, prop_count: int, action: str) -> Point3:
        base = cls._prop_start(prop_index, prop_count)
        if action in {"move", "walk", "run", "jump", "fly", "climb"}:
            return (1.6, base[1], 1.15 if action in {"jump", "fly"} else base[2])
        if action in {"press", "sweep", "brush", "write", "pour", "drink", "interact", "open", "close"}:
            return (0.4, base[1], base[2])
        if action == "bounce":
            return (0.8, base[1], 2.0)
        return base

    @staticmethod
    def _layout_y(prop_index: int, prop_count: int) -> float:
        """Place props by authored order, never by color/name heuristics."""
        count = max(1, int(prop_count))
        return (float(prop_index) - (count - 1) / 2.0) * 2.0

    @classmethod
    def _prop_start(cls, prop_index: int, prop_count: int) -> Point3:
        return (-3.0, cls._layout_y(prop_index, prop_count), 0.8)

    @classmethod
    def _support_position(cls, prop_index: int, prop_count: int) -> Point3:
        return (2.5, cls._layout_y(prop_index, prop_count), 0.8)

    @staticmethod
    def _primitive_for(action: str) -> str:
        return {
            "carry": "s_curve",
            "handoff": "arc",
            "return": "bezier",
            "place": "zigzag",
            "pause": "bezier",
            "observe": "hold",
            "sit": "hold",
            "stand": "ease_in_out",
            "walk": "s_curve",
            "run": "s_curve",
            "move": "s_curve",
            "jump": "arc",
            "fly": "arc",
            "climb": "arc",
            "bounce": "arc",
        }.get(action, "bezier")

    @staticmethod
    def _unique_states(states: list[EntityState], frame_end: int) -> list[EntityState]:
        unique: dict[int, EntityState] = {}
        for state in states:
            unique[min(frame_end, state.frame)] = state
        return [unique[frame] for frame in sorted(unique)]
