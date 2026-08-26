"""Event scheduling for DirectorAgent.

This layer turns interpreted action directives into bounded event graphs and
interaction lifecycles. It does not assign spatial trajectories or camera shots.
"""

from __future__ import annotations

from pydantic import Field, model_validator

from .director_contracts import ContractModel, DirectorEvent, DirectorRequest, InteractionLifecycle
from .director_prompt import DirectorActionDirective, PromptInterpretation


class DirectorSchedule(ContractModel):
    events: list[DirectorEvent]
    interactions: list[InteractionLifecycle] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_schedule(self) -> "DirectorSchedule":
        event_ids = [event.id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("scheduled event IDs must be unique")
        event_set = set(event_ids)
        for event in self.events:
            unknown = set(event.depends_on) - event_set
            if unknown:
                raise ValueError(f"event {event.id} references unknown dependencies: {sorted(unknown)}")
        return self


class EventScheduler:
    def schedule(self, request: DirectorRequest, interpretation: PromptInterpretation) -> DirectorSchedule:
        directives = interpretation.directives
        if not directives:
            return DirectorSchedule(events=[], interactions=[])

        windows = self._windows(request.duration_s, directives)
        events: list[DirectorEvent] = []
        last_by_prop: dict[str, str] = {}
        last_by_actor: dict[str, str] = {}

        for index, directive in enumerate(directives):
            if directive.action == "pause":
                event_id = self._event_id(directive)
                dependencies = self._dependencies(directive, last_by_prop, last_by_actor)
                events.append(
                    DirectorEvent(
                        id=event_id,
                        action="pause",
                        participant_ids=[directive.actor_id] if directive.actor_id else [],
                        target_ids=[directive.prop_id] if directive.prop_id else [],
                        depends_on=dependencies,
                        start=windows[index][0],
                        end=windows[index][1],
                    )
                )
                self._remember(directive, event_id, last_by_prop, last_by_actor)
                continue

            event_id = self._event_id(directive)
            dependencies = self._dependencies(directive, last_by_prop, last_by_actor)
            if directive.concurrency_group:
                dependencies = [
                    dependency
                    for dependency in dependencies
                    if dependency not in {event.id for event in events if event.concurrency_group == directive.concurrency_group}
                ]
            elif directive.action in {"handoff", "return"} and events:
                previous_group = events[-1].concurrency_group
                if previous_group:
                    dependencies.extend(
                        event.id for event in events if event.concurrency_group == previous_group
                    )
                    dependencies = list(dict.fromkeys(dependencies))
            participants = [directive.actor_id] if directive.actor_id else []
            if directive.action == "handoff" and directive.receiver_id:
                participants.append(directive.receiver_id)
            events.append(
                DirectorEvent(
                    id=event_id,
                    action=directive.action,
                    participant_ids=participants,
                    target_ids=[directive.prop_id] if directive.prop_id else [],
                    depends_on=dependencies,
                    concurrency_group=directive.concurrency_group,
                    start=windows[index][0],
                    end=windows[index][1],
                )
            )
            self._remember(directive, event_id, last_by_prop, last_by_actor)

        return DirectorSchedule(events=events, interactions=self._lifecycles(directives, events))

    @staticmethod
    def _windows(duration_s: float, directives: list[DirectorActionDirective]) -> list[tuple[float, float]]:
        first_concurrency = directives[0].concurrency_group
        if first_concurrency:
            concurrent_count = len(
                [directive for directive in directives if directive.concurrency_group == first_concurrency]
            )
        else:
            concurrent_count = 0
        slot_count = len(directives) - max(0, concurrent_count - 1)
        slot = duration_s / max(1, slot_count)
        windows: list[tuple[float, float]] = []
        slot_index = 0
        for directive in directives:
            if directive.concurrency_group and directive.concurrency_group == first_concurrency:
                start = 0.0
                end = slot
            else:
                start = slot_index * slot
                end = min(duration_s, (slot_index + 1) * slot)
                slot_index += 1
            windows.append((round(start, 4), round(end, 4)))
        if first_concurrency:
            for index, directive in enumerate(directives):
                if not directive.concurrency_group:
                    previous = windows[index - 1][1] if index > 0 else slot
                    windows[index] = (round(previous, 4), round(min(duration_s, previous + slot), 4))
        return windows

    @staticmethod
    def _event_id(directive: DirectorActionDirective) -> str:
        if directive.action == "handoff":
            return f"handoff_{directive.actor_id}_{directive.receiver_id}_{directive.prop_id}"
        return "_".join(part for part in (directive.action, directive.actor_id, directive.prop_id) if part)

    @staticmethod
    def _dependencies(
        directive: DirectorActionDirective,
        last_by_prop: dict[str, str],
        last_by_actor: dict[str, str],
    ) -> list[str]:
        dependencies: list[str] = []
        if directive.prop_id and directive.prop_id in last_by_prop:
            dependencies.append(last_by_prop[directive.prop_id])
        if directive.actor_id and directive.actor_id in last_by_actor:
            dependencies.append(last_by_actor[directive.actor_id])
        return list(dict.fromkeys(dependencies))

    @staticmethod
    def _remember(
        directive: DirectorActionDirective,
        event_id: str,
        last_by_prop: dict[str, str],
        last_by_actor: dict[str, str],
    ) -> None:
        if directive.prop_id:
            last_by_prop[directive.prop_id] = event_id
        if directive.actor_id:
            last_by_actor[directive.actor_id] = event_id
        if directive.receiver_id and directive.action in {"handoff", "return"}:
            last_by_actor[directive.receiver_id] = event_id

    def _lifecycles(
        self,
        directives: list[DirectorActionDirective],
        events: list[DirectorEvent],
    ) -> list[InteractionLifecycle]:
        event_ids = {event.id for event in events}
        lifecycles: list[InteractionLifecycle] = []
        props = [directive.prop_id for directive in directives if directive.prop_id]
        for prop_id in dict.fromkeys(props):
            prop_directives = [directive for directive in directives if directive.prop_id == prop_id]
            carry = next((directive for directive in prop_directives if directive.action == "carry"), None)
            handoff = next((directive for directive in prop_directives if directive.action == "handoff"), None)
            detach = next(
                (directive for directive in prop_directives if directive.action in {"place", "return"}),
                handoff or carry,
            )
            if carry is None or detach is None:
                continue
            transfer_event_id = self._event_id(handoff) if handoff is not None else None
            final_owner = (
                handoff.receiver_id
                if handoff is not None
                else detach.receiver_id
                if detach.action == "return"
                else detach.actor_id
            )
            lifecycle = InteractionLifecycle(
                id=f"{prop_id}_lifecycle",
                prop_id=prop_id,
                giver_id=handoff.actor_id if handoff is not None else carry.actor_id,
                receiver_id=handoff.receiver_id if handoff is not None else detach.receiver_id,
                attach_event_id=self._event_id(carry),
                transfer_event_id=transfer_event_id,
                detach_event_id=self._event_id(detach),
                final_owner_id=final_owner,
                final_support_id="support_surface" if detach.action == "place" else None,
            )
            if (
                lifecycle.attach_event_id in event_ids
                and lifecycle.detach_event_id in event_ids
                and (lifecycle.transfer_event_id is None or lifecycle.transfer_event_id in event_ids)
            ):
                lifecycles.append(lifecycle)
        return lifecycles
