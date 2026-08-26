"""Multi-target camera choreography for DirectorAgent."""

from __future__ import annotations

from .contracts import CameraPlan, CameraShot
from .director_contracts import DirectorRequest
from .director_prompt import PromptInterpretation
from .director_schedule import DirectorSchedule
from .director_trajectory import DirectorTrajectories


class MultiTargetCameraChoreographer:
    def compose(
        self,
        request: DirectorRequest,
        interpretation: PromptInterpretation,
        schedule: DirectorSchedule,
        trajectories: DirectorTrajectories,
    ) -> CameraPlan:
        del interpretation, trajectories
        frame_end = max(1, round(request.duration_s * request.fps))
        frame = lambda seconds: max(1, min(frame_end, round(seconds * request.fps) + 1))
        shots: list[CameraShot] = []
        consumed: set[str] = set()

        for event in schedule.events:
            if event.id in consumed:
                continue
            group_events = (
                [candidate for candidate in schedule.events if candidate.concurrency_group == event.concurrency_group]
                if event.concurrency_group
                else [event]
            )
            for candidate in group_events:
                consumed.add(candidate.id)
            targets = self._targets(group_events)
            action_set = {candidate.action for candidate in group_events}
            if "handoff" in action_set:
                intent = "handoff two-shot preserves giver, receiver, and prop contact"
                trajectory_type = "orbit"
                max_occlusion = 0.25
            elif len(group_events) > 1:
                intent = "concurrent action coverage keeps independent lanes visible"
                trajectory_type = "follow"
                max_occlusion = 0.35
            elif "place" in action_set:
                intent = "placement reveal follows prop to final support"
                trajectory_type = "dolly"
                max_occlusion = 0.3
            else:
                intent = "event coverage with visible actors and props"
                trajectory_type = "follow"
                max_occlusion = 0.4

            shots.append(
                CameraShot(
                    shot_id=f"shot_{len(shots) + 1:02d}_{group_events[0].action}",
                    start_frame=frame(min(candidate.start for candidate in group_events)),
                    end_frame=frame(max(candidate.end for candidate in group_events)),
                    target_ids=targets,
                    intent=intent,
                    lens_mm=35.0 if len(targets) > 2 else 50.0,
                    distance_range=(4.0, 8.0),
                    required_event_ids=[candidate.id for candidate in group_events],
                    trajectory_type=trajectory_type,
                    visibility_predicates={target_id: "visible" for target_id in targets},
                    max_occlusion=max_occlusion,
                    continuity_group="axis_a",
                    innovation_intent_evidence_id=f"ev_camera_{len(shots) + 1:02d}",
                )
            )
        return CameraPlan(shots=shots)

    @staticmethod
    def _targets(events) -> list[str]:
        targets: list[str] = []
        for event in events:
            targets.extend(event.participant_ids)
            targets.extend(event.target_ids)
        return list(dict.fromkeys(targets))
