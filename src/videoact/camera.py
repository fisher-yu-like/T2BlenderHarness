"""Deterministic camera shot planning and event observability."""

from __future__ import annotations

from .contracts import CameraPlan, CameraShot, EntityTrajectory, SceneContract


class CameraPlanner:
    def plan(self, contract: SceneContract, entity_trajectories: dict[str, EntityTrajectory]) -> CameraPlan:
        frame_end = max(1, round(contract.duration_s * contract.fps))
        events = {event.id: event for event in contract.events}
        required = list(contract.must_show)
        target_id = next((entity.id for entity in contract.entities if entity.role == "target_object"), contract.entities[0].id)
        table_id = next((entity.id for entity in contract.entities if entity.role == "environment"), target_id)
        complex_camera = bool(set(contract.camera_constraints) & {"camera_follow", "camera_orbit", "camera_dolly", "occlusion_reveal"})
        if complex_camera:
            return self._plan_complex(contract, events, required, target_id, table_id, frame_end)

        shots: list[CameraShot] = []
        first_ids = [event_id for event_id in required if event_id in {"walk", "observe"}]
        if not first_ids and required:
            first_ids = required[:1]
        if first_ids:
            end_frame = self._event_end_frame(events[first_ids[-1]], contract.fps, frame_end)
            shots.append(
                CameraShot(
                    shot_id="shot-overview",
                    start_frame=1,
                    end_frame=end_frame,
                    target_ids=[table_id],
                    intent="establish scene and show approach",
                    lens_mm=35.0,
                    distance_range=(4.0, 8.0),
                    required_event_ids=first_ids,
                    trajectory_type="follow",
                )
            )
        remaining = [event_id for event_id in required if event_id not in first_ids]
        if remaining:
            start_frame = self._event_start_frame(events[remaining[0]], contract.fps)
            end_frame = self._event_end_frame(events[remaining[-1]], contract.fps, frame_end)
            shots.append(
                CameraShot(
                    shot_id="shot-action-closeup",
                    start_frame=start_frame,
                    end_frame=end_frame,
                    target_ids=[target_id],
                    intent="show target interaction clearly",
                    lens_mm=70.0 if "grasp_in_closeup" in contract.camera_constraints else 50.0,
                    distance_range=(1.0, 3.0),
                    required_event_ids=remaining,
                    trajectory_type="follow",
                )
            )
        self._assert_coverage(required, shots)
        return CameraPlan(shots=shots)

    def _plan_complex(self, contract, events, required, target_id, table_id, frame_end) -> CameraPlan:
        shots: list[CameraShot] = []
        overview_ids = [event_id for event_id in required if event_id in {"walk", "observe", "reveal"}]
        if overview_ids:
            shots.append(
                self._shot(
                    "shot-establish-follow",
                    overview_ids,
                    events,
                    contract,
                    table_id,
                    35.0,
                    (4.0, 8.0),
                    "establish the scene and follow the approach",
                    "follow" if "camera_follow" in contract.camera_constraints else "hold",
                    frame_end,
                )
            )

        orbit_ids = ["carry"] if "camera_orbit" in contract.camera_constraints and "carry" in required else []
        if orbit_ids:
            shots.append(
                self._shot(
                    "shot-carry-orbit",
                    orbit_ids,
                    events,
                    contract,
                    target_id,
                    50.0,
                    (2.0, 5.0),
                    "orbit the carried object while preserving handoff visibility",
                    "orbit",
                    frame_end,
                )
            )

        covered = set(overview_ids) | set(orbit_ids)
        action_ids = [event_id for event_id in required if event_id not in covered]
        if action_ids:
            closeup_suffix = "-closeup" if "grasp_in_closeup" in contract.camera_constraints else ""
            action_shot_id = (
                f"shot-action-dolly{closeup_suffix}"
                if "camera_dolly" in contract.camera_constraints
                else f"shot-action-follow{closeup_suffix}"
            )
            shots.append(
                self._shot(
                    action_shot_id,
                    action_ids,
                    events,
                    contract,
                    target_id,
                    70.0 if "grasp_in_closeup" in contract.camera_constraints else 50.0,
                    (1.0, 3.0),
                    "dolly into the action closeup and preserve event evidence",
                    "dolly" if "camera_dolly" in contract.camera_constraints else "follow",
                    frame_end,
                )
            )
        self._assert_coverage(required, shots)
        return CameraPlan(shots=sorted(shots, key=lambda shot: shot.start_frame))

    @staticmethod
    def _shot(shot_id, event_ids, events, contract, target_id, lens, distance, intent, trajectory_type, frame_end):
        return CameraShot(
            shot_id=shot_id,
            start_frame=CameraPlanner._event_start_frame(events[event_ids[0]], contract.fps),
            end_frame=CameraPlanner._event_end_frame(events[event_ids[-1]], contract.fps, frame_end),
            target_ids=[target_id],
            intent=intent,
            lens_mm=lens,
            distance_range=distance,
            required_event_ids=event_ids,
            trajectory_type=trajectory_type,
        )

    @staticmethod
    def _assert_coverage(required, shots):
        covered = {event_id for shot in shots for event_id in shot.required_event_ids}
        missing = set(required) - covered
        if missing:
            raise ValueError(f"camera plan does not cover required events: {sorted(missing)}")

    @staticmethod
    def _event_start_frame(event, fps: int) -> int:
        return max(1, round(event.start * fps) + 1)

    @staticmethod
    def _event_end_frame(event, fps: int, frame_end: int) -> int:
        return min(frame_end, max(1, round(event.end * fps) + 1))
