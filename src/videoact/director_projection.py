"""Compatibility projection from DirectorPlan to legacy Blender contracts."""

from __future__ import annotations

from .contracts import (
    CameraPlan,
    EntitySpec,
    EventObservability,
    EventSpec,
    SceneContract,
    TrajectoryPlan,
)
from .director_contracts import DirectorPlan
from .director_trajectory import DirectorTrajectories
from .scene_contract import SceneContractBuilder
from .trajectory import TrajectoryPlanner


class DirectorProjection:
    def project_scene_contract(self, director_plan: DirectorPlan) -> SceneContract:
        request = director_plan.request
        if director_plan.events:
            entity_kind = {
                "actor": "character",
                "prop": "prop",
                "support": "support",
                "occluder": "occluder",
                "camera": "camera",
            }
            entities = [
                EntitySpec(
                    id=entity.id,
                    kind=entity_kind.get(entity.kind, entity.kind),
                    role=entity.role,
                )
                for entity in director_plan.entities
            ]
            events = [
                EventSpec(
                    id=event.id,
                    start=event.start,
                    end=event.end,
                    description=event.action,
                    target_ids=[*event.participant_ids, *event.target_ids],
                )
                for event in director_plan.events
            ]
            if not events:
                events = [
                    EventSpec(
                        id="observe",
                        start=0.0,
                        end=request.duration_s,
                        description="scene remains observable",
                    )
                ]
            event_ids = [event.id for event in events]
            return SceneContract(
                scene_id=request.scene_id,
                duration_s=request.duration_s,
                fps=request.fps,
                entities=entities,
                events=events,
                must_show=event_ids,
                physics_constraints=["no_penetration", "interaction_state_consistency"],
                camera_constraints=["multi_target_visibility", "event_coverage"],
            )
        contract = SceneContractBuilder().build(
            request.prompt,
            duration_s=request.duration_s,
            fps=request.fps,
        )
        return contract.model_copy(update={"scene_id": request.scene_id})

    def project_trajectory_plan(
        self,
        scene_contract: SceneContract,
        *,
        director_trajectories: DirectorTrajectories | None = None,
        director_camera: CameraPlan | None = None,
        director_plan: DirectorPlan | None = None,
    ) -> TrajectoryPlan:
        if director_trajectories is not None and director_plan is not None and director_plan.events:
            camera = director_camera or CameraPlan()
            event_ids = [event.id for event in (director_plan.events if director_plan else [])]
            observability = [
                EventObservability(
                    event_id=event_id,
                    covered_by_shots=[
                        shot.shot_id
                        for shot in camera.shots
                        if event_id in shot.required_event_ids
                    ],
                    target_visible_predicate="all_targets_visible_during_event",
                )
                for event_id in event_ids
            ]
            return TrajectoryPlan(
                timebase=director_trajectories.timebase,
                entities=director_trajectories.entities,
                camera=camera,
                event_observability=observability,
                validation_intents=[
                    "director_event_order",
                    "multi_entity_collision_free_lanes",
                    "attachment_lifecycle",
                    "multi_target_camera_coverage",
                ],
            )
        return TrajectoryPlanner().plan(scene_contract)

    def project(
        self,
        director_plan: DirectorPlan,
        *,
        director_trajectories: DirectorTrajectories | None = None,
        director_camera: CameraPlan | None = None,
    ) -> tuple[SceneContract, TrajectoryPlan, CameraPlan]:
        contract = self.project_scene_contract(director_plan)
        trajectory = self.project_trajectory_plan(
            contract,
            director_trajectories=director_trajectories,
            director_camera=director_camera,
            director_plan=director_plan,
        )
        return contract, trajectory, trajectory.camera
