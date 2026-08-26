"""DirectorAgent facade and compatibility planning result."""

from __future__ import annotations

from pydantic import Field

from .contracts import CameraPlan, SceneContract, TrajectoryPlan
from .director_camera import MultiTargetCameraChoreographer
from .director_contracts import (
    ContractModel,
    DirectorPlan,
    DirectorRequest,
)
from .director_projection import DirectorProjection
from .director_prompt import DeterministicPromptInterpreter
from .director_schedule import EventScheduler
from .director_trajectory import MultiEntityTrajectoryComposer
from .director_trajectory import DirectorTrajectories


class DirectorPlanningResult(ContractModel):
    director_plan: DirectorPlan
    director_trajectories: DirectorTrajectories
    director_camera: CameraPlan
    scene_contract: SceneContract
    trajectory_plan: TrajectoryPlan
    camera_plan: CameraPlan
    director_plan_hash: str = Field(min_length=64, max_length=64)


class DirectorCritic:
    def validate_and_repair(
        self,
        *,
        request: DirectorRequest,
        interpretation,
        schedule,
        trajectories,
        camera,
    ) -> DirectorPlan:
        del trajectories, camera
        hard_uncertainties = [
            uncertainty
            for uncertainty in interpretation.uncertainties
            if uncertainty.severity == "hard" and not uncertainty.resolved
        ]
        if hard_uncertainties:
            ids = ", ".join(uncertainty.id for uncertainty in hard_uncertainties)
            raise ValueError(f"unresolved hard uncertainty before Blender compilation: {ids}")
        return DirectorPlan(
            id=f"director-plan-{request.scene_id}",
            request=request,
            entities=interpretation.entities,
            events=schedule.events,
            interactions=schedule.interactions,
            assumptions=[],
            uncertainties=interpretation.uncertainties,
            evidence=interpretation.evidence,
            provider_fingerprint=f"provider:{request.provider}",
            policy_fingerprint=f"policy:{request.policy}",
        )


class DirectorAgent:
    def __init__(
        self,
        *,
        interpreter=None,
        scheduler=None,
        trajectory=None,
        camera_choreographer=None,
        critic=None,
        projector=None,
        provider: str = "deterministic",
        policy: str = "director-v1",
    ) -> None:
        self.interpreter = interpreter or DeterministicPromptInterpreter()
        self.scheduler = scheduler or EventScheduler()
        self.trajectory = trajectory or MultiEntityTrajectoryComposer()
        self.camera_choreographer = camera_choreographer or MultiTargetCameraChoreographer()
        self.critic = critic or DirectorCritic()
        self.projector = projector or DirectorProjection()
        self.provider = provider
        self.policy = policy

    def plan(
        self,
        prompt: str,
        *,
        scene_id: str,
        duration_s: float = 10.0,
        fps: int = 24,
    ) -> DirectorPlanningResult:
        request = DirectorRequest(
            prompt=prompt,
            scene_id=scene_id,
            duration_s=duration_s,
            fps=fps,
            provider=self.provider,
            policy=self.policy,
        )
        interpretation = self.interpreter.interpret(request)
        self._reject_unresolved_hard_uncertainty(interpretation)
        schedule = self.scheduler.schedule(request, interpretation)
        trajectories = self.trajectory.compose(request, interpretation, schedule)
        camera = self.camera_choreographer.compose(request, interpretation, schedule, trajectories)
        director_plan = self.critic.validate_and_repair(
            request=request,
            interpretation=interpretation,
            schedule=schedule,
            trajectories=trajectories,
            camera=camera,
        )
        scene_contract, trajectory_plan, camera_plan = self.projector.project(
            director_plan,
            director_trajectories=trajectories,
            director_camera=camera,
        )
        return DirectorPlanningResult(
            director_plan=director_plan,
            director_trajectories=trajectories,
            director_camera=camera,
            scene_contract=scene_contract,
            trajectory_plan=trajectory_plan,
            camera_plan=camera_plan,
            director_plan_hash=director_plan.content_hash(),
        )

    @staticmethod
    def _reject_unresolved_hard_uncertainty(interpretation) -> None:
        hard_uncertainties = [
            uncertainty
            for uncertainty in interpretation.uncertainties
            if uncertainty.severity == "hard" and not uncertainty.resolved
        ]
        if hard_uncertainties:
            ids = ", ".join(uncertainty.id for uncertainty in hard_uncertainties)
            raise ValueError(f"unresolved hard uncertainty before Blender compilation: {ids}")
