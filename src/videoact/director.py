"""DirectorAgent facade and compatibility planning result."""

from __future__ import annotations

from typing_extensions import Literal

from pydantic import Field

from .contracts import CameraPlan, SceneContract, TrajectoryPlan
from .director_camera import MultiTargetCameraChoreographer
from .director_contracts import (
    ContractModel,
    DirectorEntity,
    DirectorEvent,
    DirectorPlan,
    DirectorRequest,
)
from .director_projection import DirectorProjection
from .director_prompt import DeterministicPromptInterpreter
from .director_schedule import EventScheduler
from .director_trajectory import MultiEntityTrajectoryComposer
from .director_trajectory import DirectorTrajectories
from .scene_contract import SceneContractBuilder
from .trajectory import TrajectoryPlanner


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
            trajectory_summary=trajectories.model_dump(mode="json"),
            camera_plan=camera,
            coverage_obligations=list(
                dict.fromkeys(
                    [event.id for event in schedule.events]
                    + [event_id for shot in camera.shots for event_id in shot.required_event_ids]
                )
            ),
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
        mode: Literal["deterministic_baseline", "dynamic"] | None = None,
    ) -> None:
        selected_interpreter = interpreter or DeterministicPromptInterpreter()
        inferred_mode = (
            "deterministic_baseline"
            if isinstance(selected_interpreter, DeterministicPromptInterpreter)
            else "dynamic"
        )
        self.mode = mode or inferred_mode
        if self.mode == "dynamic" and isinstance(selected_interpreter, DeterministicPromptInterpreter):
            raise ValueError("dynamic mode requires a provider-assisted interpreter")
        self.interpreter = selected_interpreter
        self.scheduler = scheduler or EventScheduler()
        self.trajectory = trajectory or MultiEntityTrajectoryComposer()
        self.camera_choreographer = camera_choreographer or MultiTargetCameraChoreographer()
        self.critic = critic or DirectorCritic()
        self.projector = projector or DirectorProjection()
        self.provider = provider
        self.policy = policy

    @classmethod
    def from_provider(
        cls,
        provider,
        *,
        provider_name: str = "codex-local",
        policy: str = "director-v2",
    ) -> "DirectorAgent":
        """Build a production DirectorAgent around a structured provider."""

        from .director_prompt_llm import StructuredPromptInterpreter

        return cls(
            interpreter=StructuredPromptInterpreter(provider),
            provider=provider_name,
            policy=policy,
            mode="dynamic",
        )

    def plan(
        self,
        prompt: str,
        *,
        scene_id: str,
        duration_s: float = 10.0,
        fps: int = 24,
        obligations: dict[str, list[str]] | None = None,
    ) -> DirectorPlanningResult:
        request = DirectorRequest(
            prompt=prompt,
            scene_id=scene_id,
            duration_s=duration_s,
            fps=fps,
            provider=self.provider,
            policy=self.policy,
            obligations=obligations or {},
        )
        interpretation = self.interpreter.interpret(request)
        self._reject_unresolved_hard_uncertainty(interpretation)
        schedule = self.scheduler.schedule(request, interpretation)
        if not schedule.events:
            raise ValueError("empty event graph after DirectorAgent interpretation")
        trajectories = self.trajectory.compose(request, interpretation, schedule)
        camera = self.camera_choreographer.compose(request, interpretation, schedule, trajectories)
        self._validate_obligations(request, interpretation, schedule, trajectories, camera)
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

    def plan_explicit_baseline(
        self,
        prompt: str,
        *,
        scene_id: str,
        duration_s: float = 10.0,
        fps: int = 24,
    ) -> DirectorPlanningResult:
        """Project the historical deterministic baseline for comparison only.

        This method is deliberately separate from :meth:`plan`.  It is used by
        the old fake benchmark and the explicit ``template_baseline`` arm, so
        those experiments can preserve their historical behavior without
        making deterministic parsing an implicit production fallback.  It does
        not call an agent provider and it must never be selected by agent mode.
        """

        if self.mode != "deterministic_baseline":
            raise ValueError("explicit baseline projection requires deterministic_baseline mode")
        contract = SceneContractBuilder().build(
            prompt,
            duration_s=duration_s,
            fps=fps,
        ).model_copy(update={"scene_id": scene_id})
        trajectory_plan = TrajectoryPlanner().plan(contract)
        request = DirectorRequest(
            prompt=prompt,
            scene_id=scene_id,
            duration_s=duration_s,
            fps=fps,
            provider="deterministic-baseline",
            policy="baseline-v1",
        )
        entities = [
            DirectorEntity(
                id=entity.id,
                kind={
                    "character": "actor",
                    "occluder": "environment",
                }.get(entity.kind, entity.kind),
                role=entity.role,
                label=entity.id,
            )
            for entity in contract.entities
        ]
        actor_ids = [entity.id for entity in entities if entity.kind == "actor"]
        fallback_participant = actor_ids[0] if actor_ids else entities[0].id
        events = [
            DirectorEvent(
                id=event.id,
                action=event.id,
                participant_ids=[fallback_participant],
                target_ids=list(event.target_ids),
                start=event.start,
                end=event.end,
            )
            for event in contract.events
        ]
        director_plan = DirectorPlan(
            id=f"baseline-plan-{scene_id}",
            request=request,
            entities=entities,
            events=events,
            trajectory_summary={"entities": trajectory_plan.entities},
            camera_plan=trajectory_plan.camera,
            coverage_obligations=[event.id for event in events],
            provider_fingerprint="provider:deterministic-baseline",
            policy_fingerprint="policy:baseline-v1",
        )
        director_trajectories = DirectorTrajectories(
            timebase=trajectory_plan.timebase,
            entities=trajectory_plan.entities,
        )
        return DirectorPlanningResult(
            director_plan=director_plan,
            director_trajectories=director_trajectories,
            director_camera=trajectory_plan.camera,
            scene_contract=contract,
            trajectory_plan=trajectory_plan,
            camera_plan=trajectory_plan.camera,
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

    @staticmethod
    def _validate_obligations(request, interpretation, schedule, trajectories, camera) -> None:
        """Reject a provider plan that drops stable case obligations."""

        obligations = request.obligations or {}
        entity_ids = {entity.id for entity in interpretation.entities}
        event_ids = {event.id for event in schedule.events}
        trajectory_ids = set(trajectories.entities)
        camera_event_ids = {
            event_id
            for shot in camera.shots
            for event_id in shot.required_event_ids
        }
        checks = {
            "entities": (set(obligations.get("required_entity_ids", [])), entity_ids),
            "events": (set(obligations.get("required_event_ids", [])), event_ids),
            "trajectory_entities": (set(obligations.get("required_entity_ids", [])), trajectory_ids),
            "camera_events": (set(obligations.get("required_camera_event_ids", [])), camera_event_ids),
        }
        missing = {
            name: sorted(required - covered)
            for name, (required, covered) in checks.items()
            if required - covered
        }
        if missing:
            raise ValueError(f"DirectorPlan coverage obligations unresolved: {missing}")
