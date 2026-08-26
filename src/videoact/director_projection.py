"""Compatibility projection from DirectorPlan to legacy Blender contracts."""

from __future__ import annotations

from .contracts import CameraPlan, SceneContract, TrajectoryPlan
from .director_contracts import DirectorPlan
from .scene_contract import SceneContractBuilder
from .trajectory import TrajectoryPlanner


class DirectorProjection:
    def project_scene_contract(self, director_plan: DirectorPlan) -> SceneContract:
        request = director_plan.request
        contract = SceneContractBuilder().build(
            request.prompt,
            duration_s=request.duration_s,
            fps=request.fps,
        )
        return contract.model_copy(update={"scene_id": request.scene_id})

    def project_trajectory_plan(self, scene_contract: SceneContract) -> TrajectoryPlan:
        return TrajectoryPlanner().plan(scene_contract)

    def project(self, director_plan: DirectorPlan) -> tuple[SceneContract, TrajectoryPlan, CameraPlan]:
        contract = self.project_scene_contract(director_plan)
        trajectory = self.project_trajectory_plan(contract)
        return contract, trajectory, trajectory.camera
