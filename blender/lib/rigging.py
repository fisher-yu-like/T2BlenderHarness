"""Blender-free armature and skinning specifications."""

from __future__ import annotations

from dataclasses import dataclass
from math import dist

from . import register_primitive
from .constraints import ConstraintSpec


@dataclass(frozen=True)
class BoneSpec:
    name: str
    position: tuple[float, float, float]


@dataclass(frozen=True)
class ArmatureSpec:
    name: str
    location: tuple[float, float, float]
    bones: dict[str, BoneSpec]
    parent_map: dict[str, str]


@register_primitive(category="rigging", tags=["character", "humanoid"], cost_estimate="medium", example_usage="minimal_humanoid_armature('Alice', (0, 0, 0))")
def minimal_humanoid_armature(name: str, location: tuple[float, float, float]) -> ArmatureSpec:
    """Return a minimal humanoid rig sufficient for walking, reach, and handoff."""

    if not name:
        raise ValueError("armature name is required")
    lx, ly, lz = (float(item) for item in location)
    local_positions = {
        "root": (0.0, 0.0, 0.0),
        "hips": (0.0, 0.0, 1.0),
        "spine": (0.0, 0.0, 1.55),
        "chest": (0.0, 0.0, 2.15),
        "neck": (0.0, 0.0, 2.75),
        "head": (0.0, 0.0, 3.05),
        "shoulder.L": (-0.48, 0.0, 2.25),
        "upper_arm.L": (-0.62, 0.0, 2.05),
        "forearm.L": (-0.82, -0.02, 1.72),
        "hand.L": (-1.04, -0.04, 1.40),
        "shoulder.R": (0.48, 0.0, 2.25),
        "upper_arm.R": (0.62, 0.0, 2.05),
        "forearm.R": (0.82, -0.02, 1.72),
        "hand.R": (1.04, -0.04, 1.40),
        "thigh.L": (-0.34, 0.0, 1.0),
        "shin.L": (-0.45, 0.0, 0.55),
        "foot.L": (-0.45, -0.02, 0.10),
        "thigh.R": (0.34, 0.0, 1.0),
        "shin.R": (0.45, 0.0, 0.55),
        "foot.R": (0.45, -0.02, 0.10),
    }
    bones = {
        bone: BoneSpec(bone, (x + lx, y + ly, z + lz))
        for bone, (x, y, z) in local_positions.items()
    }
    parent_map = {
        "hips": "root",
        "spine": "hips",
        "chest": "spine",
        "neck": "chest",
        "head": "neck",
        "shoulder.L": "chest",
        "upper_arm.L": "shoulder.L",
        "forearm.L": "upper_arm.L",
        "hand.L": "forearm.L",
        "shoulder.R": "chest",
        "upper_arm.R": "shoulder.R",
        "forearm.R": "upper_arm.R",
        "hand.R": "forearm.R",
        "thigh.L": "hips",
        "shin.L": "thigh.L",
        "foot.L": "shin.L",
        "thigh.R": "hips",
        "shin.R": "thigh.R",
        "foot.R": "shin.R",
    }
    return ArmatureSpec(name=name, location=(lx, ly, lz), bones=bones, parent_map=parent_map)


@register_primitive(category="rigging", tags=["skinning", "weights"], cost_estimate="medium", example_usage="bind_mesh_to_armature(vertices, rig)")
def bind_mesh_to_armature(
    mesh_verts: list[tuple[float, float, float]],
    armature: ArmatureSpec,
) -> dict[str, list[tuple[int, float]]]:
    """Assign each vertex to its nearest bone with a normalized weight of 1."""

    if not armature.bones:
        raise ValueError("armature must contain bones")
    weights: dict[str, list[tuple[int, float]]] = {name: [] for name in armature.bones}
    for index, vertex in enumerate(mesh_verts):
        nearest = min(armature.bones.values(), key=lambda bone: dist(tuple(float(item) for item in vertex), bone.position))
        weights[nearest.name].append((index, 1.0))
    return {name: entries for name, entries in weights.items() if entries}


@register_primitive(category="rigging", tags=["ik", "reach"], cost_estimate="low", example_usage="add_ik_constraint('hand.R', 'Alice_hand_target')")
def add_ik_constraint(bone: str, target: str, chain_length: int = 2) -> ConstraintSpec:
    """Return an IK constraint specification for a hand or foot chain."""

    if not bone or not target:
        raise ValueError("bone and target are required")
    if int(chain_length) <= 0:
        raise ValueError("chain_length must be positive")
    return ConstraintSpec(
        type="IK",
        object_id=bone,
        target=target,
        influence_keyframes=[],
        chain_length=int(chain_length),
    )


__all__ = ["ArmatureSpec", "BoneSpec", "add_ik_constraint", "bind_mesh_to_armature", "minimal_humanoid_armature"]
