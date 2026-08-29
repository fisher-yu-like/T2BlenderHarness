"""Minimal articulated proxy rig helpers.

The helpers receive ``bpy`` explicitly so the contract can be unit-tested
without importing Blender.  A real job may embed or import these operations;
the module contains no prompt-specific entity names or scene layout.
"""

from __future__ import annotations

from typing import Any


MINIMAL_BONES = (
    "root", "hips", "spine", "chest", "neck", "head",
    "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
    "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
    "thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R", "foot.R",
)
PARENTS = {
    "hips": "root", "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "shoulder.L": "chest", "upper_arm.L": "shoulder.L", "forearm.L": "upper_arm.L", "hand.L": "forearm.L",
    "shoulder.R": "chest", "upper_arm.R": "shoulder.R", "forearm.R": "upper_arm.R", "hand.R": "forearm.R",
    "thigh.L": "hips", "shin.L": "thigh.L", "foot.L": "shin.L",
    "thigh.R": "hips", "shin.R": "thigh.R", "foot.R": "shin.R",
}


def rig_contract(entity_id: str) -> dict[str, Any]:
    if not entity_id:
        raise ValueError("entity_id is required")
    return {
        "entity_id": entity_id,
        "armature_name": f"{entity_id}__armature",
        "independent_per_character": True,
        "bones": list(MINIMAL_BONES),
        "ik_targets": ["hand.L", "hand.R"],
        "required_operations": ["ARMATURE", "vertex_groups", "IK", "walk_cycle"],
    }


def create_character_armature(bpy: Any, entity_id: str, location=(0.0, 0.0, 0.0)) -> Any:
    """Create one named armature with the minimal articulated hierarchy."""
    contract = rig_contract(entity_id)
    bpy.ops.object.armature_add( location=location)
    armature = bpy.context.object
    armature.name = contract["armature_name"]
    armature.data.name = f"{entity_id}__skeleton"
    armature.show_in_front = True
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    for bone_name in MINIMAL_BONES:
        bone = edit_bones.new(bone_name)
        bone.head = (0.0, 0.0, 0.0)
        bone.tail = (0.0, 0.0, 0.12)
        parent_name = PARENTS.get(bone_name)
        if parent_name:
            bone.parent = edit_bones[parent_name]
    bpy.ops.object.mode_set(mode="POSE")
    for hand_name in ("hand.L", "hand.R"):
        pose_bone = armature.pose.bones.get(hand_name)
        if pose_bone:
            pose_bone.rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature["entity_id"] = entity_id
    armature["rig_contract"] = "minimal_articulated_v1"
    return armature


def bind_mesh_to_armature(bpy: Any, mesh_object: Any, armature: Any) -> Any:
    """Add bone groups, conservative weights, and an ARMATURE modifier."""
    for bone_name in MINIMAL_BONES:
        group = mesh_object.vertex_groups.get(bone_name) or mesh_object.vertex_groups.new(name=bone_name)
        if mesh_object.data.vertices:
            group.add([vertex.index for vertex in mesh_object.data.vertices], 1.0, "REPLACE")
    modifier = mesh_object.modifiers.get("ArticulatedArmature") or mesh_object.modifiers.new("ArticulatedArmature", "ARMATURE")
    modifier.object = armature
    return modifier


def add_hand_ik_constraint(armature: Any, hand_name: str, target: Any) -> Any:
    if hand_name not in {"hand.L", "hand.R"}:
        raise ValueError("hand_name must be hand.L or hand.R")
    constraint = armature.pose.bones[hand_name].constraints.new("IK")
    constraint.name = f"{hand_name}__IK"
    constraint.target = target
    constraint.subtarget = hand_name
    constraint.chain_count = 2
    return constraint


def insert_walk_cycle(armature: Any, frame_start: int, frame_end: int, phase: float = 0.0) -> None:
    """Insert a small alternating leg cycle driven by the walk interval."""
    if frame_end <= frame_start:
        return
    for frame, sign in ((frame_start, 1.0), ((frame_start + frame_end) // 2, -1.0), (frame_end, 1.0)):
        for side, side_sign in (("L", sign), ("R", -sign)):
            bone = armature.pose.bones.get(f"thigh.{side}")
            if bone:
                bone.rotation_mode = "XYZ"
                bone.rotation_euler[1] = phase + 0.35 * side_sign
                bone.keyframe_insert(data_path="rotation_euler", index=1, frame=frame)

