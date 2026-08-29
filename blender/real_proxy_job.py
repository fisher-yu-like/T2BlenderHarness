"""Generate a self-contained Blender Python job for a real white proxy render."""

from __future__ import annotations

import json
from pathlib import Path

from videoact.contracts import CameraPlan, RunManifest, TrajectoryPlan
from videoact.director_contracts import DirectorPlan
from videoact.director_trajectory import DirectorTrajectories


def choose_render_engine(preferred: str, available: list[str] | tuple[str, ...]) -> str:
    """Choose a preferred engine with compatibility fallbacks for older Blender builds."""
    available_set = set(available)
    for candidate in (preferred, "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
        if candidate in available_set:
            return candidate
    raise ValueError(f"no supported render engine in {sorted(available_set)}")


def compile_real_proxy_job(
    plan: TrajectoryPlan,
    manifest: RunManifest,
    output_dir: str | Path,
    *,
    sample_frames: tuple[int, ...] = (1, 12, 24),
    proxy_spec: dict | None = None,
    director_plan: DirectorPlan | None = None,
    director_trajectories: DirectorTrajectories | None = None,
    director_camera: CameraPlan | None = None,
) -> str:
    output = Path(output_dir).resolve()
    plan_json = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    manifest_json = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    proxy_json = json.dumps(proxy_spec or {}, sort_keys=True)
    director_plan_json = json.dumps(
        director_plan.model_dump(mode="json") if director_plan is not None else {},
        sort_keys=True,
    )
    director_trajectories_json = json.dumps(
        director_trajectories.model_dump(mode="json") if director_trajectories is not None else {},
        sort_keys=True,
    )
    director_camera_json = json.dumps(
        director_camera.model_dump(mode="json") if director_camera is not None else {},
        sort_keys=True,
    )
    samples_json = json.dumps(list(sample_frames))
    # The authored proxy scene and DirectorPlan are authoritative for geometry
    # semantics. Legacy TrajectoryPlan entities are only a compatibility
    # fallback; no stable ID is interpreted as a character by name.
    proxy_entities = []
    kind_aliases = {
        "actor": "character",
        "character": "character",
        "prop": "prop",
        "support": "support",
        "occluder": "occluder",
        "camera": "camera",
    }
    authored_by_id = {
        str(entity.get("id")): {
            "id": str(entity.get("id")),
            "kind": kind_aliases.get(str(entity.get("kind", "prop")), str(entity.get("kind", "prop"))),
            "role": str(entity.get("role", "target_object")),
            **{
                key: value
                for key, value in entity.items()
                if key not in {"id", "kind", "role"}
            },
        }
        for entity in (proxy_spec or {}).get("entities", [])
        if entity.get("id")
    }
    for entity_id, entity in authored_by_id.items():
        proxy_entities.append(entity)
    director_kind = kind_aliases
    if director_plan is not None:
        for entity in director_plan.entities:
            if entity.id not in authored_by_id:
                proxy_entities.append(
                    {
                        "id": entity.id,
                        "kind": director_kind.get(entity.kind, entity.kind),
                        "role": entity.role,
                        "label": entity.label,
                    }
                )
    for entity_id in plan.entities:
        if entity_id not in authored_by_id and not any(item["id"] == entity_id for item in proxy_entities):
            proxy_entities.append({
                "id": entity_id,
                "kind": "support" if entity_id in {"table", "support", "surface", "platform", "drop_zone", "support_surface"} else "prop",
                "role": "environment" if entity_id in {"table", "support", "surface", "platform", "drop_zone", "support_surface"} else "target_object",
            })
    return f'''"""Generated real Blender MCP proxy job. Source is hash-bound by run_manifest.json."""
from pathlib import Path
import hashlib
import json
import math

import bpy
from mathutils import Vector

OUTPUT_DIR = Path({str(output)!r})
FRAMES_DIR = OUTPUT_DIR / "frames"
PLAN = json.loads({plan_json!r})
INITIAL_MANIFEST = json.loads({manifest_json!r})
PROXY_SPEC = json.loads({proxy_json!r})
DIRECTOR_PLAN = json.loads({director_plan_json!r})
DIRECTOR_TRAJECTORIES = json.loads({director_trajectories_json!r})
DIRECTOR_CAMERA = json.loads({director_camera_json!r})
SAMPLE_FRAMES = {samples_json}


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def white_material():
    material = bpy.data.materials.get("ProxyWhiteMaterial") or bpy.data.materials.new("ProxyWhiteMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.72, 0.74, 0.78, 1.0)
        principled.inputs["Roughness"].default_value = 0.62
    material.diffuse_color = (0.72, 0.74, 0.78, 1.0)
    return material


def principled_material(name, color, roughness=0.6, metallic=0.0):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = tuple(color) + (1.0,)
        principled.inputs["Roughness"].default_value = float(roughness)
        principled.inputs["Metallic"].default_value = float(metallic)
    material.diffuse_color = tuple(color) + (1.0,)
    return material


def material_for(entity_spec):
    kind = entity_spec.get("kind", "prop")
    label = str(entity_spec.get("label", entity_spec.get("id", "prop"))).lower()
    if kind == "character":
        return principled_material("CharacterSkinMaterial", (0.46, 0.22, 0.12), roughness=0.62)
    if kind == "support":
        return principled_material("SupportWoodMaterial", (0.24, 0.12, 0.05), roughness=0.8)
    if kind == "occluder":
        return principled_material("OccluderMaterial", (0.04, 0.05, 0.07), roughness=0.9)
    color = (0.55, 0.12, 0.08)
    if "blue" in label:
        color = (0.06, 0.20, 0.62)
    elif "green" in label:
        color = (0.10, 0.42, 0.16)
    elif "yellow" in label:
        color = (0.75, 0.52, 0.06)
    return principled_material("PropMaterial__" + str(entity_spec.get("id", "prop")), color, roughness=0.42, metallic=0.05)


def ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=30.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    plane.name = "ground_plane"
    plane["entity_kind"] = "environment"
    plane["geometry_style"] = "ground_contact_surface_v1"
    plane.data.materials.append(principled_material("GroundMaterial", (0.08, 0.09, 0.11), roughness=0.92))
    return plane


MINIMAL_BONES = (
    "root", "hips", "spine", "chest", "neck", "head",
    "shoulder.L", "upper_arm.L", "forearm.L", "hand.L",
    "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
    "thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R", "foot.R",
)
PARENTS = {{
    "hips": "root", "spine": "hips", "chest": "spine", "neck": "chest", "head": "neck",
    "shoulder.L": "chest", "upper_arm.L": "shoulder.L", "forearm.L": "upper_arm.L", "hand.L": "forearm.L",
    "shoulder.R": "chest", "upper_arm.R": "shoulder.R", "forearm.R": "upper_arm.R", "hand.R": "forearm.R",
    "thigh.L": "hips", "shin.L": "thigh.L", "foot.L": "shin.L",
    "thigh.R": "hips", "shin.R": "thigh.R", "foot.R": "shin.R",
}}
BONE_POSITIONS = {{
    "root": (0.0, 0.0, 0.0), "hips": (0.0, 0.0, 1.0),
    "spine": (0.0, 0.0, 1.55), "chest": (0.0, 0.0, 2.15),
    "neck": (0.0, 0.0, 2.75), "head": (0.0, 0.0, 3.05),
    "shoulder.L": (-0.48, 0.0, 2.25), "upper_arm.L": (-0.62, 0.0, 2.05),
    "forearm.L": (-0.82, -0.02, 1.72), "hand.L": (-1.04, -0.04, 1.40),
    "shoulder.R": (0.48, 0.0, 2.25), "upper_arm.R": (0.62, 0.0, 2.05),
    "forearm.R": (0.82, -0.02, 1.72), "hand.R": (1.04, -0.04, 1.40),
    "thigh.L": (-0.34, 0.0, 1.0), "shin.L": (-0.45, 0.0, 0.55), "foot.L": (-0.45, -0.02, 0.10),
    "thigh.R": (0.34, 0.0, 1.0), "shin.R": (0.45, 0.0, 0.55), "foot.R": (0.45, -0.02, 0.10),
}}


def create_character_armature(entity_id, location):
    bpy.ops.object.armature_add(location=location)
    armature = bpy.context.object
    armature.name = entity_id + "__armature"
    armature.data.name = entity_id + "__skeleton"
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = armature.data.edit_bones
    for bone_name in MINIMAL_BONES:
        bone = edit_bones.new(bone_name)
        bone.head = BONE_POSITIONS[bone_name]
        bone.tail = tuple(BONE_POSITIONS[bone_name][index] + (0.12 if index == 2 else 0.0) for index in range(3))
        if bone_name in PARENTS:
            bone.parent = edit_bones[PARENTS[bone_name]]
    bpy.ops.object.mode_set(mode="OBJECT")
    armature["entity_kind"] = "armature"
    armature["rig_contract"] = "minimal_articulated_v1"
    return armature


def bind_mesh_to_armature(mesh_object, armature):
    bone_positions = {{bone.name: bone.head_local.copy() for bone in armature.data.bones}}
    for bone_name in MINIMAL_BONES:
        group = mesh_object.vertex_groups.new(name=bone_name)
        nearest_bone = [
            vertex.index
            for vertex in mesh_object.data.vertices
            if min(
                MINIMAL_BONES,
                key=lambda candidate: (vertex.co - bone_positions[candidate]).length,
            ) == bone_name
        ]
        if nearest_bone:
            group.add(nearest_bone, 1.0, "REPLACE")
    modifier = mesh_object.modifiers.new("ArticulatedArmature", "ARMATURE")
    modifier.object = armature
    return modifier


def add_hand_ik_constraint(armature, hand_name, target):
    if hand_name not in {{"hand.L", "hand.R"}}:
        raise ValueError("hand_name must be hand.L or hand.R")
    pose_bone = armature.pose.bones.get(hand_name)
    if pose_bone is None:
        raise ValueError("hand bone is missing: " + hand_name)
    constraint = pose_bone.constraints.new(type="IK")
    constraint.name = hand_name + "__IK"
    constraint.target = target
    constraint.chain_count = 2
    return constraint


def create_hand_ik_targets(armature, entity_id):
    targets = {{}}
    for hand_name in ("hand.L", "hand.R"):
        target = bpy.data.objects.new(entity_id + "__IK__" + hand_name, None)
        bpy.context.collection.objects.link(target)
        target.empty_display_type = "SPHERE"
        target.empty_display_size = 0.18
        target.location = tuple(armature.location[index] + BONE_POSITIONS[hand_name][index] for index in range(3))
        target["ik_target_for"] = entity_id + ":" + hand_name
        add_hand_ik_constraint(armature, hand_name, target)
        targets[hand_name] = target
    return targets


def insert_walk_cycle(armature, frame_start, frame_end):
    midpoint = (int(frame_start) + int(frame_end)) // 2
    for frame, sign in ((int(frame_start), 1.0), (midpoint, -1.0), (int(frame_end), 1.0)):
        for side, side_sign in (("L", sign), ("R", -sign)):
            bone = armature.pose.bones.get("thigh." + side)
            if bone:
                bone.rotation_mode = "XYZ"
                bone.rotation_euler[1] = 0.35 * side_sign
                bone.keyframe_insert(data_path="rotation_euler", index=1, frame=frame)


def append_ellipsoid(vertices, faces, center, radii, segments=24, rings=12):
    base = len(vertices)
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        vertical = math.cos(phi)
        radial = math.sin(phi)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append((
                center[0] + radii[0] * radial * math.cos(theta),
                center[1] + radii[1] * radial * math.sin(theta),
                center[2] + radii[2] * vertical,
            ))
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = base + ring * segments + segment
            b = base + ring * segments + next_segment
            c = base + (ring + 1) * segments + next_segment
            d = base + (ring + 1) * segments + segment
            faces.append((a, b, c, d))


def append_capsule(vertices, faces, start, end, radius, segments=16, rings=10):
    start_vec, end_vec = Vector(start), Vector(end)
    axis = end_vec - start_vec
    length = max(axis.length, 1e-6)
    axis.normalize()
    reference = Vector((0.0, 0.0, 1.0)) if abs(axis.z) < 0.9 else Vector((1.0, 0.0, 0.0))
    basis_u = axis.cross(reference).normalized()
    basis_v = axis.cross(basis_u).normalized()
    base = len(vertices)
    for ring in range(rings + 1):
        progress = ring / rings
        center = start_vec.lerp(end_vec, progress)
        local_radius = radius * (0.72 + 0.28 * math.sin(math.pi * progress))
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            point = center + basis_u * (math.cos(theta) * local_radius) + basis_v * (math.sin(theta) * local_radius)
            vertices.append(tuple(point))
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = base + ring * segments + segment
            b = base + ring * segments + next_segment
            c = base + (ring + 1) * segments + next_segment
            d = base + (ring + 1) * segments + segment
            faces.append((a, b, c, d))


def append_box(vertices, faces, center, size):
    base = len(vertices)
    half = tuple(float(value) / 2.0 for value in size)
    cx, cy, cz = center
    vertices.extend([
        (cx - half[0], cy - half[1], cz - half[2]), (cx + half[0], cy - half[1], cz - half[2]),
        (cx + half[0], cy + half[1], cz - half[2]), (cx - half[0], cy + half[1], cz - half[2]),
        (cx - half[0], cy - half[1], cz + half[2]), (cx + half[0], cy - half[1], cz + half[2]),
        (cx + half[0], cy + half[1], cz + half[2]), (cx - half[0], cy + half[1], cz + half[2]),
    ])
    faces.extend([
        (base + 0, base + 1, base + 2, base + 3), (base + 4, base + 7, base + 6, base + 5),
        (base + 0, base + 4, base + 5, base + 1), (base + 1, base + 5, base + 6, base + 2),
        (base + 2, base + 6, base + 7, base + 3), (base + 4, base + 0, base + 3, base + 7),
    ])


def append_box_surface(vertices, faces, center, size, steps=6):
    """Create a subdivided closed box surface with non-primitive topology."""
    cx, cy, cz = center
    sx, sy, sz = (float(value) / 2.0 for value in size)
    face_specs = [
        ((sx, 0.0, 0.0), (0.0, sy, 0.0), (0.0, 0.0, sz)),
        ((-sx, 0.0, 0.0), (0.0, 0.0, sz), (0.0, sy, 0.0)),
        ((0.0, sy, 0.0), (sx, 0.0, 0.0), (0.0, 0.0, sz)),
        ((0.0, -sy, 0.0), (0.0, 0.0, sz), (sx, 0.0, 0.0)),
        ((0.0, 0.0, sz), (sx, 0.0, 0.0), (0.0, sy, 0.0)),
        ((0.0, 0.0, -sz), (0.0, sy, 0.0), (sx, 0.0, 0.0)),
    ]
    for origin, axis_u, axis_v in face_specs:
        base = len(vertices)
        for row in range(steps + 1):
            v = row / steps - 0.5
            for column in range(steps + 1):
                u = column / steps - 0.5
                vertices.append((
                    cx + origin[0] + axis_u[0] * u * 2.0 + axis_v[0] * v * 2.0,
                    cy + origin[1] + axis_u[1] * u * 2.0 + axis_v[1] * v * 2.0,
                    cz + origin[2] + axis_u[2] * u * 2.0 + axis_v[2] * v * 2.0,
                ))
        stride = steps + 1
        for row in range(steps):
            for column in range(steps):
                a = base + row * stride + column
                b = a + 1
                c = b + stride
                d = a + stride
                faces.append((a, b, c, d))


def append_revolved_profile(vertices, faces, profile, segments=32):
    base = len(vertices)
    for height, radius in profile:
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append((radius * math.cos(theta), radius * math.sin(theta), height))
    for row in range(len(profile) - 1):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = base + row * segments + segment
            b = base + row * segments + next_segment
            c = base + (row + 1) * segments + next_segment
            d = base + (row + 1) * segments + segment
            faces.append((a, b, c, d))


def mesh_object(entity_id, kind, vertices, faces, material):
    mesh = bpy.data.meshes.new(f"{{entity_id}}_detailed_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(entity_id, mesh)
    bpy.context.collection.objects.link(obj)
    obj["entity_id"] = entity_id
    obj["entity_kind"] = kind
    obj["geometry_style"] = "detailed_parametric_v1"
    obj.data.materials.append(material)
    return obj


def detailed_character(entity_id, kind, material):
    vertices, faces = [], []
    append_ellipsoid(vertices, faces, (0.0, 0.0, 1.65), (0.72, 0.46, 1.05), segments=28, rings=14)
    append_ellipsoid(vertices, faces, (0.0, 0.0, 3.05), (0.43, 0.38, 0.48), segments=28, rings=14)
    append_capsule(vertices, faces, (-0.34, 0.0, 1.0), (-0.45, 0.0, 0.10), 0.20, segments=20, rings=12)
    append_capsule(vertices, faces, (0.34, 0.0, 1.0), (0.45, 0.0, 0.10), 0.20, segments=20, rings=12)
    append_capsule(vertices, faces, (-0.56, 0.0, 2.25), (-1.02, -0.04, 1.48), 0.15, segments=20, rings=12)
    append_capsule(vertices, faces, (0.56, 0.0, 2.25), (1.02, -0.04, 1.48), 0.15, segments=20, rings=12)
    append_ellipsoid(vertices, faces, (-1.04, -0.04, 1.40), (0.18, 0.16, 0.18), segments=20, rings=10)
    append_ellipsoid(vertices, faces, (1.04, -0.04, 1.40), (0.18, 0.16, 0.18), segments=20, rings=10)
    return mesh_object(entity_id, kind, vertices, faces, material)


def detailed_support(entity_id, kind, material, drop=False):
    vertices, faces = [], []
    if drop:
        append_box_surface(vertices, faces, (0.0, 0.0, 0.18), (3.2, 2.4, 0.36), steps=6)
        append_box(vertices, faces, (0.0, 0.0, 0.02), (2.7, 1.9, 0.18))
    else:
        append_box_surface(vertices, faces, (0.0, 0.0, 1.20), (4.8, 2.8, 0.36), steps=7)
        for x in (-2.0, 2.0):
            for y in (-1.0, 1.0):
                append_box(vertices, faces, (x, y, 0.55), (0.34, 0.34, 1.1))
                append_box(vertices, faces, (x * 0.92, y, 0.96), (0.55, 0.18, 0.18))
    return mesh_object(entity_id, kind, vertices, faces, material)


def detailed_occluder(entity_id, kind, material):
    vertices, faces = [], []
    append_box_surface(vertices, faces, (0.0, 0.0, 1.7), (0.34, 3.6, 3.4), steps=6)
    for y in (-1.65, 1.65):
        append_box(vertices, faces, (0.0, y, 1.7), (0.55, 0.22, 3.8))
    append_box(vertices, faces, (0.0, 0.0, 3.45), (0.55, 3.7, 0.25))
    return mesh_object(entity_id, kind, vertices, faces, material)


def detailed_opening(entity_id, kind, material):
    vertices, faces = [], []
    append_box_surface(vertices, faces, (-0.7, 0.0, 1.5), (0.28, 2.8, 3.0), steps=5)
    append_box_surface(vertices, faces, (0.7, 0.0, 1.5), (0.28, 2.8, 3.0), steps=5)
    append_box_surface(vertices, faces, (0.0, 0.0, 2.95), (1.7, 2.8, 0.28), steps=5)
    return mesh_object(entity_id, kind, vertices, faces, material)


def detailed_prop(entity_id, kind, material):
    vertices, faces = [], []
    lower = entity_id.lower()
    if "cup" in lower:
        append_revolved_profile(vertices, faces, [(-0.48, 0.34), (-0.42, 0.40), (-0.30, 0.43), (0.20, 0.45), (0.38, 0.48), (0.45, 0.52), (0.50, 0.47), (0.54, 0.52), (0.58, 0.47)], segments=36)
        append_capsule(vertices, faces, (0.44, 0.0, 0.22), (0.68, 0.0, 0.20), 0.10, segments=16, rings=8)
    elif "book" in lower:
        append_box_surface(vertices, faces, (0.0, 0.0, 0.0), (1.5, 1.05, 0.16), steps=7)
        append_box_surface(vertices, faces, (0.0, 0.0, 0.12), (1.56, 1.10, 0.08), steps=5)
        for offset in (-0.34, -0.17, 0.0, 0.17, 0.34):
            append_box(vertices, faces, (0.16, offset, 0.18), (1.0, 0.025, 0.025))
    elif "ball" in lower:
        append_ellipsoid(vertices, faces, (0.0, 0.0, 0.0), (0.50, 0.50, 0.50), segments=32, rings=20)
    else:
        append_box_surface(vertices, faces, (0.0, 0.0, 0.0), (0.9, 0.9, 0.9), steps=8)
        append_box(vertices, faces, (0.0, 0.0, 0.48), (0.72, 0.72, 0.04))
    return mesh_object(entity_id, kind, vertices, faces, material)


def add_entity(entity_id, entity_spec, material):
    kind = entity_spec.get("kind", "prop")
    if kind == "character":
        return detailed_character(entity_id, kind, material)
    elif kind == "support":
        return detailed_support(entity_id, kind, material, drop=entity_id == "drop_zone")
    elif kind == "occluder":
        return detailed_occluder(entity_id, kind, material)
    elif kind == "opening":
        return detailed_opening(entity_id, kind, material)
    return detailed_prop(entity_id, kind, material)


def initial_location(entity_id, entity_spec):
    director_entity = DIRECTOR_TRAJECTORIES.get("entities", {{}}).get(entity_id, {{}})
    states = director_entity.get("states", [])
    if states:
        return tuple(states[0]["position"])
    kind = entity_spec.get("kind", "prop")
    layout = PROXY_SPEC.get("layout", {{}})
    support = layout.get("support_position", (2.0, 0.0, 0.0))
    if kind == "character":
        actor_positions = layout.get("actor_start_positions", {{}})
        if entity_id in actor_positions:
            return tuple(actor_positions[entity_id])
        actor_index = sum(
            1 for entity in PROXY_SPEC.get("entities", [])
            if entity.get("kind") == "character" and entity.get("id") <= entity_id
        ) - 1
        return (
            float(support[0]) + 1.8,
            float(support[1]) + 1.8 + actor_index * 1.8,
            0.0,
        )
    if entity_spec.get("role") == "environment" or kind == "support":
        if entity_spec.get("id") == "drop_zone":
            return tuple(layout.get("drop_zone_position", (0.0, 2.0, 0.0)))
        return tuple(layout.get("support_position", (0.0, 0.0, 0.0)))
    if kind == "occluder":
        support = layout.get("support_position", (0.0, 0.0, 0.0))
        return (float(support[0]) + 0.5, float(support[1]) + 0.5, 1.0)
    return (float(support[0]), float(support[1]), 1.0)


def add_light(material=None):
    bpy.ops.object.light_add(type="AREA", location=(4.0, -4.0, 6.0))
    key = bpy.context.object
    key.name = "ProxyKeyLight"
    key.data.energy = 900.0
    key.data.shape = "DISK"
    key.data.size = 5.0

    bpy.ops.object.light_add(type="AREA", location=(-4.0, -1.0, 3.5))
    fill = bpy.context.object
    fill.name = "ProxyFillLight"
    fill.data.energy = 520.0
    fill.data.shape = "DISK"
    fill.data.size = 4.0

    bpy.ops.object.light_add(type="AREA", location=(1.5, 5.0, 5.5))
    rim = bpy.context.object
    rim.name = "ProxyRimLight"
    rim.data.energy = 760.0
    rim.data.shape = "DISK"
    rim.data.size = 3.0
    return [key, fill, rim]


def look_at(camera, target):
    direction = Vector(target) - camera.location
    if direction.length > 0:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera():
    global CAMERA_TARGET
    bpy.ops.object.camera_add(location=(7.0, -8.0, 5.0))
    camera = bpy.context.object
    camera.name = "ProxyCamera"
    bpy.context.scene.camera = camera
    target = bpy.data.objects.new("ProxyCameraLookTarget", None)
    bpy.context.collection.objects.link(target)
    constraint = camera.constraints.new(type="TRACK_TO")
    constraint.name = "DirectorTrackTo"
    constraint.target = target
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis = "UP_Y"
    camera["camera_constraint"] = "TRACK_TO"
    camera["camera_dsl"] = "orbit_follow_dolly_hold_v1"
    camera["multi_target_framing"] = True
    camera["visibility_predicate_checks"] = True
    camera["continuity_group_checks"] = True
    CAMERA_TARGET = target
    return camera


def configure_render(scene, manifest):
    preferred_engine = manifest["render_settings"].get("engine", "BLENDER_EEVEE_NEXT")
    for candidate in (preferred_engine, "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
        try:
            scene.render.engine = candidate
            manifest["render_settings"]["engine"] = candidate
            break
        except (TypeError, ValueError):
            continue
    else:
        raise RuntimeError("connected Blender exposes no supported render engine")
    settings = manifest["render_settings"]
    scene.render.resolution_x = int(settings.get("resolution", [256, 256])[0])
    scene.render.resolution_y = int(settings.get("resolution", [256, 256])[1])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(manifest["fps"])
    scene.frame_start = int(manifest["frame_start"])
    scene.frame_end = int(manifest["frame_end"])
    # The connected Blender may be built without FFMPEG; the host assembles MP4
    # from this PNG sequence after MCP execution.
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(FRAMES_DIR / "animation" / "frame_")
    scene.world.color = (0.04, 0.04, 0.04)


def _child_of_constraint(prop, target, name, subtarget=None):
    constraint = prop.constraints.get(name) or prop.constraints.new(type="CHILD_OF")
    constraint.name = name
    constraint.target = target
    if subtarget:
        constraint.subtarget = subtarget
    prop["attachment_constraint_mode"] = "CHILD_OF"
    prop["attachment_constraint_target"] = target.name
    prop["attachment_constraint_subtarget"] = subtarget or ""
    return constraint


def _key_constraint_influence(constraint, frame, influence):
    constraint.influence = float(influence)
    constraint.keyframe_insert(data_path="influence", frame=int(frame))
    key = (constraint.id_data.name, constraint.name)
    CONSTRAINT_KEYFRAMES.setdefault(key, []).append(
        {{"frame": int(frame), "influence": float(influence)}}
    )


CONSTRAINT_KEYFRAMES = {{}}


def apply_attachment_constraints(objects, armatures):
    """Compile attach/transfer/detach as Child Of influence curves."""
    for prop_id, trajectory in PLAN["entities"].items():
        prop = objects.get(prop_id)
        if prop is None:
            continue
        active = dict()
        for event in sorted(trajectory.get("attachment_events", []), key=lambda item: int(item["frame"])):
            frame = int(event["frame"])
            action = event.get("action")
            if action in {"attach", "transfer"}:
                actor_id = event.get("object_id")
                armature = armatures.get(actor_id)
                if armature is None:
                    continue
                if action == "transfer":
                    for previous in active.values():
                        _key_constraint_influence(previous, frame, 0.0)
                constraint = _child_of_constraint(
                    prop,
                    armature,
                    "Attach__" + str(actor_id),
                    event.get("subtarget") or "hand.R",
                )
                if frame > 1:
                    _key_constraint_influence(constraint, frame - 1, 0.0)
                _key_constraint_influence(constraint, frame, 1.0)
                active[actor_id] = constraint
            elif action == "detach":
                for constraint in active.values():
                    _key_constraint_influence(constraint, frame, 0.0)
                active.clear()
                support = objects.get("support_surface") or objects.get("drop_zone") or objects.get("table")
                if support is not None:
                    constraint = _child_of_constraint(prop, support, "Place__support_surface")
                    _key_constraint_influence(constraint, frame, 1.0)


def _is_attached_at(trajectory, frame):
    attached = False
    for event in sorted(trajectory.get("attachment_events", []), key=lambda item: int(item["frame"])):
        if int(event["frame"]) > int(frame):
            break
        if event.get("action") in {"attach", "transfer"}:
            attached = True
        elif event.get("action") == "detach":
            attached = False
    return attached


def audit_attachment_penetration(scene, objects):
    findings = []
    checked = []
    for prop_id, trajectory in PLAN["entities"].items():
        prop = objects.get(prop_id)
        if prop is None:
            continue
        for event in trajectory.get("attachment_events", []):
            if event.get("action") not in {{"attach", "transfer"}}:
                continue
            actor = objects.get(event.get("object_id"))
            if actor is None:
                continue
            frame = int(event["frame"])
            scene.frame_set(frame)
            prop_center = prop.matrix_world.translation.copy()
            torso_center = actor.matrix_world @ Vector((0.0, 0.0, 1.65))
            distance = float((prop_center - torso_center).length)
            checked.append({{
                "prop_id": prop_id,
                "actor_id": event.get("object_id"),
                "frame": frame,
                "torso_distance": distance,
            }})
            if distance < 0.35:
                findings.append({{
                    "failure_id": "no_prop_penetration",
                    "owner": "director_trajectory",
                    "category": "interaction_geometry",
                    "severity": "hard",
                    "message": f"{{prop_id}} penetrates {{event.get('object_id')}} torso at frame {{frame}}",
                    "evidence": [prop_id, str(event.get("object_id")), str(frame)],
                }})
    return findings, checked


def _constraint_telemetry(obj):
    return [
        {{
            "name": constraint.name,
            "type": constraint.type,
            "target": constraint.target.name if constraint.target else None,
            "subtarget": constraint.subtarget,
            "influence": constraint.influence,
            "influence_keyframes": CONSTRAINT_KEYFRAMES.get((obj.name, constraint.name), []),
        }}
        for constraint in obj.constraints
        if constraint.type == "CHILD_OF"
    ]


def animate_entities(objects):
    for entity_id, trajectory in PLAN["entities"].items():
        obj = objects[entity_id]
        for state in trajectory["states"]:
            # During carry/handoff the Child Of constraint owns translation;
            # emitting a competing location curve would reintroduce drift.
            if _is_attached_at(trajectory, int(state["frame"])):
                continue
            obj.location = tuple(state["position"])
            obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
            obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
            obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))


CAMERA_FINDINGS = []
CAMERA_TARGET = None
ATTACHMENT_PENETRATION = []


def orbit_points(center, radius, start_angle, end_angle, height, frame_count):
    """Parameterised circular arc; never replace an orbit with a chord."""
    frame_count = max(2, int(frame_count))
    step = (float(end_angle) - float(start_angle)) / float(frame_count - 1)
    return [
        (
            float(center[0]) + float(radius) * math.cos(math.radians(float(start_angle) + step * index)),
            float(center[1]) + float(radius) * math.sin(math.radians(float(start_angle) + step * index)),
            float(height),
        )
        for index in range(frame_count)
    ]


def target_bounds(objects, target_ids):
    points = [Vector(objects[target_id].location) for target_id in target_ids if target_id in objects]
    if not points:
        return Vector((0.0, 0.0, 1.0)), 1.0
    center = sum(points, Vector()) / len(points)
    radius = max((point - center).length for point in points)
    return center, max(0.5, radius)


def camera_point_for_shot(shot, center, bound_radius, sample_index, sample_count):
    trajectory_type = shot.get("trajectory_type", "hold")
    distance_range = shot.get("distance_range", (4.0, 8.0))
    minimum_distance = max(2.0, float(distance_range[0]))
    maximum_distance = max(minimum_distance, float(distance_range[1]))
    if trajectory_type == "orbit":
        radius = max(minimum_distance, min(maximum_distance, bound_radius * 2.0 + 1.5))
        points = orbit_points(
            center=tuple(center),
            radius=radius,
            start_angle=-55.0,
            end_angle=55.0,
            height=float(center.z) + max(1.5, radius * 0.45),
            frame_count=max(8, int(sample_count)),
        )
        return Vector(points[min(sample_index, len(points) - 1)])
    if trajectory_type == "dolly":
        start = Vector((7.0, -8.0, 5.0))
        end = Vector((3.5, -4.0, 2.8))
        fraction = float(sample_index) / float(max(1, sample_count - 1))
        return start.lerp(end, fraction)
    if trajectory_type == "follow":
        offset = Vector((7.0, -8.0, 5.0)) - center
        offset_length = max(offset.length, 1e-6)
        offset = offset / offset_length * max(minimum_distance, min(maximum_distance, offset_length))
        return center + offset
    return Vector((7.0, -8.0, 5.0))


def animate_camera(camera, objects):
    shots = (DIRECTOR_CAMERA or PLAN["camera"]).get("shots", [])
    for shot in shots:
        target_ids = [target_id for target_id in shot.get("target_ids", []) if target_id in objects]
        if not target_ids:
            CAMERA_FINDINGS.append({{
                "failure_id": "camera_target_missing",
                "owner": "director_camera",
                "category": "camera_coverage",
                "severity": "error",
                "message": "camera shot has no executable target object",
                "evidence": [shot.get("shot_id", "unknown")],
            }})
            continue
        start_frame = int(shot["start_frame"])
        end_frame = int(shot["end_frame"])
        sample_count = max(8, min(64, max(2, end_frame - start_frame + 1)))
        center, bound_radius = target_bounds(objects, target_ids)
        camera.data.lens = float(shot.get("lens_mm", 50.0))
        for sample_index in range(sample_count):
            frame = round(start_frame + (end_frame - start_frame) * sample_index / max(1, sample_count - 1))
            scene = bpy.context.scene
            scene.frame_set(int(frame))
            center, bound_radius = target_bounds(objects, target_ids)
            camera.location = camera_point_for_shot(shot, center, bound_radius, sample_index, sample_count)
            if CAMERA_TARGET is not None:
                CAMERA_TARGET.location = center
                CAMERA_TARGET.keyframe_insert(data_path="location", frame=int(frame))
            look_at(camera, center)
            camera.keyframe_insert(data_path="location", frame=int(frame))
            camera.keyframe_insert(data_path="rotation_euler", frame=int(frame))
            camera.data.keyframe_insert(data_path="lens", frame=int(frame))


def _ray_occlusion(scene, camera, target):
    origin = camera.matrix_world.translation.copy()
    bounds = [target.matrix_world @ Vector(corner) for corner in target.bound_box]
    destination = sum(bounds, Vector()) / max(1, len(bounds)) if bounds else target.matrix_world.translation.copy()
    direction = destination - origin
    distance = direction.length
    if distance <= 1e-6:
        return 0.0
    direction.normalize()
    try:
        hit, _location, _normal, _index, hit_object, _matrix = scene.ray_cast(
            bpy.context.evaluated_depsgraph_get(), origin, direction, distance=max(0.0, distance - 0.05)
        )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return 0.0
    return 1.0 if hit and hit_object != target else 0.0


def audit_camera_visibility(scene, camera, objects, shots):
    for shot in shots:
        target_ids = [target_id for target_id in shot.get("target_ids", []) if target_id in objects]
        if not target_ids:
            continue
        frames = sorted({{
            int(shot["start_frame"]),
            int(shot["end_frame"]),
            (int(shot["start_frame"]) + int(shot["end_frame"])) // 2,
        }})
        maximum_observed = 0.0
        for frame in frames:
            scene.frame_set(frame)
            for target_id in target_ids:
                maximum_observed = max(maximum_observed, _ray_occlusion(scene, camera, objects[target_id]))
        allowed = float(shot.get("max_occlusion", 1.0))
        if maximum_observed > allowed:
            CAMERA_FINDINGS.append({{
                "failure_id": "camera_occlusion_exceeded",
                "owner": "director_camera",
                "category": "camera_coverage",
                "severity": "error",
                "message": f"{{shot.get('shot_id', 'unknown')}} occlusion {{maximum_observed:.3f}} exceeds {{allowed:.3f}}",
                "evidence": target_ids,
            }})


def audit_camera_continuity(shots):
    last_by_group = {{}}
    for shot in shots:
        group = shot.get("continuity_group")
        if not group:
            continue
        previous = last_by_group.get(group)
        previous_side = previous.get("axis_side") if previous else None
        current_side = shot.get("axis_side")
        if previous_side and current_side and previous_side != current_side:
            CAMERA_FINDINGS.append({{
                "failure_id": "camera_continuity_violation",
                "owner": "director_camera",
                "category": "camera_continuity",
                "severity": "error",
                "message": f"continuity group {{group}} crosses the declared axis",
                "evidence": [previous.get("shot_id", "unknown"), shot.get("shot_id", "unknown")],
            }})
        last_by_group[group] = shot


def validate_transfer(prop_id, interaction, event_by_id, attachment_events):
    transfer_event = event_by_id.get(interaction.get("transfer_event_id"))
    if transfer_event is None:
        return {{"valid": False, "reason": "missing_transfer_event"}}
    fps = int(INITIAL_MANIFEST["fps"])
    window_start = max(1, round(float(transfer_event["start"]) * fps) + 1)
    window_end = max(window_start, round(float(transfer_event["end"]) * fps) + 1)
    giver_id = interaction.get("giver_id")
    receiver_id = interaction.get("receiver_id")
    in_window = [
        item for item in attachment_events
        if item.get("subject_id") == prop_id and window_start <= int(item.get("frame", 0)) <= window_end
    ]
    transfer = next((item for item in in_window if item.get("action") == "transfer"), None)
    giver_id = interaction.get("giver_id")
    receiver_id = interaction.get("receiver_id")
    handoff_frame = int(transfer.get("frame")) if transfer else window_start
    giver_curve = CONSTRAINT_KEYFRAMES.get((prop_id, "Attach__" + str(giver_id)), [])
    receiver_curve = CONSTRAINT_KEYFRAMES.get((prop_id, "Attach__" + str(receiver_id)), [])
    giver_released = any(
        int(item.get("frame", -1)) == handoff_frame and float(item.get("influence", 1.0)) <= 0.01
        for item in giver_curve
    )
    receiver_acquired = any(
        int(item.get("frame", -1)) == handoff_frame and float(item.get("influence", 0.0)) >= 0.99
        for item in receiver_curve
    )
    constraint_curve = {{
        "valid": giver_released and receiver_acquired,
        "handoff_frame": handoff_frame,
        "giver_curve": giver_curve,
        "receiver_curve": receiver_curve,
    }}
    # The single transfer marker is compiled as a validated atomic pair: the
    # giver releases and the receiver acquires on the same handoff frame.
    return {{
        "valid": bool(
            transfer
            and transfer.get("object_id") == receiver_id
            and giver_id
            and receiver_id
            and constraint_curve["valid"]
        ),
        "window": [window_start, window_end],
        "giver_detach": {{"actor_id": giver_id, "frame": window_start}},
        "receiver_attach": {{"actor_id": receiver_id, "frame": window_start}},
        "observed_attachment": transfer,
        "constraint_curve": constraint_curve,
    }}


def write_telemetry(objects, camera, manifest):
    event_by_id = {{event["id"]: event for event in DIRECTOR_PLAN.get("events", [])}}
    interaction_state = {{}}
    transfer_constraints = []
    for interaction in DIRECTOR_PLAN.get("interactions", []):
        prop_id = interaction["prop_id"]
        attachment_events = PLAN["entities"].get(prop_id, {{}}).get("attachment_events", [])
        transfer_state = validate_transfer(prop_id, interaction, event_by_id, attachment_events)
        interaction_state[interaction["id"]] = {{
            "prop_id": prop_id,
            "giver_id": interaction.get("giver_id"),
            "receiver_id": interaction.get("receiver_id"),
            "final_owner_id": interaction.get("final_owner_id"),
            "final_support_id": interaction.get("final_support_id"),
            "transfer": transfer_state,
        }}
        if interaction.get("transfer_event_id"):
            transfer_constraints.append(transfer_state)
    camera_payload = DIRECTOR_CAMERA or PLAN["camera"]
    telemetry = {{
        "blender_version": bpy.app.version_string,
        "frame_start": manifest["frame_start"],
        "frame_end": manifest["frame_end"],
        "fps": manifest["fps"],
        "director_plan_hash": manifest.get("director_plan_hash"),
        "objects": {{
            entity_id: {{
                "kind": obj.get("entity_kind", "unknown"),
                "location": list(obj.location),
                "keyframe_count": len(PLAN["entities"].get(entity_id, dict()).get("states", [])),
            }}
            for entity_id, obj in objects.items()
        }},
        "attachment_constraints": {{
            entity_id: _constraint_telemetry(obj)
            for entity_id, obj in objects.items()
            if any(constraint.type == "CHILD_OF" for constraint in obj.constraints)
        }},
        "attachment_penetration": ATTACHMENT_PENETRATION,
        "camera": dict(name=camera.name, active=(bpy.context.scene.camera.name == camera.name)),
        "camera_findings": CAMERA_FINDINGS,
        "proxy_scene": {{
            "scene_id": PROXY_SPEC.get("scene_id"),
            "scene_seed": PROXY_SPEC.get("scene_seed"),
            "path_shape": PROXY_SPEC.get("layout", {{}}).get("path_shape"),
        }},
        "camera_shots": [
            {{
                "shot_id": shot["shot_id"],
                "trajectory_type": shot.get("trajectory_type", "hold"),
                "target_ids": shot.get("target_ids", []),
                "required_event_ids": shot.get("required_event_ids", []),
                "visibility": shot.get("visibility_predicates", {{}}),
                "max_occlusion": shot.get("max_occlusion", 1.0),
            }}
            for shot in camera_payload.get("shots", [])
        ],
        "current_owner_by_event": DIRECTOR_TRAJECTORIES.get("current_owner_by_event", {{}}),
        "final_support_by_prop": DIRECTOR_TRAJECTORIES.get("final_support_by_prop", {{}}),
        "interaction_state": interaction_state,
        "transfer_constraints": transfer_constraints,
        "visibility": [
            {{
                "shot_id": shot["shot_id"],
                "target_ids": shot.get("target_ids", []),
                "predicates": shot.get("visibility_predicates", {{}}),
                "max_occlusion": shot.get("max_occlusion", 1.0),
            }}
            for shot in camera_payload.get("shots", [])
        ],
        "event_observability": PLAN.get("event_observability", []),
        "render_settings": manifest["render_settings"],
    }}
    (OUTPUT_DIR / "telemetry.json").write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")


def write_sample_frames(scene):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    scene.render.image_settings.file_format = "PNG"
    for frame in SAMPLE_FRAMES:
        scene.frame_set(int(frame))
        relative = f"frame_{{int(frame):06d}}.png"
        scene.render.filepath = str(FRAMES_DIR / relative)
        bpy.ops.render.render(write_still=True)
        index.append({{"frame": int(frame), "path": relative}})
    (FRAMES_DIR / "index.json").write_text(json.dumps({{"frames": index}}, indent=2), encoding="utf-8")


def update_manifest(manifest):
    # The host writes code_hash after freezing this source.  Reload the
    # persisted manifest so the generated job cannot overwrite that binding
    # with the pre-freeze value embedded in INITIAL_MANIFEST.
    persisted_manifest_path = OUTPUT_DIR / "run_manifest.json"
    if persisted_manifest_path.is_file():
        try:
            persisted = json.loads(persisted_manifest_path.read_text(encoding="utf-8"))
            if persisted.get("code_hash"):
                manifest["code_hash"] = persisted["code_hash"]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    manifest["blender_version"] = bpy.app.version_string
    manifest["state"] = "rendered"
    manifest["fingerprint"] = canonical_hash({{
        "prompt_hash": manifest["prompt_hash"],
        "plan_hash": manifest["plan_hash"],
        "director_plan_hash": manifest.get("director_plan_hash"),
        "harness_version": manifest["harness_version"],
        "evaluator_version": manifest["evaluator_version"],
        "blender_version": manifest["blender_version"],
        "render_settings": manifest["render_settings"],
        "rollout_seed": manifest.get("rollout_seed"),
    }})
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
(FRAMES_DIR / "animation").mkdir(parents=True, exist_ok=True)
reset_scene()
scene = bpy.context.scene
ground_plane()
objects = {{
    entity["id"]: add_entity(entity["id"], entity, material_for(entity))
    for entity in {json.dumps(proxy_entities, sort_keys=True)}
}}
entity_specs = {{
    entity["id"]: entity
    for entity in {json.dumps(proxy_entities, sort_keys=True)}
}}
for entity_id, obj in objects.items():
    entity_spec = entity_specs[entity_id]
    entity_kind = entity_spec.get("kind", obj.get("entity_kind", "prop"))
    obj.location = initial_location(entity_id, entity_spec)
    if entity_spec.get("role") == "environment" and entity_kind == "support":
        obj.scale = tuple(PROXY_SPEC.get("layout", {{}}).get("support_scale", obj.scale))
    elif entity_kind == "prop":
        obj.scale = tuple(PROXY_SPEC.get("layout", {{}}).get("object_scale", obj.scale))
armatures = {{}}
ik_targets = {{}}
for entity_id, obj in objects.items():
    if entity_specs[entity_id].get("kind") == "character":
        armature = create_character_armature(entity_id, tuple(obj.location))
        bind_mesh_to_armature(obj, armature)
        ik_targets[entity_id] = create_hand_ik_targets(armature, entity_id)
        insert_walk_cycle(armature, INITIAL_MANIFEST["frame_start"], INITIAL_MANIFEST["frame_end"])
        armatures[entity_id] = armature
apply_attachment_constraints(objects, armatures)
penetration_findings, penetration_checks = audit_attachment_penetration(scene, objects)
ATTACHMENT_PENETRATION.extend(penetration_findings)
add_light()
camera = add_camera()
configure_render(scene, INITIAL_MANIFEST)
if DIRECTOR_PLAN:
    (OUTPUT_DIR / "director_plan.json").write_text(
        json.dumps(DIRECTOR_PLAN, indent=2, sort_keys=True),
        encoding="utf-8",
    )
animate_entities(objects)
animate_camera(camera, objects)
camera_payload = (DIRECTOR_CAMERA or PLAN["camera"]).get("shots", [])
audit_camera_visibility(scene, camera, objects, camera_payload)
audit_camera_continuity(camera_payload)
write_telemetry(objects, camera, INITIAL_MANIFEST)
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "proxy.blend"))
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(FRAMES_DIR / "animation" / "frame_")
bpy.ops.render.render(animation=True)
write_sample_frames(scene)
update_manifest(INITIAL_MANIFEST)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(FRAMES_DIR / "animation" / "frame_")
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "proxy.blend"))
'''
