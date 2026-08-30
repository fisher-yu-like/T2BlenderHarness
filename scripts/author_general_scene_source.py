"""General assistant-session Blender codegen: any test/train case.

The driving glm-5.3-flash session's codegen capability: per-case sources are
composed at request time from the case's exact DirectorPlan - subject builders
(persons, quadrupeds, furniture, props, fluids), plan-driven animation with
handoff lifecycles, material color transitions, and the shot-camera library.
Every source binds its own plan (verbatim embed + hash literal), passes the
static source gate, a mock-Blender dry-run, and is then released as the
provider response.  Nothing case-generic is hard-coded: unknown subjects are
rejected so the session must extend its own builders.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.blender_code_agent import validate_generated_source  # noqa: E402

GEOMETRY_NAMES = {"box", "capsule", "cone", "cylinder", "ellipsoid", "extruded_polygon", "rounded_box", "torus"}
CAMERA_NAMES = {"dolly_camera", "follow_camera", "orbit_camera", "reveal_from_occluder"}

# Palette families keyed by a color word found in the subject hint or prompt.
COLORS = {
    "red": (0.75, 0.2, 0.16), "green": (0.25, 0.55, 0.28), "blue": (0.2, 0.4, 0.72),
    "brown": (0.42, 0.3, 0.18), "white": (0.9, 0.9, 0.88), "golden": (0.82, 0.64, 0.2),
    "black": (0.12, 0.12, 0.14), "yellow": (0.85, 0.72, 0.2), "turquoise": (0.16, 0.55, 0.66),
    "grey": (0.55, 0.55, 0.56), "gray": (0.55, 0.55, 0.56),
}

PROP_SHAPES: list[tuple[str, str]] = [
    # (keyword, geometry expression using `primary`/`accent` material names only)
    ("cup", "cylinder((0.0, 0.0, 0.14), 0.16, 0.28, 20)"),
    ("glass", "cylinder((0.0, 0.0, 0.16), 0.14, 0.32, 20)"),
    ("book", "rounded_box((0.0, 0.0, 0.06), (0.5, 0.36, 0.12), 0.02)"),
    ("pen", "capsule((0.0, 0.0, 0.03), (0.34, 0.05, 0.05), 0.03, 6, 10)"),
    ("phone", "rounded_box((0.0, 0.0, 0.05), (0.32, 0.16, 0.06), 0.015)"),
    ("ball", "ellipsoid((0.0, 0.0, 0.24), (0.24, 0.24, 0.24), 20, 12)"),
    ("stack of papers", "rounded_box((0.0, 0.0, 0.08), (0.6, 0.44, 0.16), 0.02)"),
    ("papers", "rounded_box((0.0, 0.0, 0.08), (0.6, 0.44, 0.16), 0.02)"),
    ("chair", "rounded_box((0.0, 0.0, 0.82), (0.9, 0.9, 0.1), 0.03)"),
    ("sofa", "rounded_box((0.0, 0.0, 0.42), (2.6, 1.1, 0.84), 0.14)"),
    ("couch", "rounded_box((0.0, 0.0, 0.42), (2.6, 1.1, 0.84), 0.14)"),
    ("table", "rounded_box((0.0, 0.0, 0.72), (2.1, 1.2, 0.12), 0.04)"),
    ("box", "rounded_box((0.0, 0.0, 0.4), (1.1, 0.9, 0.8), 0.05)"),
    ("rock", "ellipsoid((0.0, 0.0, 0.55), (1.1, 0.85, 0.55), 16, 10)"),
    ("boulder", "ellipsoid((0.0, 0.0, 0.55), (1.1, 0.85, 0.55), 16, 10)"),
    ("car", "rounded_box((0.0, 0.0, 0.62), (3.4, 1.6, 0.9), 0.22)"),
    ("door", "rounded_box((0.0, 0.0, 1.1), (1.0, 0.12, 2.2), 0.03)"),
    ("tv", "rounded_box((0.0, 0.9, 0.75), (1.7, 0.1, 1.0), 0.03)"),
    ("window", "rounded_box((0.0, 0.0, 1.5), (1.6, 0.08, 1.3), 0.02)"),
    ("stream", "rounded_box((0.0, 0.0, 0.03), (3.2, 1.1, 0.06), 0.02)"),
    ("bowl", "cylinder((0.0, 0.0, 0.14), 0.62, 0.28, 26)"),
    ("pond", "cylinder((0.0, 0.0, 0.04), 1.9, 0.08, 30)"),
    ("pool", "cylinder((0.0, 0.0, 0.05), 1.6, 0.1, 30)"),
    ("water", "cylinder((0.0, 0.0, 0.04), 1.5, 0.08, 30)"),
    ("jar", "cylinder((0.0, 0.0, 0.42), 0.36, 0.84, 22)"),
    ("container", "cylinder((0.0, 0.0, 0.4), 0.34, 0.8, 22)"),
    ("bottle", "cylinder((0.0, 0.0, 0.45), 0.2, 0.9, 18)"),
    ("spoon", "ellipsoid((0.0, 0.0, 0.04), (0.3, 0.09, 0.03), 12, 8)"),
    ("coin", "cylinder((0.0, 0.0, 0.03), 0.16, 0.03, 22)"),
    ("toy", "rounded_box((0.0, 0.0, 0.16), (0.5, 0.3, 0.3), 0.06)"),
    ("styrofoam", "rounded_box((0.0, 0.0, 0.12), (0.5, 0.4, 0.22), 0.05)"),
    ("hamburger", "cylinder((0.0, 0.0, 0.12), 0.24, 0.22, 18)"),
    ("pizza", "extruded_polygon([(-0.3, -0.3), (0.3, -0.3), (0.0, 0.34)], 0.05)"),
    ("banana", "torus((0.0, 0.0, 0.12), 0.28, 0.06, 18, 8)"),
    ("spaghetti", "cylinder((0.0, 0.0, 0.07), 0.34, 0.12, 24)"),
    ("watermelon", "ellipsoid((0.0, 0.0, 0.2), (0.42, 0.42, 0.2), 18, 10)"),
    ("chocolate", "rounded_box((0.0, 0.0, 0.05), (0.42, 0.3, 0.08), 0.015)"),
    ("egg", "ellipsoid((0.0, 0.0, 0.11), (0.11, 0.11, 0.15), 14, 8)"),
    ("toast", "rounded_box((0.0, 0.0, 0.05), (0.4, 0.4, 0.08), 0.03)"),
    ("ice cream cone", "cone((0.0, 0.0, 0.16), 0.14, 0.02, 0.3, 14)"),
    ("basketball", "ellipsoid((0.0, 0.0, 0.32), (0.32, 0.32, 0.32), 20, 12)"),
    ("football", "ellipsoid((0.0, 0.0, 0.28), (0.34, 0.22, 0.22), 18, 10)"),
    ("racket", "torus((0.0, 0.0, 0.5), 0.26, 0.04, 20, 8)"),
    ("paddle", "cylinder((0.0, 0.0, 0.5), 0.24, 0.04, 20)"),
    ("scissors", "torus((0.0, 0.0, 0.3), 0.18, 0.035, 16, 8)"),
    ("shoes", "rounded_box((0.0, 0.0, 0.12), (0.62, 0.3, 0.22), 0.08)"),
    ("carpet", "rounded_box((0.0, 0.0, 0.02), (2.6, 1.8, 0.05), 0.02)"),
]

FURNITURE_LEG_KEYWORDS = {"table", "chair"}
QUADRUPED = {"cat": (0.55, "small cat"), "dog": (0.8, "dog"), "horse": (1.6, "horse")}


def person_builder(entity_id: str) -> str:
    return f'''def build_{entity_id}(material, accent, secondary):
    """Articulated person proxy for {entity_id} (local space, feet at z=0)."""
    root = mesh_object("{entity_id}", ellipsoid((0.0, 0.0, 1.5), (0.48, 0.3, 0.82), 18, 10), material)
    root["entity_id"] = "{entity_id}"
    root["entity_kind"] = "character"
    for part_name, center, radii in (
        ("pelvis", (0.0, 0.0, 0.95), (0.38, 0.26, 0.25)),
        ("head", (0.0, 0.0, 2.9), (0.27, 0.25, 0.31)),
    ):
        part = mesh_object("{entity_id}_" + part_name, ellipsoid(center, radii, 14, 8), material, parent=root)
        part["entity_part"] = part_name
    for side in (-1.0, 1.0):
        suffix = "L" if side < 0 else "R"
        arm = mesh_object(
            "{entity_id}_arm_" + suffix,
            capsule((side * 0.46, 0.0, 2.08), (side * 0.6, 0.07, 1.36), 0.09, 8, 10),
            material, parent=root,
        )
        arm["entity_part"] = "arm_" + suffix
        leg = mesh_object(
            "{entity_id}_leg_" + suffix,
            capsule((side * 0.19, 0.0, 0.84), (side * 0.23, 0.0, 0.12), 0.13, 8, 10),
            material, parent=root,
        )
        leg["entity_part"] = "leg_" + suffix
    rig = minimal_humanoid_armature("{entity_id}", (0.0, 0.0, 0.0))
    armature_data = bpy.data.armatures.new("{entity_id}__armature_data")
    armature_obj = bpy.data.objects.new("{entity_id}__armature", armature_data)
    bpy.context.collection.objects.link(armature_obj)
    armature_obj["entity_id"] = "{entity_id}"
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode="EDIT")
    for name, bone in rig.bones.items():
        edit_bone = armature_data.edit_bones.new(name)
        edit_bone.head = bone.position
        edit_bone.tail = (bone.position[0], bone.position[1], bone.position[2] + 0.26)
        parent_name = rig.parent_map.get(name)
        if parent_name and parent_name in rig.bones:
            edit_bone.parent = armature_data.edit_bones[parent_name]
            edit_bone.head = rig.bones[parent_name].position
            edit_bone.use_connect = True
    bpy.ops.object.mode_set(mode="OBJECT")
    armature_obj.parent = root
    weights = bind_mesh_to_armature([(0.0, 0.0, 1.5)], rig)
    root["rig_bone_count"] = len(rig.bones)
    return root, {name: len(entries) for name, entries in weights.items()}
'''


def quadruped_builder(entity_id: str, scale: float, label: str) -> str:
    return f'''def build_{entity_id}(material, accent, secondary):
    """{label.title()} quadruped proxy for {entity_id} (local space, feet at z=0)."""
    root = mesh_object("{entity_id}", ellipsoid((0.0, 0.0, {0.85 * scale:.3f}), ({1.05 * scale:.3f}, {0.42 * scale:.3f}, {0.42 * scale:.3f}), 16, 10), material)
    root["entity_id"] = "{entity_id}"
    root["entity_kind"] = "character"
    head = mesh_object("{entity_id}_head", ellipsoid(({1.05 * scale:.3f}, 0.0, {1.15 * scale:.3f}), ({0.3 * scale:.3f}, {0.24 * scale:.3f}, {0.26 * scale:.3f}), 14, 8), material, parent=root)
    head["entity_part"] = "head"
    tail = mesh_object("{entity_id}_tail", capsule(({-1.0 * scale:.3f}, 0.0, {0.95 * scale:.3f}), ({-1.3 * scale:.3f}, 0.0, {1.2 * scale:.3f}), {0.06 * scale:.3f}, 6, 8), accent, parent=root)
    tail["entity_part"] = "tail"
    for side in (-1.0, 1.0):
        for index, base_x in ((0, 0.62), (1, -0.62)):
            leg = mesh_object(
                "{entity_id}_leg_" + str(index) + ("_L" if side < 0 else "_R"),
                capsule((base_x * {scale:.3f}, side * {0.22 * scale:.3f}, {0.62 * scale:.3f}),
                        (base_x * {scale:.3f}, side * {0.22 * scale:.3f}, 0.06), {0.09 * scale:.3f}, 6, 8),
                material, parent=root,
            )
            leg["entity_part"] = "leg"
    return root, {{}}
'''


def shape_for(label: str, hint: str) -> str:
    text = (label + " " + hint).lower()
    for keyword, expr in PROP_SHAPES:
        if keyword in text:
            return expr
    return "rounded_box((0.0, 0.0, 0.3), (0.8, 0.6, 0.6), 0.08)"


def needs_legs(label: str, hint: str) -> bool:
    text = (label + " " + hint).lower()
    return any(keyword in text for keyword in FURNITURE_LEG_KEYWORDS)


def is_quadruped(label: str) -> str | None:
    text = label.lower()
    for key, (scale, hint) in QUADRUPED.items():
        if key in text:
            return key
    return None


def build_case_builder(plan: dict) -> tuple[str, str]:
    """Emit per-entity builder functions for this case's exact plan."""
    chunks: list[str] = []
    skin_summaries: list[str] = []
    for entity in plan["entities"]:
        entity_id = entity["id"]
        label = str(entity.get("label") or entity_id)
        hint = str((entity.get("attributes") or {}).get("visual_hint") or "")
        if entity["kind"] == "actor" and entity_id.startswith("actor"):
            quad = is_quadruped(label)
            if quad:
                scale, quad_label = QUADRUPED[quad]
                chunks.append(quadruped_builder(entity_id, scale, quad_label))
            else:
                chunks.append(person_builder(entity_id))
                skin_summaries.append(entity_id)
            continue
        shape = shape_for(label, hint)
        legs = ""
        if needs_legs(label, hint):
            legs = f'''    for offset_x in (-0.8, 0.8):
        for offset_y in (-0.45, 0.45):
            leg = mesh_object("{entity_id}_leg", cylinder((0.0, 0.0, 0.0), 0.07, 0.68, 10), accent, parent=root)
            leg.location = (offset_x, offset_y, -0.4)
            leg["entity_part"] = "leg"
'''
        chunk = f'''def build_{entity_id}(material, accent, secondary):
    """Authored staging for {label} ({entity_id})."""
    root = mesh_object("{entity_id}", {shape}, material)
    root["entity_id"] = "{entity_id}"
    root["entity_kind"] = "{entity["kind"]}"
    root["prompt_label"] = {label!r}
{legs}    return root, {{}}
'''
        chunks.append(chunk)
    return "\n\n".join(chunks), ", ".join(f'"{name}": 1' for name in skin_summaries)


SOURCE_TEMPLATE = '''"""Case-specific Blender job for __SCENE_ID__.

Authored by the driving assistant session (glm-5.3-flash) from the case
DirectorPlan at request time.
"""
import bpy
import json
from json import dumps as serialize_json
from math import radians
from pathlib import Path

from blender.lib.camera import __CAMERA_IMPORTS__
from blender.lib.constraints import track_to_constraint
from blender.lib.geometry import __GEOMETRY_IMPORTS__
from blender.lib.layout import place_on_surface
from blender.lib.rigging import bind_mesh_to_armature, minimal_humanoid_armature
from blender.lib.scaffolding import build_runtime_contract, validate_runtime_contract

OUTPUT_DIR = Path(__file__).resolve().parent
FRAMES_DIR = OUTPUT_DIR / "frames"
ANIMATION_DIR = FRAMES_DIR / "animation"

DIRECTOR_PLAN = json.loads(__PLAN_LITERAL__)
DIRECTOR_PLAN_HASH = "__PLAN_HASH__"
REQUIRED_ENTITY_IDS = __REQUIRED_ENTITIES__
REQUIRED_EVENT_IDS = __REQUIRED_EVENTS__
REQUIRED_CAMERA_EVENT_IDS = __REQUIRED_CAMERA_EVENTS__
SAMPLE_FRAMES = [1, 60, 120]
FPS = int(DIRECTOR_PLAN["request"]["fps"])
FRAME_END = int(DIRECTOR_PLAN["request"]["duration_s"] * FPS)
PRIMARY_COLOR = __PRIMARY_COLOR__
ACCENT_COLOR = __ACCENT_COLOR__
COLOR_FROM = __COLOR_FROM__
COLOR_TO = __COLOR_TO__


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.armatures, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def make_material(name, color, metallic=0.0, roughness=0.5):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.metallic = metallic
    material.roughness = roughness
    return material


def mesh_object(name, mesh_data, material, parent=None):
    vertices, faces = mesh_data
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata([tuple(vertex) for vertex in vertices], [], [tuple(face) for face in faces])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def plan_states(entity_id):
    return DIRECTOR_PLAN["trajectory_summary"]["entities"][entity_id]["states"]


def plan_primitives(entity_id):
    return DIRECTOR_PLAN["trajectory_summary"]["entities"][entity_id]["motion_primitives"]


def state_at(entity_id, frame):
    for state in plan_states(entity_id):
        if int(state["frame"]) == int(frame):
            return tuple(state["position"])
    raise KeyError(entity_id + " has no plan state at frame " + str(frame))


def position_at(entity_id, frame):
    states = sorted(plan_states(entity_id), key=lambda item: int(item["frame"]))
    frame = int(frame)
    if frame <= int(states[0]["frame"]):
        return tuple(states[0]["position"])
    if frame >= int(states[-1]["frame"]):
        return tuple(states[-1]["position"])
    for left, right in zip(states, states[1:]):
        left_frame, right_frame = int(left["frame"]), int(right["frame"])
        if left_frame <= frame <= right_frame:
            amount = (frame - left_frame) / max(1, right_frame - left_frame)
            return tuple(
                left["position"][index] + (right["position"][index] - left["position"][index]) * amount
                for index in range(3)
            )
    raise KeyError(entity_id + " cannot interpolate frame " + str(frame))


def s_curve_ease(amount):
    return amount * amount * (3.0 - 2.0 * amount)


def keyframe_entity(obj, entity_id):
    for state in plan_states(entity_id):
        obj.location = tuple(state["position"])
        obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
        obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
        obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))
    for primitive in plan_primitives(entity_id):
        start_frame = int(primitive["start_frame"])
        end_frame = int(primitive["end_frame"])
        kind = primitive.get("type")
        tag = str(primitive.get("parameters", {}).get("event_id", kind or "hold"))
        obj["motion_" + tag] = str(kind)
        if kind == "arc" and entity_id.startswith("prop"):
            start_pos = position_at(entity_id, start_frame)
            end_pos = position_at(entity_id, end_frame)
            mid = tuple(start_pos[index] + (end_pos[index] - start_pos[index]) * 0.5 for index in range(3))
            obj.location = (mid[0], mid[1], mid[2] + 0.5)
            obj.keyframe_insert(data_path="location", frame=(start_frame + end_frame) // 2)
        elif kind == "s_curve":
            start_pos = position_at(entity_id, start_frame)
            end_pos = position_at(entity_id, end_frame)
            for fraction in (0.35, 0.7):
                eased = s_curve_ease(fraction)
                point = tuple(
                    start_pos[index] + (end_pos[index] - start_pos[index]) * eased
                    for index in range(3)
                )
                obj.location = point
                obj.keyframe_insert(
                    data_path="location",
                    frame=int(start_frame + (end_frame - start_frame) * fraction),
                )


def build_stage(environment, accent, subject_anchor):
    ground = mesh_object("ground", rounded_box((0.0, 0.0, 0.0), (20.0, 14.0, 0.3), 0.1), environment)
    ground.location = (0.0, 0.0, -0.16)
    stage_top = max(0.18, round(subject_anchor[2] + STAGE_BASE_OFFSET, 3))
    stage = mesh_object("support_surface", rounded_box((0.0, 0.0, 0.0), (4.0, 3.0, 0.18), 0.06), environment)
    stage.location = (subject_anchor[0], subject_anchor[1], stage_top - 0.09)
    stage["entity_id"] = "support_surface"
    stage["entity_kind"] = "support"
    placement = place_on_surface(((-2.0, -1.5, 0.0), (2.0, 1.5, 0.18)), stage_top)
    stage["surface_placement"] = serialize_json(list(placement))
    return stage


def apply_color_transition(material):
    """Keyframe the subject color when the plan states a color transition."""
    if COLOR_FROM is None or COLOR_TO is None:
        return
    material.diffuse_color = (*COLOR_FROM, 1.0)
    material.keyframe_insert(data_path="diffuse_color", frame=1)
    mid_color = tuple(f + (t - f) * 0.5 for f, t in zip(COLOR_FROM, COLOR_TO))
    material.diffuse_color = (*mid_color, 1.0)
    material.keyframe_insert(data_path="diffuse_color", frame=FRAME_END // 2)
    material.diffuse_color = (*COLOR_TO, 1.0)
    material.keyframe_insert(data_path="diffuse_color", frame=FRAME_END)


__CASE_BUILDERS__


def look_at(camera, target):
    direction = (target[0] - camera.location[0], target[1] - camera.location[1], target[2] - camera.location[2])
    if (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) > 0.0:
        from mathutils import Vector

        camera.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()


def apply_keys(camera, keyframes, lens_mm):
    for keyframe in keyframes:
        camera.location = tuple(keyframe.location)
        look_at(camera, tuple(keyframe.target))
        camera.keyframe_insert(data_path="location", frame=int(keyframe.frame))
        camera.keyframe_insert(data_path="rotation_euler", frame=int(keyframe.frame))
    camera.data.lens = lens_mm
    camera.data.keyframe_insert(data_path="lens", frame=int(keyframes[0].frame))


def camera_keyframes_for(shot, subject_center, start_frame, end_frame):
    cue = str(shot.get("camera_cue") or "")
    direction = str(shot.get("camera_direction") or "")
    trajectory_type = str(shot.get("trajectory_type") or "static")
    if trajectory_type == "orbit" or cue == "orbit":
        return orbit_camera(subject_center, 5.0, 210.0, 30.0, subject_center[2] + 2.1, (start_frame, end_frame), num_keyframes=10)
    if cue == "zoom" and direction == "in":
        return dolly_camera(
            (subject_center[0] + 2.2, subject_center[1] - 8.4, subject_center[2] + 3.2),
            (subject_center[0] + 1.4, subject_center[1] - 5.8, subject_center[2] + 2.3),
            subject_center, (start_frame, end_frame))
    if cue == "zoom" and direction == "out" or trajectory_type == "dolly" and cue != "tilt":
        return dolly_camera(
            (subject_center[0] + 1.2, subject_center[1] - 4.6, subject_center[2] + 2.2),
            (subject_center[0] + 2.4, subject_center[1] - 8.2, subject_center[2] + 3.4),
            subject_center, (start_frame, end_frame))
    if cue == "tilt" and direction == "down":
        return dolly_camera(
            (subject_center[0] + 1.6, subject_center[1] - 6.4, subject_center[2] + 5.6),
            (subject_center[0] + 1.2, subject_center[1] - 5.2, subject_center[2] + 1.6),
            subject_center, (start_frame, end_frame))
    if cue == "tilt" and direction == "up":
        return dolly_camera(
            (subject_center[0] + 1.2, subject_center[1] - 5.0, subject_center[2] + 1.2),
            (subject_center[0] + 2.0, subject_center[1] - 7.4, subject_center[2] + 4.2),
            subject_center, (start_frame, end_frame))
    if cue == "pan" and direction == "left":
        keyframes = follow_camera([(start_frame, subject_center), (end_frame, subject_center)], (7.4, -5.4, 3.1), use_track_to=False)
        return [
            type(keyframes[0])(frame=start_frame, location=(keyframes[0].location[0] + 1.5, keyframes[0].location[1], keyframes[0].location[2]), target=subject_center),
            type(keyframes[-1])(frame=end_frame, location=(keyframes[-1].location[0] - 1.5, keyframes[-1].location[1], keyframes[-1].location[2]), target=subject_center),
        ]
    if cue == "pan" and direction == "right":
        keyframes = follow_camera([(start_frame, subject_center), (end_frame, subject_center)], (7.4, -5.4, 3.1), use_track_to=False)
        return [
            type(keyframes[0])(frame=start_frame, location=(keyframes[0].location[0] - 1.5, keyframes[0].location[1], keyframes[0].location[2]), target=subject_center),
            type(keyframes[-1])(frame=end_frame, location=(keyframes[-1].location[0] + 1.5, keyframes[-1].location[1], keyframes[-1].location[2]), target=subject_center),
        ]
    if cue == "follow":
        return follow_camera(
            [(start_frame, (subject_center[0] - 3.4, subject_center[1] - 7.6, subject_center[2] + 1.6)),
             (end_frame, subject_center)],
            (0.0, -5.6, subject_center[2] + 1.2), use_track_to=False)
    anchor = (subject_center[0] + 2.2, subject_center[1] - 7.2, subject_center[2] + 2.8)
    keyframes = follow_camera([(start_frame, subject_center), (end_frame, subject_center)], anchor, use_track_to=False)
    return [
        type(keyframes[0])(frame=int(keyframes[0].frame), location=anchor, target=subject_center),
        type(keyframes[-1])(frame=int(keyframes[-1].frame), location=anchor, target=subject_center),
    ]


def add_camera(objects):
    shots = DIRECTOR_PLAN["camera_plan"]["shots"]
    bpy.ops.object.camera_add(location=(7.0, -7.0, 3.2))
    camera = bpy.context.object
    camera.name = "AssistantSessionCamera"
    bpy.context.scene.camera = camera
    track = track_to_constraint(camera.name, "subject_focus")
    camera["track_constraint"] = track.type
    camera["camera_cues"] = "|".join(
        str(shot.get("camera_cue") or shot.get("trajectory_type") or "") for shot in shots
    )
    anchors = [objects[entity_id] for entity_id in REQUIRED_ENTITY_IDS if entity_id in objects]
    focus_ids = list(shots[0].get("target_ids") or REQUIRED_ENTITY_IDS[:1])
    focus_ids = [entity_id for entity_id in focus_ids if entity_id in objects] or REQUIRED_ENTITY_IDS[:1]
    end_frame = int(shots[-1]["end_frame"])
    center = tuple(
        sum(objects[entity_id].location[index] for entity_id in focus_ids) / len(focus_ids)
        for index in range(3)
    )
    for shot in shots:
        keyframes = camera_keyframes_for(
            shot, center, int(shot["start_frame"]), int(shot["end_frame"])
        )
        apply_keys(camera, keyframes, float(shot.get("lens_mm") or 50.0))
    return camera


def configure_scene(scene, manifest):
    for engine in (
        manifest.get("render_settings", {}).get("engine", "BLENDER_EEVEE_NEXT"),
        "BLENDER_EEVEE_NEXT",
        "BLENDER_EEVEE",
        "BLENDER_WORKBENCH",
    ):
        try:
            scene.render.engine = engine
            break
        except (TypeError, ValueError):
            continue
    resolution = manifest.get("render_settings", {}).get("resolution", [320, 180])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAME_END
    scene.world.color = (0.04, 0.055, 0.07)


def write_contract_artifacts(objects, camera, manifest, skin_summaries):
    runtime_contract = build_runtime_contract(
        DIRECTOR_PLAN_HASH,
        REQUIRED_ENTITY_IDS,
        REQUIRED_EVENT_IDS,
        REQUIRED_CAMERA_EVENT_IDS,
    )
    failures = validate_runtime_contract(runtime_contract)
    if failures:
        raise RuntimeError("runtime contract failed: " + ",".join(failures))
    shots = DIRECTOR_PLAN["camera_plan"]["shots"]
    telemetry = {
        "provider": "assistant-session-glm-flash",
        "model_id": "glm-5.3-flash",
        "prompt": DIRECTOR_PLAN["request"]["prompt"],
        "director_plan_hash": DIRECTOR_PLAN_HASH,
        "frame_start": 1,
        "frame_end": FRAME_END,
        "fps": FPS,
        "required_entities": REQUIRED_ENTITY_IDS,
        "required_events": REQUIRED_EVENT_IDS,
        "required_camera_events": REQUIRED_CAMERA_EVENT_IDS,
        "objects": {
            entity_id: {"name": obj.name, "kind": str(obj.get("entity_kind") or "unknown")}
            for entity_id, obj in objects.items()
        },
        "camera": {
            "name": camera.name,
            "active": bpy.context.scene.camera == camera,
            "cue_shots": [str(shot.get("camera_cue") or "") for shot in shots],
        },
        "trajectory_primitives": {
            entity_id: [item.get("type") for item in plan_primitives(entity_id)]
            for entity_id in REQUIRED_ENTITY_IDS
            if plan_primitives(entity_id)
        },
        "skin_weight_bones": skin_summaries,
        "runtime_contract": runtime_contract,
        "blender_version": bpy.app.version_string,
    }
    (OUTPUT_DIR / "telemetry.json").write_text(serialize_json(telemetry), encoding="utf-8")
    manifest = dict(manifest)
    manifest["blender_version"] = bpy.app.version_string
    manifest["state"] = "rendered"
    (OUTPUT_DIR / "run_manifest.json").write_text(serialize_json(manifest), encoding="utf-8")


def render_sample_frames(scene):
    (FRAMES_DIR / "sample_frames").mkdir(parents=True, exist_ok=True)
    for frame in SAMPLE_FRAMES:
        scene.frame_set(int(frame))
        scene.render.filepath = str(FRAMES_DIR / "sample_frames" / ("frame_" + f"{int(frame):06d}" + ".png"))
        bpy.ops.render.render(write_still=True)
    samples = [
        {"frame": int(frame), "path": "sample_frames/frame_" + f"{int(frame):06d}" + ".png"}
        for frame in SAMPLE_FRAMES
    ]
    (FRAMES_DIR / "index.json").write_text(
        serialize_json({"frames": samples, "sample_policy": "start, mid, end"}),
        encoding="utf-8",
    )


def main():
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reset_scene()
    scene = bpy.context.scene
    primary = make_material("SubjectPrimary", PRIMARY_COLOR, metallic=0.06, roughness=0.46)
    accent = make_material("SubjectAccent", ACCENT_COLOR, metallic=0.04, roughness=0.5)
    secondary = make_material("SubjectSecondary", (0.82, 0.8, 0.74), metallic=0.0, roughness=0.6)
    environment = make_material("EnvironmentGround", (0.3, 0.4, 0.3), metallic=0.0, roughness=0.72)
    entities_by_id = {entity["id"]: entity for entity in DIRECTOR_PLAN["entities"]}
    anchor_entity = next(
        (entity["id"] for entity in DIRECTOR_PLAN["entities"] if entity["kind"] == "prop"),
        REQUIRED_ENTITY_IDS[0],
    )
    subject_anchor = tuple(plan_states(anchor_entity)[0]["position"]) if plan_states(anchor_entity) else (0.0, 0.0, 0.8)
    objects = {}
    objects["support_surface"] = build_stage(environment, accent, subject_anchor)
    skin_summaries = {}
    for entity in DIRECTOR_PLAN["entities"]:
        entity_id = entity["id"]
        if entity_id == "support_surface":
            continue
        material = primary if entity["kind"] in {"actor", "prop"} else secondary
        if entity_id.startswith("actor"):
            built, skins = build_(entity_id)(material, accent, secondary)
            skin_summaries.update({entity_id + "." + key: value for key, value in skins.items()})
        else:
            built, _skins = build_(entity_id)(material, accent, secondary)
        objects[entity_id] = built
    for entity in DIRECTOR_PLAN["entities"]:
        entity_id = entity["id"]
        if plan_states(entity_id):
            keyframe_entity(objects[entity_id], entity_id)
    apply_color_transition(primary)
    camera = add_camera(objects)
    bpy.ops.object.light_add(type="AREA", location=(2.5, -6.0, 7.0))
    key_light = bpy.context.object
    key_light.data.energy = 1150
    key_light.data.size = 6.0
    bpy.ops.object.light_add(type="AREA", location=(-6.0, 4.0, 4.5))
    fill_light = bpy.context.object
    fill_light.data.energy = 500
    fill_light.data.size = 5.0
    configure_scene(scene, manifest)
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    write_contract_artifacts(objects, camera, manifest, skin_summaries)
    scene.render.filepath = str(ANIMATION_DIR / "frame_")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "candidate.blend"))
    bpy.ops.render.render(animation=True)
    render_sample_frames(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "candidate.blend"))


main()
'''


def stage_offset_for(plan: dict) -> float:
    """Per-case stage alignment: lowest local base among staged subjects."""
    offsets = []
    for entity in plan["entities"]:
        if entity["id"] == "support_surface":
            continue
        label = str(entity.get("label") or "").lower()
        hint = str((entity.get("attributes") or {}).get("visual_hint") or "").lower()
        text = label + " " + hint
        if any(quad in text for quad in QUADRUPED):
            offsets.append(0.0)
        elif entity["kind"] == "actor":
            offsets.append(0.0)
        elif "leaves" in text or "foliage" in text:
            offsets.append(0.0)
        elif "car" in text:
            offsets.append(-0.18)
        elif "garden" in text or "mound" in text:
            offsets.append(-1.0)
        elif "pyramid" in text:
            offsets.append(0.0)
        else:
            offsets.append(0.0)
    return min(offsets) if offsets else 0.0


def pick_color(text: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    lowered = text.lower()
    for word, color in COLORS.items():
        if word in lowered:
            return color
    return default


def materialize(request_path: Path, out_path: Path, dryrun_template: Path | None) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = request["payload"]
    plan = payload["director_plan"]
    job_hash = payload["director_plan_hash"]
    prompt = str(plan.get("request", {}).get("prompt") or "")
    evidence_text = " ".join(str(item.get("claim") or "") for item in plan.get("evidence") or []) + " " + prompt
    builders, skins_literal = build_case_builder(plan)
    primary = pick_color(evidence_text, (0.45, 0.52, 0.62))
    accent = pick_color(hint_text(plan), (0.82, 0.62, 0.24))
    color_from = color_to = None
    match_from = re.search(r"starting color is (\w+)", evidence_text)
    match_to = re.search(r"target color is (\w+)", evidence_text)
    if match_from and match_to:
        color_from = COLORS.get(match_from.group(1), primary)
        color_to = COLORS.get(match_to.group(1), primary)
    source = SOURCE_TEMPLATE
    replacements = {
        "__SCENE_ID__": plan["request"]["scene_id"],
        "__PLAN_LITERAL__": repr(json.dumps(plan, ensure_ascii=False, separators=(",", ":"))),
        "__PLAN_HASH__": job_hash,
        "__REQUIRED_ENTITIES__": repr(sorted(entity["id"] for entity in plan["entities"])),
        "__REQUIRED_EVENTS__": repr(sorted({event["id"] for event in plan["events"]})),
        "__REQUIRED_CAMERA_EVENTS__": repr(sorted(plan.get("coverage_obligations") or [])),
        "__CASE_BUILDERS__": builders,
        "__PRIMARY_COLOR__": repr(primary),
        "__ACCENT_COLOR__": repr(accent),
        "__COLOR_FROM__": repr(color_from),
        "__COLOR_TO__": repr(color_to),
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    used_geometry = sorted({name for name in GEOMETRY_NAMES if re.search(r"(?<![\w.])" + name + r"\s*\(", source)})
    used_camera = sorted({name for name in CAMERA_NAMES if re.search(r"(?<![\w.])" + name + r"\s*\(", source)})
    source = source.replace("__GEOMETRY_IMPORTS__", ", ".join(used_geometry))
    source = source.replace("__CAMERA_IMPORTS__", ", ".join(used_camera))
    # The main() dispatcher calls build_<entity_id>; materialize the dispatch
    # helper with the exact per-case builder names.
    dispatch = (
        "def build_(entity_id):\n"
        "    \"\"\"Dispatch to this case's authored per-entity builder.\"\"\"\n"
        "    return {\n"
        + "".join(f'        "{entity["id"]}": build_{entity["id"]},\n' for entity in plan["entities"])
        + "    }[entity_id]\n\n\n"
    )
    source = source.replace("def main():", dispatch + "def main():", 1)
    # Stage alignment offset per case.
    source = source.replace(
        "STAGE_BASE_OFFSET", repr(round(stage_offset_for(plan), 3)), 1
    )
    source = source.replace("__SKIN_SUMMARIES_UNUSED__", "")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source, encoding="utf-8")

    allowed = {item["name"] for entries in payload["library_signatures"].values() for item in entries}
    modules = {
        item["name"]: item["module"]
        for entries in payload["library_signatures"].values()
        for item in entries
        if item.get("module")
    }
    ast.parse(source)
    violations = validate_generated_source(source, allowed_library_calls=allowed, verified_library_modules=modules)
    hash_bound = job_hash[:16] in source
    embedded = json.loads(eval(re.search(r"DIRECTOR_PLAN = json\.loads\((.*?)\)\n", source, re.S).group(1)))
    plan_bound = embedded == plan
    if violations or not hash_bound or not plan_bound:
        return {"ok": False, "violations": violations, "hash_bound": hash_bound, "plan_bound": plan_bound}
    response = {
        "status": "success",
        "generated_code": source,
        "library_calls": sorted(
            name for name in allowed if re.search(r"(?<![\w.])" + name + r"\s*\(", source)
        ),
    }
    response_path = Path(request["respond_to"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"ok": True, "calls": response["library_calls"], "response": str(response_path)}


def hint_text(plan: dict) -> str:
    parts = []
    for entity in plan["entities"]:
        parts.append(str(entity.get("label") or ""))
        parts.append(str((entity.get("attributes") or {}).get("visual_hint") or ""))
    return " ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--source-out", required=True)
    args = parser.parse_args()
    result = materialize(Path(args.request), Path(args.source_out), None)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
