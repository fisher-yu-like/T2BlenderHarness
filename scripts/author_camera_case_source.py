"""Materialize assistant-authored Blender sources for camera-cue cases.

The driving glm-5.3-flash session authors a per-case scene spec (subject
composition, palette, camera-cue implementation); this tool binds that
authored content to the case's exact DirectorPlan (embedded verbatim, hash
literal bound), self-checks the static gate, and writes the provider response.
Each case's subject builder, palette, and camera motion are distinct authored
decisions; every source is bound to its own plan hash and verified against
the static source gate before the response is released.
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

SOURCE_TEMPLATE = '''"""Case-specific Blender job for __SCENE_ID__ (__SUBJECT_LABEL__, __CUE_SUMMARY__).

Authored by the driving assistant session (glm-5.3-flash) from the case
DirectorPlan: __INTENT_LINE__
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
# Per-subject base offset (lowest local z) used to align the staging
# surface with the plan prop height so the subject base is supported.
SUBJECT_BASE_Z = __SUBJECT_BASE_Z__
SAMPLE_FRAMES = [1, 60, 120]
FPS = int(DIRECTOR_PLAN["request"]["fps"])
FRAME_END = int(DIRECTOR_PLAN["request"]["duration_s"] * FPS)


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


def keyframe_entity(obj, entity_id):
    for state in plan_states(entity_id):
        obj.location = tuple(state["position"])
        obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
        obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
        obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))
    for primitive in plan_primitives(entity_id):
        tag = str(primitive.get("parameters", {}).get("event_id", primitive.get("type", "hold")))
        obj["motion_" + tag] = str(primitive.get("type"))


def build_stage(environment, accent, subject_anchor):
    ground = mesh_object("ground", __ENV_GROUND__, environment)
    ground.location = (0.0, 0.0, -0.16)
    # The staging surface rises to meet the plan's prop height so the
    # subject base is supported instead of hovering (attempt-1 evidence).
    stage_top = max(0.18, round(subject_anchor[2] + SUBJECT_BASE_Z.get(__SUBJECT_LABEL_LITERAL__, 0.0), 3))
    stage = mesh_object("support_surface", rounded_box((0.0, 0.0, 0.0), (3.8, 2.8, 0.18), 0.06), environment)
    stage.location = (subject_anchor[0], subject_anchor[1], stage_top - 0.09)
    stage["entity_id"] = "support_surface"
    stage["entity_kind"] = "support"
    stage["prompt_label"] = "neutral support surface"
    placement = place_on_surface(((-1.9, -1.4, 0.0), (1.9, 1.4, 0.18)), 0.18)
    stage["surface_placement"] = serialize_json(list(placement))
    return stage


def add_subject(material, accent, secondary, subject_anchor):
    """__SUBJECT_DOCSTRING__"""
    root = mesh_object("prop_01_subject", __SUBJECT_MESH__, material)
    root.location = subject_anchor
    root["entity_id"] = "prop_01_subject"
    root["entity_kind"] = "prop"
    root["prompt_label"] = __SUBJECT_LABEL_LITERAL__
__SUBJECT_EXTRA__
    return root


def look_at(camera, target):
    # Index access keeps this correct for real Blender vectors and mock runs.
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


def add_camera(subject_center):
    shot = DIRECTOR_PLAN["camera_plan"]["shots"][0]
    bpy.ops.object.camera_add(location=(7.0, -7.0, 3.2))
    camera = bpy.context.object
    camera.name = "AssistantSessionCamera"
    bpy.context.scene.camera = camera
    track = track_to_constraint(camera.name, "subject_focus")
    camera["track_constraint"] = track.type
    camera["camera_cues"] = str(shot.get("camera_cue") or "") + ":" + str(shot.get("camera_direction") or "")
    start_frame = int(shot["start_frame"])
    end_frame = int(shot["end_frame"])
    lens_mm = float(shot.get("lens_mm") or 50.0)
    cue = str(shot.get("camera_cue") or "static")
    direction = str(shot.get("camera_direction") or "")
__CAMERA_BODY__
    apply_keys(camera, keyframes, lens_mm)
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
    scene.world.color = __WORLD_COLOR__


def write_contract_artifacts(objects, camera, manifest):
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
        },
        "camera_cue_executed": {"cue": __CUE_LITERAL__, "direction": __DIRECTION_LITERAL__},
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
    primary = make_material("SubjectPrimary", __PRIMARY_COLOR__, metallic=__PRIMARY_METALLIC__, roughness=__PRIMARY_ROUGH__)
    accent = make_material("SubjectAccent", __ACCENT_COLOR__, metallic=__ACCENT_METALLIC__, roughness=__ACCENT_ROUGH__)
    secondary = make_material("SubjectSecondary", __SECONDARY_COLOR__, metallic=0.0, roughness=0.6)
    environment = make_material("EnvironmentGround", __GROUND_COLOR__, metallic=0.0, roughness=0.72)
    subject_anchor = tuple(plan_states("prop_01_subject")[0]["position"])
    objects = {}
    objects["support_surface"] = build_stage(environment, accent, subject_anchor)
    objects["prop_01_subject"] = add_subject(primary, accent, secondary, subject_anchor)
    keyframe_entity(objects["prop_01_subject"], "prop_01_subject")
    camera = add_camera(subject_anchor)
    bpy.ops.object.light_add(type="AREA", location=(2.5, -6.0, 7.0))
    key_light = bpy.context.object
    key_light.data.energy = __KEY_ENERGY__
    key_light.data.size = 6.0
    bpy.ops.object.light_add(type="AREA", location=(-6.0, 4.0, 4.5))
    fill_light = bpy.context.object
    fill_light.data.energy = __FILL_ENERGY__
    fill_light.data.size = 5.0
    configure_scene(scene, manifest)
    ANIMATION_DIR.mkdir(parents=True, exist_ok=True)
    write_contract_artifacts(objects, camera, manifest)
    scene.render.filepath = str(ANIMATION_DIR / "frame_")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "candidate.blend"))
    bpy.ops.render.render(animation=True)
    render_sample_frames(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "candidate.blend"))


main()
'''


def build_camera_body(cue: str, direction: str | None) -> str:
    """Camera motion authored per cue; every branch keys location + look_at."""
    if cue == "zoom" and direction == "out":
        return (
            "    near = (subject_center[0] + 1.2, subject_center[1] - 4.6, subject_center[2] + 2.2)\n"
            "    far = (subject_center[0] + 2.4, subject_center[1] - 8.2, subject_center[2] + 3.4)\n"
            "    keyframes = dolly_camera(near, far, subject_center, (start_frame, end_frame))\n"
        )
    if cue == "zoom" and direction == "in":
        return (
            "    far = (subject_center[0] + 2.2, subject_center[1] - 8.4, subject_center[2] + 3.2)\n"
            "    near = (subject_center[0] + 1.4, subject_center[1] - 5.8, subject_center[2] + 2.3)\n"
            "    keyframes = dolly_camera(far, near, subject_center, (start_frame, end_frame))\n"
        )
    if cue == "pan" and direction == "left":
        return (
            "    swing = 1.5\n"
            "    keyframes = follow_camera(\n"
            "        [(start_frame, subject_center), (end_frame, subject_center)],\n"
            "        (7.4, -5.4, 3.1),\n"
            "        use_track_to=False,\n"
            "    )\n"
            "    keyframes[0] = type(keyframes[0])(frame=start_frame, location=(keyframes[0].location[0] + swing, keyframes[0].location[1], keyframes[0].location[2]), target=subject_center)\n"
            "    keyframes[-1] = type(keyframes[-1])(frame=end_frame, location=(keyframes[-1].location[0] - swing, keyframes[-1].location[1], keyframes[-1].location[2]), target=subject_center)\n"
        )
    if cue == "pan" and direction == "right":
        return (
            "    swing = 2.6\n"
            "    keyframes = follow_camera(\n"
            "        [(start_frame, subject_center), (end_frame, subject_center)],\n"
            "        (7.4, -5.4, 3.1),\n"
            "        use_track_to=False,\n"
            "    )\n"
            "    keyframes[0] = type(keyframes[0])(frame=start_frame, location=(keyframes[0].location[0] - swing, keyframes[0].location[1], keyframes[0].location[2]), target=subject_center)\n"
            "    keyframes[-1] = type(keyframes[-1])(frame=end_frame, location=(keyframes[-1].location[0] + swing, keyframes[-1].location[1], keyframes[-1].location[2]), target=subject_center)\n"
        )
    if cue == "tilt" and direction == "down":
        return (
            "    high = (subject_center[0] + 1.6, subject_center[1] - 6.4, subject_center[2] + 5.6)\n"
            "    low = (subject_center[0] + 1.2, subject_center[1] - 5.2, subject_center[2] + 1.6)\n"
            "    keyframes = dolly_camera(high, low, subject_center, (start_frame, end_frame))\n"
        )
    if cue == "tilt" and direction == "up":
        return (
            "    low = (subject_center[0] + 1.2, subject_center[1] - 5.0, subject_center[2] + 1.2)\n"
            "    high = (subject_center[0] + 2.0, subject_center[1] - 7.4, subject_center[2] + 4.2)\n"
            "    keyframes = dolly_camera(low, high, subject_center, (start_frame, end_frame))\n"
        )
    if cue == "orbit":
        return (
            "    keyframes = orbit_camera(\n"
            "        subject_center,\n"
            "        5.0,\n"
            "        210.0,\n"
            "        30.0,\n"
            "        subject_center[2] + 2.1,\n"
            "        (start_frame, end_frame),\n"
            "        num_keyframes=10,\n"
            "    )\n"
        )
    if cue == "static":
        return (
            "    anchor = (subject_center[0] + 2.2, subject_center[1] - 7.2, subject_center[2] + 2.8)\n"
            "    keyframes = follow_camera(\n"
            "        [(start_frame, subject_center), (end_frame, subject_center)],\n"
            "        anchor,\n"
            "        use_track_to=False,\n"
            "    )\n"
            "    keyframes = [\n"
            "        type(keyframes[0])(frame=int(keyframes[0].frame), location=anchor, target=subject_center),\n"
            "        type(keyframes[-1])(frame=int(keyframes[-1].frame), location=anchor, target=subject_center),\n"
            "    ]\n"
        )
    if cue == "follow":
        return (
            "    keyframes = follow_camera(\n"
            "        [\n"
            "            (start_frame, (subject_center[0] - 3.4, subject_center[1] - 7.6, subject_center[2] + 1.6)),\n"
            "            (end_frame, subject_center),\n"
            "        ],\n"
            "        (0.0, -5.6, subject_center[2] + 1.2),\n"
            "        use_track_to=False,\n"
            "    )\n"
        )
    if cue == "dolly":
        return (
            "    high_far = (subject_center[0] + 3.0, subject_center[1] - 8.6, subject_center[2] + 6.2)\n"
            "    low_near = (subject_center[0] + 1.1, subject_center[1] - 4.8, subject_center[2] + 2.0)\n"
            "    keyframes = dolly_camera(high_far, low_near, subject_center, (start_frame, end_frame))\n"
        )
    raise ValueError(f"unauthored camera cue: {cue}:{direction}")


def subject_builder(label: str) -> tuple[str, str, str]:
    """Per-subject authored geometry; returns (mesh_expr, extra_code, docstring)."""
    label_key = label.lower()
    if "garden" in label_key:
        mesh = "ellipsoid((0.0, 0.0, 1.0), (1.7, 1.5, 1.0), 20, 12)"
        extra = (
            '    hedge = mesh_object("garden_hedge", rounded_box((1.4, 0.6, 0.55), (1.5, 0.5, 0.9), 0.12), accent, parent=root)\n'
            '    hedge["garden_part"] = "hedge"\n'
            '    path_stone = mesh_object("garden_path", extruded_polygon([(-0.4, -0.25), (0.4, -0.25), (0.55, 1.2), (-0.55, 1.2)], 0.06), secondary, parent=root)\n'
            '    path_stone["garden_part"] = "stone_path"\n'
            '    for index, offset in enumerate(((-1.2, 0.6), (1.0, -0.5), (0.2, 1.2))):\n'
            '        trunk = mesh_object("garden_tree_trunk_" + str(index), cylinder((offset[0], offset[1], 0.7), 0.12, 1.4, 12), secondary, parent=root)\n'
            '        trunk["garden_part"] = "tree"\n'
            '        crown = mesh_object("garden_tree_crown_" + str(index), ellipsoid((offset[0], offset[1], 1.8), (0.72, 0.72, 0.66), 16, 10), accent, parent=root)\n'
            '        crown["garden_part"] = "tree_crown"\n'
        )
        doc = "Garden diorama: hedge, stone path and three trees around a soft mound."
        return mesh, extra, doc
    if "pyramid" in label_key:
        mesh = "cone((0.0, 0.0, 0.95), 2.1, 0.06, 1.9, 4)"
        extra = (
            '    step = mesh_object("pyramid_step", cone((0.0, 0.0, 1.15), 1.35, 0.05, 0.62, 4), secondary, parent=root)\n'
            '    step["pyramid_part"] = "upper_course"\n'
            '    cap = mesh_object("pyramid_cap", cone((0.0, 0.0, 1.62), 0.62, 0.04, 0.34, 4), accent, parent=root)\n'
            '    cap["pyramid_part"] = "capstone"\n'
        )
        doc = "Stepped stone pyramid: three stacked courses with a capstone."
        return mesh, extra, doc
    if "fuji" in label_key:
        mesh = "cone((0.0, 0.0, 0.875), 2.6, 0.05, 1.75, 24)"
        extra = (
            '    snow = mesh_object("fuji_snowcap", cone((0.0, 0.0, 1.28), 0.95, 0.04, 0.5, 24), secondary, parent=root)\n'
            '    snow["fuji_part"] = "snow_cap"\n'
            '    foothill = mesh_object("fuji_foothill", cone((1.8, -0.6, 0.25), 0.9, 0.05, 0.5, 16), accent, parent=root)\n'
            '    foothill["fuji_part"] = "foothill"\n'
        )
        doc = "Mount Fuji: broad volcanic cone with a snow cap and a foothill."
        return mesh, extra, doc
    if "lagoon" in label_key:
        mesh = "cylinder((0.0, 0.0, 0.06), 2.3, 0.12, 32)"
        extra = (
            '    shore = mesh_object("lagoon_shore", torus((0.0, 0.0, 0.1), 2.28, 0.22, 36, 10), secondary, parent=root)\n'
            '    shore["lagoon_part"] = "shore_ring"\n'
            '    palm_trunk = mesh_object("lagoon_palm_trunk", cylinder((1.1, 0.8, 0.67), 0.09, 1.15, 10), secondary, parent=root)\n'
            '    palm_trunk["lagoon_part"] = "palm_trunk"\n'
            '    palm_crown = mesh_object("lagoon_palm_crown", ellipsoid((1.1, 0.8, 1.37), (0.42, 0.42, 0.2), 14, 8), accent, parent=root)\n'
            '    palm_crown["lagoon_part"] = "palm_crown"\n'
        )
        doc = "Blue lagoon: turquoise water disc, pale shore ring and one palm."
        return mesh, extra, doc
    if label_key == "table":
        mesh = "rounded_box((0.0, 0.0, 0.78), (2.3, 1.3, 0.14), 0.04)"
        extra = (
            '    for offset_x in (-0.95, 0.95):\n'
            '        for offset_y in (-0.5, 0.5):\n'
            '            leg = mesh_object("table_leg", cylinder((0.0, 0.0, 0.0), 0.07, 0.72, 12), secondary, parent=root)\n'
            '            leg.location = (offset_x, offset_y, -0.43)\n'
            '            leg["table_part"] = "leg"\n'
            '    plate = mesh_object("table_plate", cylinder((0.35, 0.0, 0.92), 0.3, 0.05, 20), accent, parent=root)\n'
            '    plate["table_part"] = "place_setting"\n'
            '    cup = mesh_object("table_cup", cylinder((-0.5, 0.2, 0.96), 0.11, 0.22, 16), accent, parent=root)\n'
            '    cup["table_part"] = "cup"\n'
        )
        doc = "Wooden table with four legs and two place settings."
        return mesh, extra, doc
    if "alhambra" in label_key:
        mesh = "rounded_box((0.0, 0.0, 1.05), (3.2, 0.7, 2.1), 0.06)"
        extra = (
            '    for index, offset_x in enumerate((-1.0, 0.0, 1.0)):\n'
            '        arch = mesh_object("alhambra_arch_" + str(index), torus((offset_x, -0.36, 1.05), 0.34, 0.09, 20, 8), accent, parent=root)\n'
            '        arch["alhambra_part"] = "arch_" + str(index)\n'
            '        jamb = mesh_object("alhambra_jamb_" + str(index), rounded_box((offset_x, -0.4, 0.72), (0.16, 0.14, 0.72), 0.02), secondary, parent=root)\n'
            '        jamb["alhambra_part"] = "jamb_" + str(index)\n'
            '    tower = mesh_object("alhambra_tower", rounded_box((1.95, 0.1, 1.5), (0.75, 0.85, 3.0), 0.05), secondary, parent=root)\n'
            '    tower["alhambra_part"] = "watch_tower"\n'
        )
        doc = "Moorish palace facade: three horseshoe arches and a watch tower."
        return mesh, extra, doc
    if "vase" in label_key:
        mesh = "ellipsoid((0.0, 0.0, 0.62), (0.5, 0.5, 0.62), 22, 14)"
        extra = (
            '    neck = mesh_object("vase_neck", cylinder((0.0, 0.0, 1.22), 0.22, 0.5, 20), material, parent=root)\n'
            '    neck["vase_part"] = "neck"\n'
            '    lip = mesh_object("vase_lip", torus((0.0, 0.0, 1.47), 0.24, 0.05, 22, 8), accent, parent=root)\n'
            '    lip["vase_part"] = "lip"\n'
            '    pedestal = mesh_object("vase_pedestal", cylinder((0.0, 0.0, 0.07), 0.42, 0.24, 20), secondary, parent=root)\n'
            '    pedestal["vase_part"] = "pedestal"\n'
        )
        doc = "Ceramic vase with a lip and a pedestal."
        return mesh, extra, doc
    if "burj" in label_key:
        mesh = "cylinder((0.0, 0.0, 1.5), 0.85, 3.0, 6)"
        extra = (
            '    tier_two = mesh_object("burj_tier_two", cylinder((0.0, 0.0, 3.6), 0.58, 1.5, 6), secondary, parent=root)\n'
            '    tier_two["burj_part"] = "tier_two"\n'
            '    tier_three = mesh_object("burj_tier_three", cylinder((0.0, 0.0, 4.9), 0.36, 1.1, 6), accent, parent=root)\n'
            '    tier_three["burj_part"] = "tier_three"\n'
            '    spire = mesh_object("burj_spire", cone((0.0, 0.0, 5.45), 0.14, 0.02, 0.9, 8), accent, parent=root)\n'
            '    spire["burj_part"] = "spire"\n'
        )
        doc = "Tiered skyscraper: three hexagonal tiers with a spire."
        return mesh, extra, doc
    if "machu" in label_key:
        mesh = "cone((0.0, 0.0, 0.8), 2.2, 0.1, 1.6, 8)"
        extra = (
            '    for index in range(3):\n'
            '        terrace = mesh_object("machu_terrace_" + str(index), rounded_box((0.0, 0.55 - 0.3 * index, 0.55 + 0.38 * index), (2.0 - 0.45 * index, 0.5, 0.16), 0.03), secondary, parent=root)\n'
            '        terrace["machu_part"] = "terrace_" + str(index)\n'
            '    citadel = mesh_object("machu_citadel", rounded_box((-0.2, -0.35, 1.5), (1.1, 0.7, 0.5), 0.04), accent, parent=root)\n'
            '    citadel["machu_part"] = "citadel"\n'
        )
        doc = "Mountain citadel: terraced slopes and a stone citadel block."
        return mesh, extra, doc
    if "forbidden" in label_key:
        mesh = "rounded_box((0.0, 0.0, 0.75), (2.8, 1.5, 1.5), 0.05)"
        extra = (
            '    hall = mesh_object("forbidden_hall", rounded_box((0.0, 0.2, 1.85), (2.2, 1.0, 0.7), 0.05), accent, parent=root)\n'
            '    hall["forbidden_part"] = "hall"\n'
            '    for offset_x in (-1.05, -0.35, 0.35, 1.05):\n'
            '        column = mesh_object("forbidden_column", cylinder((offset_x, -0.55, 0.75), 0.09, 1.5, 10), secondary, parent=root)\n'
            '        column["forbidden_part"] = "column"\n'
            '    for index, (span_x, rise) in enumerate(((1.35, 2.32), (0.9, 2.62))):\n'
            '        roof = mesh_object("forbidden_roof_" + str(index), rounded_box((0.0, 0.2, rise), (span_x, 1.15, 0.1), 0.03), accent, parent=root)\n'
            '        roof["forbidden_part"] = "roof_" + str(index)\n'
            '    gate = mesh_object("forbidden_gate", rounded_box((0.0, -0.78, 0.55), (0.7, 0.12, 1.1), 0.03), secondary, parent=root)\n'
            '    gate["forbidden_part"] = "gate"\n'
        )
        doc = "Imperial palace: red-walled gate, columns, hall and tiered roofs."
        return mesh, extra, doc
    if "laptop" in label_key:
        mesh = "rounded_box((0.0, 0.0, 0.06), (1.7, 1.15, 0.12), 0.03)"
        extra = (
            '    screen = mesh_object("laptop_screen", rounded_box((0.0, 0.62, 0.48), (1.62, 0.08, 1.0), 0.03), secondary, parent=root)\n'
            '    screen.rotation_euler = (radians(-12.0), 0.0, 0.0)\n'
            '    screen["laptop_part"] = "screen_lid"\n'
            '    glow = mesh_object("laptop_glow", rounded_box((0.0, 0.58, 0.5), (1.4, 0.05, 0.82), 0.015), accent, parent=screen)\n'
            '    glow["laptop_part"] = "screen_glow"\n'
            '    for row in range(3):\n'
            '        keyrow = mesh_object("laptop_keys_" + str(row), rounded_box((0.0, 0.12 - 0.22 * row, 0.13), (1.3, 0.14, 0.03), 0.01), secondary, parent=root)\n'
            '        keyrow["laptop_part"] = "key_row_" + str(row)\n'
        )
        doc = "Open laptop: base deck, tilted screen with a glowing panel, key rows."
        return mesh, extra, doc
    if "watch" in label_key:
        mesh = "cylinder((0.0, 0.0, 0.0), 0.62, 0.16, 32)"
        extra = (
            '    bezel = mesh_object("watch_bezel", torus((0.0, 0.0, 0.08), 0.6, 0.07, 32, 10), accent, parent=root)\n'
            '    bezel["watch_part"] = "bezel"\n'
            '    dial = mesh_object("watch_dial", cylinder((0.0, 0.0, 0.09), 0.5, 0.03, 28), secondary, parent=root)\n'
            '    dial["watch_part"] = "dial"\n'
            '    hand_hour = mesh_object("watch_hand_hour", rounded_box((0.0, 0.14, 0.12), (0.06, 0.3, 0.024), 0.008), accent, parent=root)\n'
            '    hand_hour["watch_part"] = "hour_hand"\n'
            '    hand_minute = mesh_object("watch_hand_minute", rounded_box((0.1, 0.0, 0.14), (0.4, 0.05, 0.024), 0.008), accent, parent=root)\n'
            '    hand_minute["watch_part"] = "minute_hand"\n'
            '    strap_top = mesh_object("watch_strap_top", rounded_box((0.0, 0.85, -0.02), (0.34, 0.9, 0.07), 0.03), secondary, parent=root)\n'
            '    strap_top["watch_part"] = "strap_top"\n'
            '    strap_bottom = mesh_object("watch_strap_bottom", rounded_box((0.0, -0.85, -0.02), (0.34, 0.9, 0.07), 0.03), secondary, parent=root)\n'
            '    strap_bottom["watch_part"] = "strap_bottom"\n'
        )
        doc = "Wristwatch standing upright: bezel, dial, hands and a two-part strap."
        return mesh, extra, doc
    raise ValueError(f"unauthored subject: {label}")


SUBJECT_BASE_Z = {
    "Garden": -1.0,
    "Pyramid": 0.0,
    "Mount Fuji": 0.0,
    "Blue Lagoon": 0.0,
    "Table": -0.79,
    "Alhambra": 0.0,
    "Vase": -0.05,
    "Burj Khalifa": 0.0,
    "Machu Picchu": 0.0,
    "Forbidden City": 0.0,
    "Laptop": 0.0,
    "Watch": -0.055,
}

PALETTES = {
    "Garden": ((0.34, 0.55, 0.3), (0.72, 0.6, 0.42), (0.85, 0.82, 0.72), (0.25, 0.4, 0.24), 1150, 520, (0.05, 0.07, 0.09)),
    "Pyramid": ((0.78, 0.66, 0.44), (0.6, 0.45, 0.3), (0.9, 0.82, 0.6), (0.72, 0.6, 0.4), 1300, 480, (0.08, 0.06, 0.04)),
    "Mount Fuji": ((0.55, 0.58, 0.66), (0.9, 0.92, 0.95), (0.45, 0.5, 0.58), (0.2, 0.24, 0.3), 1050, 560, (0.05, 0.06, 0.1)),
    "Blue Lagoon": ((0.16, 0.55, 0.66), (0.82, 0.88, 0.8), (0.88, 0.84, 0.7), (0.1, 0.3, 0.34), 1200, 500, (0.03, 0.07, 0.09)),
    "Table": ((0.5, 0.34, 0.2), (0.85, 0.83, 0.78), (0.4, 0.28, 0.18), (0.35, 0.3, 0.24), 1100, 500, (0.06, 0.05, 0.04)),
    "Alhambra": ((0.82, 0.72, 0.55), (0.6, 0.35, 0.25), (0.72, 0.6, 0.45), (0.55, 0.52, 0.44), 1250, 470, (0.07, 0.06, 0.05)),
    "Vase": ((0.55, 0.26, 0.22), (0.85, 0.75, 0.55), (0.4, 0.2, 0.18), (0.3, 0.28, 0.26), 1050, 520, (0.06, 0.05, 0.05)),
    "Burj Khalifa": ((0.45, 0.55, 0.62), (0.75, 0.8, 0.85), (0.32, 0.4, 0.48), (0.28, 0.34, 0.42), 1150, 540, (0.04, 0.06, 0.1)),
    "Machu Picchu": ((0.45, 0.52, 0.38), (0.7, 0.66, 0.5), (0.32, 0.38, 0.28), (0.26, 0.34, 0.24), 1200, 500, (0.05, 0.07, 0.06)),
    "Forbidden City": ((0.62, 0.2, 0.16), (0.88, 0.72, 0.35), (0.5, 0.16, 0.13), (0.3, 0.28, 0.26), 1150, 500, (0.07, 0.05, 0.04)),
    "Laptop": ((0.28, 0.3, 0.34), (0.25, 0.6, 0.85), (0.2, 0.22, 0.26), (0.22, 0.24, 0.28), 900, 480, (0.03, 0.04, 0.05)),
    "Watch": ((0.75, 0.66, 0.4), (0.85, 0.8, 0.65), (0.25, 0.28, 0.32), (0.3, 0.32, 0.36), 1000, 520, (0.04, 0.04, 0.05)),
}

GEOMETRY_NAMES = {"box", "capsule", "cone", "cylinder", "ellipsoid", "extruded_polygon", "rounded_box", "torus"}
CAMERA_NAMES = {"dolly_camera", "follow_camera", "orbit_camera", "reveal_from_occluder"}


def materialize(request_path: Path, out_path: Path) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = request["payload"]
    plan = payload["director_plan"]
    job_hash = payload["director_plan_hash"]
    shot = plan["camera_plan"]["shots"][0]
    cue = str(shot.get("camera_cue") or "static")
    direction = shot.get("camera_direction")
    subject = next(entity for entity in plan["entities"] if entity["id"] == "prop_01_subject")
    label = subject["label"]
    mesh, extra, doc = subject_builder(label)
    palette = PALETTES[label]
    source = SOURCE_TEMPLATE
    intent = str(shot.get("intent") or "").replace("\n", " ")
    replacements = {
        "__SCENE_ID__": plan["request"]["scene_id"],
        "__SUBJECT_LABEL__": label,
        "__CUE_SUMMARY__": (cue + " " + (direction or "")).strip(),
        "__INTENT_LINE__": intent,
        "__PLAN_LITERAL__": repr(json.dumps(plan, ensure_ascii=False, separators=(",", ":"))),
        "__PLAN_HASH__": job_hash,
        "__REQUIRED_ENTITIES__": repr(sorted(entity["id"] for entity in plan["entities"])),
        "__REQUIRED_EVENTS__": repr(sorted({event["id"] for event in plan["events"]})),
        "__REQUIRED_CAMERA_EVENTS__": repr(sorted(plan.get("coverage_obligations") or [])),
        "__SUBJECT_BASE_Z__": repr(SUBJECT_BASE_Z),
        "__ENV_GROUND__": "rounded_box((0.0, 0.0, 0.0), (18.0, 13.0, 0.3), 0.1)",
        "__SUBJECT_MESH__": mesh,
        "__SUBJECT_EXTRA__": extra,
        "__SUBJECT_DOCSTRING__": doc,
        "__SUBJECT_LABEL_LITERAL__": repr(label),
        "__CAMERA_BODY__": build_camera_body(cue, direction),
        "__WORLD_COLOR__": repr(palette[6]),
        "__PRIMARY_COLOR__": repr(palette[0]),
        "__PRIMARY_METALLIC__": "0.08",
        "__PRIMARY_ROUGH__": "0.45",
        "__ACCENT_COLOR__": repr(palette[1]),
        "__ACCENT_METALLIC__": "0.05",
        "__ACCENT_ROUGH__": "0.5",
        "__SECONDARY_COLOR__": repr(palette[2]),
        "__GROUND_COLOR__": repr(palette[3]),
        "__KEY_ENERGY__": str(palette[4]),
        "__FILL_ENERGY__": str(palette[5]),
        "__CUE_LITERAL__": repr(cue),
        "__DIRECTION_LITERAL__": repr(direction),
    }
    for key, value in replacements.items():
        source = source.replace(key, value)
    used_geometry = sorted({name for name in GEOMETRY_NAMES if re.search(r"(?<![\w.])" + name + r"\s*\(", source)})
    used_camera = sorted({name for name in CAMERA_NAMES if re.search(r"(?<![\w.])" + name + r"\s*\(", source)})
    source = source.replace("__GEOMETRY_IMPORTS__", ", ".join(used_geometry))
    source = source.replace("__CAMERA_IMPORTS__", ", ".join(used_camera))
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
            name
            for name in allowed
            if re.search(r"(?<![\w.])" + name + r"\s*\(", source)
        ),
    }
    response_path = Path(request["respond_to"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"ok": True, "calls": response["library_calls"], "response": str(response_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="pending codegen request JSON path")
    parser.add_argument("--source-out", required=True)
    args = parser.parse_args()
    result = materialize(Path(args.request), Path(args.source_out))
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
