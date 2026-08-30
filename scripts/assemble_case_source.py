"""Assemble an authored case scene into the host-contract Blender job.

Split of responsibilities (the "no fixed template" boundary):

- AUTHORED PER CASE (by the driving glm-5.3-flash session, at request time):
  every entity's geometry, every animation, the camera work, colors - the
  session writes ``case_scene.py`` containing ``build_entities(materials,
  plan)``, ``animate(objects, plan)`` and ``add_camera(objects, plan)``.
- HOST CONTRACT (invariant, shared by every generated job by harness design):
  DIRECTOR_PLAN binding, artifact writes (telemetry.json, frames/index.json),
  candidate.blend save, animation render to frames/animation/frame_, render
  config, and the verified blender.lib import surface.  This wrapper contains
  zero scene decisions.

The assembler embeds the authored scene, enforces the static source gate, and
runs a mock-Blender dry-run before releasing the provider response.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
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

RUNTIME_TEMPLATE = '''"""Case-specific Blender job for __SCENE_ID__ - scene authored per case.

Host-contract runtime assembled around the session's authored scene code;
every scene decision (geometry, animation, camera) lives in CASE_SCENE below.
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
    return DIRECTOR_PLAN["trajectory_summary"]["entities"].get(entity_id, {}).get("states", [])


def plan_primitives(entity_id):
    return DIRECTOR_PLAN["trajectory_summary"]["entities"].get(entity_id, {}).get("motion_primitives", [])


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


def keyframe_states(obj, entity_id):
    for state in plan_states(entity_id):
        obj.location = tuple(state["position"])
        obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
        obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
        obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))


def build_runtime_stage(environment, stage_top):
    ground = mesh_object("ground", rounded_box((0.0, 0.0, 0.0), (20.0, 14.0, 0.3), 0.1), environment)
    ground.location = (0.0, 0.0, -0.16)
    stage = mesh_object("support_surface", rounded_box((0.0, 0.0, 0.0), (4.0, 3.0, 0.18), 0.06), environment)
    stage.location = (0.0, 0.0, max(0.18, stage_top) - 0.09)
    stage["entity_id"] = "support_surface"
    stage["entity_kind"] = "support"
    placement = place_on_surface(((-2.0, -1.5, 0.0), (2.0, 1.5, 0.18)), max(0.18, stage_top))
    stage["surface_placement"] = serialize_json(list(placement))
    return stage


def look_at(camera, target):
    direction = (target[0] - camera.location[0], target[1] - camera.location[1], target[2] - camera.location[2])
    if (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) > 0.0:
        from mathutils import Vector

        camera.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()


def apply_camera_keys(camera, keyframes, lens_mm):
    for keyframe in keyframes:
        camera.location = tuple(keyframe.location)
        look_at(camera, tuple(keyframe.target))
        camera.keyframe_insert(data_path="location", frame=int(keyframe.frame))
        camera.keyframe_insert(data_path="rotation_euler", frame=int(keyframe.frame))
    camera.data.lens = lens_mm
    camera.data.keyframe_insert(data_path="lens", frame=int(keyframes[0].frame))


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
            if plan_primitives(entity_id)
        },
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


# ---------------------------------------------------------------------------
# CASE SCENE - authored by the driving session for THIS case at request time.
# ---------------------------------------------------------------------------
__CASE_SCENE__


def main():
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reset_scene()
    scene = bpy.context.scene
    objects, skin_summaries = build_entities()
    # Common-cause guard: the trusted observer keys entities by the
    # entity_id custom property, so every planned entity must exist as an
    # object carrying its tag.  Fail here, not at evaluation time.
    for entity_id in REQUIRED_ENTITY_IDS:
        if entity_id not in objects:
            raise RuntimeError("authored scene did not build planned entity: " + entity_id)
        entity_obj = objects[entity_id]
        if not entity_obj.get("entity_id"):
            entity_obj["entity_id"] = entity_id
        if not entity_obj.get("entity_kind"):
            entity_obj["entity_kind"] = "entity"
    camera = add_camera(objects)
    apply_animation(objects)
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
    write_contract_artifacts(objects, camera, manifest)
    scene.render.filepath = str(ANIMATION_DIR / "frame_")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "candidate.blend"))
    bpy.ops.render.render(animation=True)
    render_sample_frames(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "candidate.blend"))


main()
'''


def materialize(request_path: Path, scene_code_path: Path, out_path: Path,
               dryrun_template: Path | None) -> dict:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    payload = request["payload"]
    plan = payload["director_plan"]
    job_hash = payload["director_plan_hash"]
    scene_code = scene_code_path.read_text(encoding="utf-8")
    for required in ("def build_entities", "def apply_animation", "def add_camera"):
        if required not in scene_code:
            return {"ok": False, "error": f"authored scene code missing {required}()"}
    source = RUNTIME_TEMPLATE
    source = source.replace("__SCENE_ID__", str(plan["request"]["scene_id"]))
    source = source.replace("__PLAN_LITERAL__", repr(json.dumps(plan, ensure_ascii=False, separators=(",", ":"))))
    source = source.replace("__PLAN_HASH__", job_hash)
    source = source.replace("__REQUIRED_ENTITIES__", repr(sorted(entity["id"] for entity in plan["entities"])))
    source = source.replace("__REQUIRED_EVENTS__", repr(sorted({event["id"] for event in plan["events"]})))
    source = source.replace("__REQUIRED_CAMERA_EVENTS__", repr(sorted(plan.get("coverage_obligations") or [])))
    source = source.replace("__CASE_SCENE__", scene_code)
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

    dry = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "mock_blender_dryrun.py"), str(out_path),
         str(dryrun_template or "")],
        capture_output=True, text=True,
    )
    if dry.returncode != 0:
        return {"ok": False, "error": "dryrun failed", "detail": (dry.stdout or "")[-400:]}

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--scene-code", required=True, help="authored case scene python")
    parser.add_argument("--source-out", required=True)
    parser.add_argument("--dryrun-template", default=None)
    args = parser.parse_args()
    result = materialize(Path(args.request), Path(args.scene_code), Path(args.source_out),
                         Path(args.dryrun_template) if args.dryrun_template else None)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
