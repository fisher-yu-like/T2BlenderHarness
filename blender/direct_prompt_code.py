"""One-step raw-prompt to Blender-code compiler for the ablation arm.

This module deliberately has no dependency on the Director planning stack. It
derives only a small visual scaffold from the raw prompt and emits a
self-contained Blender Python job. The ablation is intentionally weaker than
the contract-first Harness; its purpose is to isolate the value of explicit
event, trajectory, and camera planning.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ACTOR_NAMES = ("Alice", "Bob", "Carla", "Dana", "Eve", "Mina", "Noah")
COLORS = ("red", "blue", "green", "yellow", "orange", "purple")


def _tokens(prompt: str) -> tuple[list[str], list[str]]:
    actors = []
    for match in re.finditer(r"\b[A-Z][a-z]+\b", prompt):
        name = match.group(0)
        if name in ACTOR_NAMES and name not in actors:
            actors.append(name)
    actors = (actors + ["Alice", "Bob"])[:2]

    props = []
    for match in re.finditer(r"\b(?:" + "|".join(COLORS) + r")\s+[a-z]+\b", prompt.lower()):
        label = match.group(0)
        if label not in props:
            props.append(label)
    props = (props + ["red cube", "blue cup"])[:2]
    return actors, props


def _camera_style(prompt: str) -> str:
    lower = prompt.lower()
    for token in ("orbit", "zoom", "dolly", "follow", "pan", "tilt"):
        if token in lower:
            return token
    return "hold"


def _state(frame: int, position: tuple[float, float, float]) -> dict[str, Any]:
    return {"frame": frame, "position": list(position), "rotation": [0.0, 0.0, 0.0]}


def build_direct_spec(prompt: str, *, duration_s: float, fps: int, seed: int) -> dict[str, Any]:
    actors, props = _tokens(prompt)
    end = max(2, round(duration_s * fps))
    mid = max(2, round(end * 0.48))
    late = max(mid + 1, round(end * 0.78))
    entities = {
        "actor_a": {"kind": "character", "label": actors[0], "states": [
            _state(1, (-3.4, -1.0, 0.0)),
            _state(mid, (-0.8, -0.7, 0.0)),
            _state(end, (2.0, -0.55, 0.0)),
        ]},
        "actor_b": {"kind": "character", "label": actors[1], "states": [
            _state(1, (-3.3, 1.0, 0.0)),
            _state(late, (0.3, 1.0, 0.0)),
            _state(end, (2.3, 1.0, 0.0)),
        ]},
        "prop_a": {"kind": "prop", "label": props[0], "states": [
            _state(1, (-2.6, -0.7, 1.6)),
            _state(mid, (-0.45, -0.2, 2.2)),
            _state(end, (2.0, 0.1, 1.45)),
        ]},
        "prop_b": {"kind": "prop", "label": props[1], "states": [
            _state(1, (-2.4, 1.0, 1.55)),
            _state(late, (0.8, 1.0, 1.55)),
            _state(end, (2.4, 1.0, 1.45)),
        ]},
    }
    camera_style = _camera_style(prompt)
    if camera_style == "orbit":
        camera_locations = [(8.0, -10.0, 6.0), (-7.0, -9.0, 5.0)]
    elif camera_style in {"zoom", "dolly"}:
        camera_locations = [(8.0, -10.0, 6.0), (4.0, -5.0, 3.3)]
    elif camera_style in {"pan", "tilt", "follow"}:
        camera_locations = [(7.0, -10.0, 5.2), (7.0, -7.0, 4.6)]
    else:
        camera_locations = [(8.0, -10.0, 6.0), (8.0, -10.0, 6.0)]
    return {
        "planning_mode": "direct_prompt_code",
        "prompt": prompt,
        "seed": seed,
        "fps": fps,
        "frame_start": 1,
        "frame_end": end,
        "actors": actors,
        "props": props,
        "entities": entities,
        "camera": {
            "style": camera_style,
            "start": list(camera_locations[0]),
            "end": list(camera_locations[1]),
            "target": [0.0, 0.0, 1.25],
        },
    }


def compile_direct_prompt_job(
    prompt: str,
    *,
    case_id: str,
    output_dir: str | Path,
    duration_s: float,
    fps: int,
    seed: int,
    render_settings: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    output = Path(output_dir).resolve()
    spec = build_direct_spec(prompt, duration_s=duration_s, fps=fps, seed=seed)
    spec_json = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    settings_json = json.dumps(render_settings, ensure_ascii=False, sort_keys=True)
    sample_frames = sorted({1, max(1, spec["frame_end"] // 2), spec["frame_end"]})
    source = f'''"""Generated one-step raw-prompt Blender code for an ablation run."""
from pathlib import Path
import hashlib
import json
import math

import bpy
from mathutils import Vector

OUTPUT_DIR = Path({str(output)!r})
FRAMES_DIR = OUTPUT_DIR / "frames"
SPEC = json.loads({spec_json!r})
INITIAL_MANIFEST = json.loads({manifest_json!r})
RENDER_SETTINGS = json.loads({settings_json!r})
SAMPLE_FRAMES = {sample_frames!r}


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def material(name, color):
    item = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    item.diffuse_color = (*color, 1.0)
    return item


def mesh_piece(name, kind, location, scale, mat, primitive="cube", parent=None):
    if primitive == "sphere":
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1.0, location=location)
    elif primitive == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=1.0, depth=2.0, location=location)
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    obj.data.materials.append(mat)
    obj["entity_kind"] = kind
    if parent is not None:
        obj.parent = parent
        obj.location = location
    return obj


def add_actor(entity_id, location, mat):
    root = bpy.data.objects.new(entity_id, None)
    bpy.context.collection.objects.link(root)
    root.location = location
    root["entity_kind"] = "character"
    mesh_piece(entity_id + "_body", "character", (0.0, 0.0, 1.55), (0.62, 0.46, 1.05), mat, "sphere", root)
    mesh_piece(entity_id + "_head", "character", (0.0, 0.0, 2.95), (0.40, 0.36, 0.42), mat, "sphere", root)
    for side in (-1.0, 1.0):
        mesh_piece(entity_id + ("_arm_l" if side < 0 else "_arm_r"), "character", (side * 0.72, 0.0, 1.9), (0.14, 0.14, 0.75), mat, "cylinder", root)
        mesh_piece(entity_id + ("_leg_l" if side < 0 else "_leg_r"), "character", (side * 0.28, 0.0, 0.55), (0.18, 0.18, 0.65), mat, "cylinder", root)
    return root


def add_prop(entity_id, location, mat):
    return mesh_piece(entity_id, "prop", location, (0.48, 0.48, 0.48), mat, "cube")


def add_support(mat):
    support = mesh_piece("support_surface", "support", (0.0, 0.0, 0.75), (4.8, 2.7, 0.28), mat, "cube")
    for x in (-3.8, 3.8):
        for y in (-1.9, 1.9):
            mesh_piece(f"support_leg_{{x}}_{{y}}", "support", (x, y, 0.25), (0.25, 0.25, 0.8), mat, "cube")
    mesh_piece("drop_zone", "support", (2.8, 0.3, 1.1), (1.0, 1.0, 0.05), mat, "cube")


def look_at(camera, target):
    direction = Vector(target) - camera.location
    if direction.length > 0:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera():
    bpy.ops.object.camera_add(location=SPEC["camera"]["start"])
    camera = bpy.context.object
    camera.name = "DirectPromptCamera"
    bpy.context.scene.camera = camera
    camera.data.lens = 48.0
    look_at(camera, SPEC["camera"]["target"])
    camera.keyframe_insert(data_path="location", frame=SPEC["frame_start"])
    camera.keyframe_insert(data_path="rotation_euler", frame=SPEC["frame_start"])
    camera.location = SPEC["camera"]["end"]
    look_at(camera, SPEC["camera"]["target"])
    camera.keyframe_insert(data_path="location", frame=SPEC["frame_end"])
    camera.keyframe_insert(data_path="rotation_euler", frame=SPEC["frame_end"])
    return camera


def configure(scene):
    preferred = RENDER_SETTINGS.get("engine", "BLENDER_EEVEE_NEXT")
    for candidate in (preferred, "BLENDER_EEVEE", "BLENDER_WORKBENCH", "CYCLES"):
        try:
            scene.render.engine = candidate
            break
        except (TypeError, ValueError):
            continue
    scene.render.resolution_x = int(RENDER_SETTINGS.get("resolution", [128, 128])[0])
    scene.render.resolution_y = int(RENDER_SETTINGS.get("resolution", [128, 128])[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = int(SPEC["fps"])
    scene.frame_start = int(SPEC["frame_start"])
    scene.frame_end = int(SPEC["frame_end"])
    scene.render.filepath = str(FRAMES_DIR / "animation" / "frame_")
    scene.world.color = (0.04, 0.04, 0.04)


def animate(obj, states):
    for state in states:
        obj.location = tuple(state["position"])
        obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
        obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
        obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))


def write_telemetry(objects, camera):
    telemetry = {{
        "planning_mode": "direct_prompt_code",
        "prompt": SPEC["prompt"],
        "frame_start": SPEC["frame_start"],
        "frame_end": SPEC["frame_end"],
        "fps": SPEC["fps"],
        "blender_version": bpy.app.version_string,
        "objects": {{entity_id: {{"kind": obj.get("entity_kind", "unknown"), "keyframe_count": len(SPEC["entities"].get(entity_id, {{}}).get("states", []))}} for entity_id, obj in objects.items()}},
        "camera": {{"name": camera.name, "active": bpy.context.scene.camera.name == camera.name}},
        "camera_shots": [{{"shot_id": "direct_prompt_camera", "trajectory_type": SPEC["camera"]["style"], "target_ids": list(objects), "required_event_ids": []}}],
        "event_observability": [],
        "current_owner_by_event": {{}},
        "final_support_by_prop": {{}},
        "interaction_state": {{}},
        "render_settings": RENDER_SETTINGS,
    }}
    (OUTPUT_DIR / "telemetry.json").write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")


def write_sample_frames(scene):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for frame in SAMPLE_FRAMES:
        scene.frame_set(int(frame))
        relative = f"frame_{{int(frame):06d}}.png"
        scene.render.filepath = str(FRAMES_DIR / relative)
        bpy.ops.render.render(write_still=True)
        index.append({{"frame": int(frame), "path": relative}})
    (FRAMES_DIR / "index.json").write_text(json.dumps({{"frames": index}}, indent=2), encoding="utf-8")


def write_metadata():
    entity_payload = [
        {{"id": entity_id, "kind": data["kind"], "label": data["label"]}}
        for entity_id, data in SPEC["entities"].items()
    ]
    (OUTPUT_DIR / "scene_contract.json").write_text(json.dumps({{
        "planning_mode": "direct_prompt_code", "prompt": SPEC["prompt"], "fps": SPEC["fps"],
        "entities": entity_payload, "events": [], "must_show": [],
    }}, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "trajectory.json").write_text(json.dumps({{
        "planning_mode": "direct_prompt_code", "entities": SPEC["entities"], "camera": {{"shots": []}},
    }}, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "camera_plan.json").write_text(json.dumps({{
        "planning_mode": "direct_prompt_code", "shots": [{{
            "shot_id": "direct_prompt_camera", "start_frame": SPEC["frame_start"],
            "end_frame": SPEC["frame_end"], "trajectory_type": SPEC["camera"]["style"],
            "target_ids": list(SPEC["entities"]), "required_event_ids": [],
        }}],
    }}, indent=2, sort_keys=True), encoding="utf-8")


def update_manifest():
    manifest = dict(INITIAL_MANIFEST)
    manifest["blender_version"] = bpy.app.version_string
    manifest["state"] = "rendered"
    encoded = json.dumps({{
        "prompt_hash": manifest["prompt_hash"], "plan_hash": manifest["plan_hash"],
        "director_plan_hash": None, "harness_version": manifest["harness_version"],
        "evaluator_version": manifest["evaluator_version"], "blender_version": manifest["blender_version"],
        "render_settings": manifest["render_settings"],
    }}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
(FRAMES_DIR / "animation").mkdir(parents=True, exist_ok=True)
reset_scene()
scene = bpy.context.scene
white = material("ProxyWhiteMaterial", (0.8, 0.8, 0.8))
add_support(white)
objects = {{
    "actor_a": add_actor("actor_a", tuple(SPEC["entities"]["actor_a"]["states"][0]["position"]), white),
    "actor_b": add_actor("actor_b", tuple(SPEC["entities"]["actor_b"]["states"][0]["position"]), white),
    "prop_a": add_prop("prop_a", tuple(SPEC["entities"]["prop_a"]["states"][0]["position"]), white),
    "prop_b": add_prop("prop_b", tuple(SPEC["entities"]["prop_b"]["states"][0]["position"]), white),
}}
for entity_id in ("actor_a", "actor_b", "prop_a", "prop_b"):
    animate(objects[entity_id], SPEC["entities"][entity_id]["states"])
add_camera()
camera = bpy.context.scene.camera
bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 7.0))
bpy.context.object.data.energy = 1100
bpy.context.object.data.shape = "DISK"
bpy.context.object.data.size = 6.0
configure(scene)
write_metadata()
write_telemetry(objects, camera)
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "proxy.blend"))
scene.render.filepath = str(FRAMES_DIR / "animation" / "frame_")
bpy.ops.render.render(animation=True)
write_sample_frames(scene)
update_manifest()
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "proxy.blend"))
'''
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    metadata = {
        "planning_mode": "direct_prompt_code",
        "case_id": case_id,
        "prompt": prompt,
        "source_hash": source_hash,
        "seed": seed,
        "derived_actors": spec["actors"],
        "derived_props": spec["props"],
        "camera_style": spec["camera"]["style"],
        "event_graph_source": "none; raw prompt is compiled directly",
    }
    return source, metadata
