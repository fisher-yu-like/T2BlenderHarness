"""Generate a self-contained Blender Python job for a real white proxy render."""

from __future__ import annotations

import json
from pathlib import Path

from videoact.contracts import RunManifest, TrajectoryPlan


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
) -> str:
    output = Path(output_dir).resolve()
    plan_json = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    manifest_json = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    proxy_json = json.dumps(proxy_spec or {}, sort_keys=True)
    samples_json = json.dumps(list(sample_frames))
    # The authored proxy scene is authoritative for geometry semantics.  Plan
    # entities are added only as a compatibility fallback so this renderer
    # patch does not rewrite the frozen SceneContract/TrajectoryPlan.
    proxy_entities = []
    authored_by_id = {
        str(entity.get("id")): {"id": str(entity.get("id")), "kind": str(entity.get("kind", "prop"))}
        for entity in (proxy_spec or {}).get("entities", [])
        if entity.get("id")
    }
    for entity_id, entity in authored_by_id.items():
        proxy_entities.append(entity)
    for entity_id in plan.entities:
        if entity_id not in authored_by_id:
            proxy_entities.append({
                "id": entity_id,
                "kind": (
                    "character"
                    if entity_id == "character"
                    else "support"
                    if entity_id in {"table", "support", "surface", "platform", "drop_zone"}
                    else "prop"
                ),
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
    material.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    return material


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
    elif entity_id == "opening":
        return detailed_opening(entity_id, kind, material)
    return detailed_prop(entity_id, kind, material)


def initial_location(entity_id, kind):
    layout = PROXY_SPEC.get("layout", {{}})
    support = layout.get("support_position", (2.0, 0.0, 0.0))
    if kind == "character":
        if entity_id == "character":
            return tuple(layout.get("character_start_position", (-3.0, -2.0, 0.0)))
        return (float(support[0]) + 1.8, float(support[1]) + 1.8, 0.0)
    if entity_id == "table":
        return tuple(layout.get("support_position", (0.0, 0.0, 0.0)))
    if entity_id == "drop_zone":
        return tuple(layout.get("drop_zone_position", (0.0, 2.0, 0.0)))
    if kind == "occluder":
        support = layout.get("support_position", (0.0, 0.0, 0.0))
        return (float(support[0]) + 0.5, float(support[1]) + 0.5, 1.0)
    return (float(support[0]), float(support[1]), 1.0)


def add_light(material):
    bpy.ops.object.light_add(type="AREA", location=(4.0, -4.0, 6.0))
    light = bpy.context.object
    light.name = "ProxyKeyLight"
    light.data.energy = 900
    light.data.shape = "DISK"
    light.data.size = 5.0
    return light


def look_at(camera, target):
    direction = Vector(target) - camera.location
    if direction.length > 0:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera():
    bpy.ops.object.camera_add(location=(7.0, -8.0, 5.0))
    camera = bpy.context.object
    camera.name = "ProxyCamera"
    bpy.context.scene.camera = camera
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


def animate_entities(objects):
    for entity_id, trajectory in PLAN["entities"].items():
        obj = objects[entity_id]
        for state in trajectory["states"]:
            obj.location = tuple(state["position"])
            obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
            obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
            obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))


def animate_camera(camera, objects):
    shots = PLAN["camera"]["shots"]
    for shot in shots:
        target_id = (shot.get("target_ids") or ["character"])[0]
        target = objects.get(target_id, objects["character"])
        target_point = tuple(target.location)
        trajectory_type = shot.get("trajectory_type", "hold")
        if trajectory_type == "dolly" or shot["shot_id"].endswith("closeup"):
            start_location, end_location = (7.0, -8.0, 5.0), (3.5, -4.0, 2.8)
        elif trajectory_type == "orbit":
            start_location, end_location = (7.0, -8.0, 5.0), (-7.0, -8.0, 5.0)
        else:
            start_location = end_location = (7.0, -8.0, 5.0)
        camera.data.lens = float(shot.get("lens_mm", 50.0))
        camera.location = start_location
        look_at(camera, target_point)
        camera.keyframe_insert(data_path="location", frame=int(shot["start_frame"]))
        camera.keyframe_insert(data_path="rotation_euler", frame=int(shot["start_frame"]))
        camera.data.keyframe_insert(data_path="lens", frame=int(shot["start_frame"]))
        camera.location = end_location
        look_at(camera, target_point)
        camera.keyframe_insert(data_path="location", frame=int(shot["end_frame"]))
        camera.keyframe_insert(data_path="rotation_euler", frame=int(shot["end_frame"]))
        camera.data.keyframe_insert(data_path="lens", frame=int(shot["end_frame"]))


def write_telemetry(objects, camera, manifest):
    telemetry = {{
        "blender_version": bpy.app.version_string,
        "frame_start": manifest["frame_start"],
        "frame_end": manifest["frame_end"],
        "fps": manifest["fps"],
        "objects": {{
            entity_id: {{
                "kind": obj.get("entity_kind", "unknown"),
                "location": list(obj.location),
                "keyframe_count": len(PLAN["entities"].get(entity_id, dict()).get("states", [])),
            }}
            for entity_id, obj in objects.items()
        }},
        "camera": dict(name=camera.name, active=(bpy.context.scene.camera.name == camera.name)),
        "proxy_scene": {{
            "scene_id": PROXY_SPEC.get("scene_id"),
            "scene_seed": PROXY_SPEC.get("scene_seed"),
            "path_shape": PROXY_SPEC.get("layout", {{}}).get("path_shape"),
        }},
        "camera_shots": [
            {{"shot_id": shot["shot_id"], "trajectory_type": shot.get("trajectory_type", "hold")}}
            for shot in PLAN["camera"].get("shots", [])
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
    manifest["blender_version"] = bpy.app.version_string
    manifest["state"] = "rendered"
    manifest["fingerprint"] = canonical_hash({{
        "prompt_hash": manifest["prompt_hash"],
        "plan_hash": manifest["plan_hash"],
        "harness_version": manifest["harness_version"],
        "evaluator_version": manifest["evaluator_version"],
        "blender_version": manifest["blender_version"],
        "render_settings": manifest["render_settings"],
    }})
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
(FRAMES_DIR / "animation").mkdir(parents=True, exist_ok=True)
reset_scene()
scene = bpy.context.scene
material = white_material()
objects = {{
    entity["id"]: add_entity(entity["id"], entity, material)
    for entity in {json.dumps(proxy_entities, sort_keys=True)}
}}
for entity_id, obj in objects.items():
    entity_kind = obj.get("entity_kind", "prop")
    obj.location = initial_location(entity_id, entity_kind)
    if entity_id == "table":
        obj.scale = tuple(PROXY_SPEC.get("layout", {{}}).get("support_scale", obj.scale))
    elif entity_kind == "prop":
        obj.scale = tuple(PROXY_SPEC.get("layout", {{}}).get("object_scale", obj.scale))
add_light(material)
camera = add_camera()
configure_render(scene, INITIAL_MANIFEST)
animate_entities(objects)
animate_camera(camera, objects)
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
