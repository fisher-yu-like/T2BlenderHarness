"""Fixed Blender-side observer for untrusted generated scene jobs.

This file is executed in a fresh Blender process after a generated job has
saved ``candidate.blend``.  It never imports or reads the generated job and it
does not trust any telemetry already present in the case directory.  Only the
raw transforms, bounds, camera state, pose state, and constraint state read
from Blender are emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


OBSERVER_SCHEMA_VERSION = "trusted-observer-v1"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_vector(value: Any) -> list[float]:
    return [round(float(value[index]), 6) for index in range(3)]


def _json_matrix(value: Any) -> list[list[float]]:
    return [[round(float(value[row][column]), 6) for column in range(4)] for row in range(4)]


def _world_bounds(obj: Any) -> dict[str, Any] | None:
    from mathutils import Vector  # type: ignore

    bound_box = getattr(obj, "bound_box", None)
    matrix_world = getattr(obj, "matrix_world", None)
    if not bound_box or matrix_world is None:
        return None
    # Blender exposes ``Object.bound_box`` as a bpy_prop_array rather than a
    # mathutils.Vector.  Convert at the boundary before matrix multiplication.
    points = [matrix_world @ Vector(point) for point in bound_box]
    mins = [min(float(point[index]) for point in points) for index in range(3)]
    maxs = [max(float(point[index]) for point in points) for index in range(3)]
    return {"min": [round(value, 6) for value in mins], "max": [round(value, 6) for value in maxs]}


def _world_obb(obj: Any) -> dict[str, Any] | None:
    """Export raw oriented bounds; no contact or ownership is inferred here."""

    from mathutils import Vector  # type: ignore

    bound_box = getattr(obj, "bound_box", None)
    matrix_world = getattr(obj, "matrix_world", None)
    if not bound_box or matrix_world is None:
        return None
    local_points = [Vector(point) for point in bound_box]
    local_min = [min(float(point[index]) for point in local_points) for index in range(3)]
    local_max = [max(float(point[index]) for point in local_points) for index in range(3)]
    local_center = Vector([(low + high) / 2.0 for low, high in zip(local_min, local_max)])
    local_half = [(high - low) / 2.0 for low, high in zip(local_min, local_max)]
    origin = matrix_world @ Vector((0.0, 0.0, 0.0))
    linear = matrix_world.to_3x3()
    axes = []
    half_extents = []
    for index, basis in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        transformed = linear @ Vector(basis)
        length = float(transformed.length)
        if length <= 1e-9:
            return None
        axes.append(_json_vector(transformed.normalized()))
        half_extents.append(round(float(local_half[index]) * length, 6))
    center = matrix_world @ local_center
    return {
        "center": _json_vector(center),
        "axes": axes,
        "half_extents": half_extents,
    }


def _world_mesh_triangles(obj: Any, *, max_triangles: int = 512) -> list[list[list[float]]]:
    """Export a bounded world-space collision mesh for the physics narrow phase."""

    from mathutils import Vector  # type: ignore

    mesh = getattr(obj, "data", None)
    matrix_world = getattr(obj, "matrix_world", None)
    vertices = getattr(mesh, "vertices", None)
    polygons = getattr(mesh, "polygons", None)
    if mesh is None or matrix_world is None or vertices is None or polygons is None:
        return []
    world_vertices = {
        int(vertex.index): matrix_world @ Vector(vertex.co)
        for vertex in vertices
        if hasattr(vertex, "index") and hasattr(vertex, "co")
    }
    result: list[list[list[float]]] = []
    for polygon in polygons:
        indices = [int(index) for index in getattr(polygon, "vertices", [])]
        if len(indices) < 3:
            continue
        for index in range(1, len(indices) - 1):
            points = [world_vertices.get(indices[0]), world_vertices.get(indices[index]), world_vertices.get(indices[index + 1])]
            if any(point is None for point in points):
                continue
            result.append([_json_vector(point) for point in points])  # type: ignore[arg-type]
            if len(result) >= max_triangles:
                return result
    return result


def _constraint_state(obj: Any) -> list[dict[str, Any]]:
    result = []
    for constraint in getattr(obj, "constraints", []):
        target = getattr(constraint, "target", None)
        result.append(
            {
                "name": str(getattr(constraint, "name", "")),
                "type": str(getattr(constraint, "type", "")),
                "influence": round(float(getattr(constraint, "influence", 0.0)), 6),
                "target": str(getattr(target, "name", "")) if target is not None else None,
                "subtarget": str(getattr(constraint, "subtarget", "")),
            }
        )
    return result


def _bone_state(obj: Any) -> dict[str, Any]:
    pose = getattr(obj, "pose", None)
    if pose is None:
        return {}
    result: dict[str, Any] = {}
    for bone in sorted(pose.bones, key=lambda item: item.name):
        result[str(bone.name)] = {
            "head": _json_vector(obj.matrix_world @ bone.head),
            "tail": _json_vector(obj.matrix_world @ bone.tail),
            "matrix": _json_matrix(obj.matrix_world @ bone.matrix),
        }
    return result


def _entity_id(obj: Any) -> str | None:
    value = obj.get("entity_id") if hasattr(obj, "get") else None
    if value:
        return str(value)
    name = str(getattr(obj, "name", ""))
    if name.endswith("__armature"):
        return name.removesuffix("__armature")
    return None


def _entity_objects(bpy: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for obj in sorted(bpy.data.objects, key=lambda item: item.name):
        if getattr(obj, "type", None) not in {"MESH", "ARMATURE"}:
            continue
        entity_id = _entity_id(obj)
        if entity_id:
            # The mesh is the visual entity. Blender's lexical ordering can
            # put an armature before the visual mesh, so never let the rig
            # shadow the entity geometry in the primary map.
            current = result.get(entity_id)
            if current is None or (current.type == "ARMATURE" and obj.type == "MESH"):
                result[entity_id] = obj
    return result


def _armature_objects(bpy: Any) -> dict[str, Any]:
    return {
        entity_id: obj
        for obj in sorted(bpy.data.objects, key=lambda item: item.name)
        if getattr(obj, "type", None) == "ARMATURE"
        for entity_id in [_entity_id(obj)]
        if entity_id
    }


def _raw_entity_metadata(obj: Any, rig: Any | None = None) -> dict[str, Any]:
    mesh = getattr(obj, "data", None)
    return {
        "object_name": str(obj.name),
        "object_type": str(obj.type),
        "entity_kind": obj.get("entity_kind") if hasattr(obj, "get") else None,
        "geometry_style": obj.get("geometry_style") if hasattr(obj, "get") else None,
        "vertex_count": int(len(mesh.vertices)) if mesh is not None and hasattr(mesh, "vertices") else 0,
        "face_count": int(len(mesh.polygons)) if mesh is not None and hasattr(mesh, "polygons") else 0,
        "constraints": _constraint_state(obj),
        "rig_object_name": str(rig.name) if rig is not None else None,
        "rig_bone_names": [str(bone.name) for bone in sorted(rig.data.bones, key=lambda item: item.name)] if rig is not None else [],
    }


def _raw_frame_observation(
    scene: Any,
    entity_objects: dict[str, Any],
    armature_objects: dict[str, Any],
    camera: Any,
    frame: int,
    mesh_entity_ids: set[str] | None = None,
) -> dict[str, Any]:
    scene.frame_set(int(frame))
    entities = {}
    for entity_id, obj in entity_objects.items():
        entity = {
            "object_name": str(obj.name),
            "matrix_world": _json_matrix(obj.matrix_world),
            "location": _json_vector(obj.matrix_world.translation),
            "rotation_euler": _json_vector(obj.matrix_world.to_euler()),
            "world_bounds": _world_bounds(obj),
            "obb": _world_obb(obj),
            "constraints": _constraint_state(obj),
            "bones": _bone_state(obj) if obj.type == "ARMATURE" else {},
            "pose_bones": _bone_state(armature_objects[entity_id]) if entity_id in armature_objects else {},
        }
        if mesh_entity_ids and entity_id in mesh_entity_ids:
            triangles = _world_mesh_triangles(obj)
            if triangles:
                entity["mesh_triangles"] = triangles
        entities[entity_id] = entity
    camera_state = None
    if camera is not None:
        camera_state = {
            "object_name": str(camera.name),
            "matrix_world": _json_matrix(camera.matrix_world),
            "location": _json_vector(camera.matrix_world.translation),
            "rotation_euler": _json_vector(camera.matrix_world.to_euler()),
            "lens": round(float(getattr(camera.data, "lens", 0.0)), 6),
        }
    return {"frame": int(frame), "entities": entities, "camera": camera_state}


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--observer-source-sha256", required=True)
    return parser.parse_args(argv)


def observe(*, run_dir: str | Path, request_path: str | Path, observer_source_hash: str) -> dict[str, Any]:
    """Collect raw Blender state and render trusted observer frames."""

    import bpy  # type: ignore

    root = Path(run_dir).resolve()
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    candidate = root / "candidate.blend"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    candidate_hash = _hash_file(candidate)
    if candidate_hash != request.get("candidate_blend_hash"):
        raise RuntimeError("candidate blend changed after observer request")
    if str(observer_source_hash) != str(request.get("observer_source_hash")):
        raise RuntimeError("observer source hash does not match host request")

    scene = bpy.context.scene
    entity_objects = _entity_objects(bpy)
    armature_objects = _armature_objects(bpy)
    camera = scene.camera
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    fps = int(round(float(scene.render.fps)))
    mesh_entity_ids = {
        str(value)
        for value in request.get("mesh_entity_ids", [])
        if isinstance(value, str) and value.strip()
    }
    obligation_ids = [
        str(value)
        for value in request.get("obligation_ids", [])
        if isinstance(value, str) and value.strip()
    ]
    frames = list(range(frame_start, frame_end + 1))
    observations = [
        _raw_frame_observation(
            scene,
            entity_objects,
            armature_objects,
            camera,
            frame,
            mesh_entity_ids=mesh_entity_ids,
        )
        for frame in frames
    ]
    camera_observations = [
        {"frame": item["frame"], "camera": item["camera"]}
        for item in observations
        if item.get("camera") is not None
    ]
    telemetry = {
        "schema_version": OBSERVER_SCHEMA_VERSION,
        "observer_version": OBSERVER_SCHEMA_VERSION,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "fps": fps,
        "entities": {
            entity_id: _raw_entity_metadata(obj, armature_objects.get(entity_id))
            for entity_id, obj in entity_objects.items()
        },
        "objects": {
            entity_id: {
                "entity_id": entity_id,
                "source_entity_id": entity_id,
                "kind": (
                    str(obj.get("entity_kind"))
                    if obj.get("entity_kind")
                    else "character" if entity_id in armature_objects else "prop"
                ),
                "object_name": str(obj.name),
            }
            for entity_id, obj in entity_objects.items()
        },
        "observations": observations,
        "camera_observations": camera_observations,
        "camera": {
            "active": camera is not None,
            "object_name": str(camera.name) if camera is not None else None,
        },
        "raw_scene_object_count": int(len(bpy.data.objects)),
        # Identity-only trace anchors.  The observer never derives semantic
        # success/failure from them and never accepts generated telemetry.
        "obligation_ids": list(dict.fromkeys(obligation_ids)),
    }
    telemetry_path = root / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    frames_dir = root / "frames"
    animation_dir = frames_dir / "animation"
    animation_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(animation_dir / "frame_")
    bpy.ops.render.render(animation=True)
    rendered = sorted(animation_dir.glob("frame_*.png"))
    (frames_dir / "index.json").write_text(
        json.dumps(
            {"schema_version": OBSERVER_SCHEMA_VERSION, "frames": [{"frame": int(path.stem.rsplit("_", 1)[-1]), "path": str(path.relative_to(frames_dir)).replace("\\", "/")} for path in rendered]},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    telemetry_hash = _hash_file(telemetry_path)
    manifest = {
        "schema_version": OBSERVER_SCHEMA_VERSION,
        "trusted": True,
        "observer_version": OBSERVER_SCHEMA_VERSION,
        "request_nonce": request.get("nonce"),
        "candidate_blend_hash": candidate_hash,
        "observer_source_hash": str(observer_source_hash),
        "telemetry_hash": telemetry_hash,
        "frame_count": len(rendered),
        "obligation_ids": list(dict.fromkeys(obligation_ids)),
    }
    (root / "telemetry_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main() -> int:
    # Blender passes arguments after a literal ``--`` to this script.
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    args = _parse_args(argv)
    observe(run_dir=args.run_dir, request_path=args.request, observer_source_hash=args.observer_source_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
