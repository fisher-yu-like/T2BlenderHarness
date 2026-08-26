"""Run inside Blender and export mesh topology/semantic evidence as JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


def primitive_hint(obj: object) -> str | None:
    # Entity IDs such as ``blue_cube`` are semantic labels, not evidence that
    # Blender used a cube primitive. Inspect the mesh datablock name or an
    # explicit author-provided hint instead of the object/entity ID.
    custom_hint = getattr(obj, "get", lambda *_args: None)("primitive_hint")
    if custom_hint:
        return str(custom_hint).lower()
    text = str(getattr(getattr(obj, "data", None), "name", "")).lower().strip()
    # Match Blender's primitive datablock names exactly. A generated mesh may
    # legitimately be named ``blue_cube_detailed_mesh`` because ``cube`` is a
    # semantic object label; that must not be treated as primitive evidence.
    for token in ("uv_sphere", "sphere", "cylinder", "cube"):
        if re.fullmatch(rf"{re.escape(token)}(?:\.\d+)?", text):
            return token
    return None


def connected_component_count(mesh: object) -> int:
    vertex_count = len(getattr(mesh, "vertices", []))
    if vertex_count == 0:
        return 0
    parent = list(range(vertex_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    for edge in getattr(mesh, "edges", []):
        edge_vertices = list(edge.vertices)
        if len(edge_vertices) == 2:
            union(int(edge_vertices[0]), int(edge_vertices[1]))
    return len({find(index) for index in range(vertex_count)})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    # Blender appends script arguments after a literal ``--`` but retains the
    # separator in sys.argv.  Parse only the payload after it; this also keeps
    # the script usable when launched directly with normal Python arguments.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    args = parser.parse_args(argv)
    meshes = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        meshes.append({
            "name": obj.name,
            "entity_id": obj.get("entity_id", obj.name),
            "entity_kind": obj.get("entity_kind"),
            "geometry_style": obj.get("geometry_style"),
            "mesh_name": obj.data.name,
            "primitive_hint": primitive_hint(obj),
            "vertex_count": len(obj.data.vertices),
            "face_count": len(obj.data.polygons),
            "polygon_count": len(obj.data.polygons),
            "material_slot_count": len(obj.material_slots),
            "modifier_count": len(obj.modifiers),
            "connected_component_count": connected_component_count(obj.data),
            "dimensions": [float(value) for value in obj.dimensions],
        })
    payload = {
        "audit_available": True,
        "blender_version": bpy.app.version_string,
        "mesh_count": len(meshes),
        "meshes": meshes,
    }
    Path(args.output).resolve().write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(Path(args.output).resolve()), "mesh_count": len(meshes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
