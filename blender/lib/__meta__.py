"""Discover the public verified primitive signatures without importing Blender."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from . import primitive_registry


_RETURN_CONTRACTS = {
    "geometry": "mesh_data_vertices_faces",
    "camera": "camera_keyframes",
    "constraints": "constraint_spec",
    "rigging": {
        "minimal_humanoid_armature": "armature_spec",
        "bind_mesh_to_armature": "vertex_weight_map",
        "add_ik_constraint": "constraint_spec",
    },
    "layout": {
        "lane_separated_paths": "trajectories_by_entity",
        "place_on_surface": "translation_vector",
        "handoff_constraint_sequence": "constraint_specs",
        "avoid_penetration": "trajectory",
    },
    "scaffolding": {
        "build_runtime_contract": "runtime_contract",
        "validate_runtime_contract": "validation_failures",
    },
}


def _return_contract(category: str, name: str) -> str:
    value = _RETURN_CONTRACTS.get(category, "verified_value")
    if isinstance(value, dict):
        return str(value.get(name, "verified_value"))
    return str(value)


def _library_functions() -> list[Callable[..., Any]]:
    # Imports are intentionally local: importing ``blender.lib`` alone should
    # remain cheap, while signature export eagerly loads every public module.
    from . import camera, constraints, geometry, layout, rigging, scaffolding

    modules = (geometry, rigging, constraints, camera, layout, scaffolding)
    functions: list[Callable[..., Any]] = []
    for module in modules:
        for _name, function in inspect.getmembers(module, inspect.isfunction):
            if hasattr(function, "primitive_metadata"):
                functions.append(function)
    return functions


def collect_library_signatures() -> dict[str, list[dict[str, Any]]]:
    """Return deterministic, JSON-compatible metadata grouped by category."""

    functions = _library_functions()
    registry = primitive_registry()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for function in functions:
        metadata = dict(registry[function.__name__])
        metadata["signature"] = f"{function.__name__}{inspect.signature(function)}"
        metadata["docstring"] = (function.__doc__ or "").strip()
        metadata["module"] = str(function.__module__)
        metadata["return_contract"] = _return_contract(
            str(metadata["category"]), function.__name__
        )
        grouped.setdefault(str(metadata["category"]), []).append(metadata)
    return {
        category: sorted(entries, key=lambda item: str(item["name"]))
        for category, entries in sorted(grouped.items())
    }


__all__ = ["collect_library_signatures"]
