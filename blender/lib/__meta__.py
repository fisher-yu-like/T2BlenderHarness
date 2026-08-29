"""Discover the public verified primitive signatures without importing Blender."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from . import primitive_registry


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
        grouped.setdefault(str(metadata["category"]), []).append(metadata)
    return {
        category: sorted(entries, key=lambda item: str(item["name"]))
        for category, entries in sorted(grouped.items())
    }


__all__ = ["collect_library_signatures"]
