"""Blender-free verified primitive registry used by the code agent."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_PRIMITIVE_REGISTRY: dict[str, dict[str, Any]] = {}


def register_primitive(**metadata: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register public metadata without changing a primitive's call behavior."""

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        entry = dict(metadata)
        entry["name"] = function.__name__
        entry.setdefault("category", "uncategorized")
        entry.setdefault("tags", [])
        entry.setdefault("cost_estimate", "unknown")
        entry.setdefault("example_usage", "")
        entry.setdefault("usage_count", 0)
        entry["docstring"] = (function.__doc__ or "").strip()
        _PRIMITIVE_REGISTRY[function.__name__] = entry
        setattr(function, "primitive_metadata", entry)
        return function

    return decorator


def primitive_registry() -> dict[str, dict[str, Any]]:
    """Return a copy of registered primitive metadata."""

    return {name: dict(metadata) for name, metadata in _PRIMITIVE_REGISTRY.items()}


__all__ = ["primitive_registry", "register_primitive"]

