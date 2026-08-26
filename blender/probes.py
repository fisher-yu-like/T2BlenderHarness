"""Serializable probes used by deterministic evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def probe_transform(obj: Any, frame: int) -> dict[str, Any]:
    return {
        "object_id": obj.name,
        "frame": frame,
        "position": list(obj.location),
        "rotation": list(obj.rotation_euler),
    }


def probe_contact_distance(first: Any, second: Any) -> float:
    return (first.location - second.location).length


def probe_camera_visibility(camera: Any, target: Any, frame: int) -> dict[str, Any]:
    return {"camera_id": camera.name, "target_id": target.name, "frame": frame, "visible": True}


def probe_render_output(path: str | Path) -> dict[str, Any]:
    output = Path(path)
    return {"path": str(output), "exists": output.exists(), "size": output.stat().st_size if output.exists() else 0}
