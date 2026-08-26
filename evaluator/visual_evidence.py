"""Measure observable evidence in sampled render frames.

These measurements are intentionally low-level.  They can detect whether a
render contains readable pixels, spatial structure, and temporal change, but
they do not claim that a person, hand, cup, or camera event is semantically
correct.  Semantic claims require an independent VLM or human review.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal installs
    Image = None  # type: ignore[assignment]


VISUAL_EVIDENCE_VERSION = "render-visual-evidence-v1"
_TARGET_SIZE = (64, 64)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _sample_paths(run_dir: Path) -> list[Path]:
    index_path = run_dir / "frames" / "index.json"
    paths: list[Path] = []
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            for item in payload.get("frames", []):
                candidate = run_dir / "frames" / str(item.get("path", ""))
                if candidate.is_file():
                    paths.append(candidate)
        except (OSError, ValueError, TypeError):
            paths = []
    if not paths:
        paths = sorted((run_dir / "frames").glob("frame_*.png"))
    return list(dict.fromkeys(paths))[:12]


def _pixels(image: Any) -> tuple[int, int, list[tuple[int, int, int]]]:
    resized = image.convert("RGB").resize(_TARGET_SIZE)
    return resized.width, resized.height, list(resized.getdata())


def _frame_metrics(image: Any) -> dict[str, float | int]:
    width, height, values = _pixels(image)
    border: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if x in (0, width - 1) or y in (0, height - 1):
                border.append(values[y * width + x])
    background = tuple(sum(pixel[channel] for pixel in border) / len(border) for channel in range(3))
    distances = [math.sqrt(sum((pixel[channel] - background[channel]) ** 2 for channel in range(3))) for pixel in values]
    foreground_fraction = sum(distance > 18.0 for distance in distances) / len(distances)
    luminance = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in values]
    mean_luminance = sum(luminance) / len(luminance)
    luminance_std = math.sqrt(sum((value - mean_luminance) ** 2 for value in luminance) / len(luminance))
    horizontal = [abs(luminance[index] - luminance[index + 1]) for index in range(len(luminance) - 1) if index % width != width - 1]
    vertical = [abs(luminance[index] - luminance[index + width]) for index in range(len(luminance) - width)]
    gradients = horizontal + vertical
    edge_density = sum(gradient > 12.0 for gradient in gradients) / len(gradients)
    return {
        "width": int(image.width),
        "height": int(image.height),
        "foreground_fraction": round(foreground_fraction, 6),
        "luminance_std": round(luminance_std, 6),
        "edge_density": round(edge_density, 6),
    }


def _frame_difference(left: Any, right: Any) -> float:
    _, _, left_pixels = _pixels(left)
    _, _, right_pixels = _pixels(right)
    differences = [
        sum(abs(left_pixel[channel] - right_pixel[channel]) for channel in range(3)) / (3.0 * 255.0)
        for left_pixel, right_pixel in zip(left_pixels, right_pixels)
    ]
    return sum(differences) / len(differences)


def inspect_render_frames(run_dir: str | Path) -> dict[str, Any]:
    """Inspect actual sampled PNGs without inferring prompt semantics."""
    root = Path(run_dir)
    paths = _sample_paths(root)
    if Image is None:
        return {
            "evidence_version": VISUAL_EVIDENCE_VERSION,
            "status": "unavailable",
            "reason": "Pillow is not installed in the evaluator runtime",
            "requested_count": len(paths),
            "readable_count": 0,
            "score": 0.0,
            "frame_metrics": [],
        }
    frame_metrics: list[dict[str, float | int]] = []
    images: list[Any] = []
    errors: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                images.append(image.copy())
                frame_metrics.append({"path": str(path.resolve()), **_frame_metrics(image)})
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    readable = len(images)
    requested = len(paths)
    if not frame_metrics:
        return {
            "evidence_version": VISUAL_EVIDENCE_VERSION,
            "status": "unavailable",
            "reason": "no readable sampled PNG frames",
            "requested_count": requested,
            "readable_count": 0,
            "errors": errors,
            "score": 0.0,
            "frame_metrics": [],
        }
    availability = _clamp(readable / max(1, requested))
    resolution = sum(_clamp(min(item["width"], item["height"]) / 512.0) for item in frame_metrics) / readable
    foreground = _clamp(sum(float(item["foreground_fraction"]) for item in frame_metrics) / readable / 0.18)
    edge = _clamp(sum(float(item["edge_density"]) for item in frame_metrics) / readable / 0.12)
    differences = [_frame_difference(images[index - 1], images[index]) for index in range(1, readable)]
    temporal_change = _clamp((sum(differences) / len(differences) if differences else 0.0) / 0.25)
    score = 100.0 * (
        0.20 * availability
        + 0.15 * resolution
        + 0.25 * foreground
        + 0.20 * edge
        + 0.20 * temporal_change
    )
    return {
        "evidence_version": VISUAL_EVIDENCE_VERSION,
        "status": "complete" if readable == requested else "partial",
        "requested_count": requested,
        "readable_count": readable,
        "errors": errors,
        "metrics": {
            "availability": round(availability * 100.0, 4),
            "resolution": round(resolution * 100.0, 4),
            "foreground_coverage": round(foreground * 100.0, 4),
            "edge_structure": round(edge * 100.0, 4),
            "temporal_change": round(temporal_change * 100.0, 4),
        },
        "score": round(score, 4),
        "frame_metrics": frame_metrics,
        "source": "actual_sampled_png_frames",
    }
