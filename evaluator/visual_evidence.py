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
RENDER_FRAME_HEALTH_VERSION = "render-frame-health-v1"
_TARGET_SIZE = (64, 64)
FRAME_STATISTICS_MEASURABLE_DIMENSIONS = (
    "visual_clarity",
    "temporal_smoothness",
    "appearance_detail",
    "spatial_consistency",
    "visual_presentation",
)
FRAME_STATISTICS_UNOBSERVABLE_DIMENSIONS = (
    "prompt_compliance",
    "physical_plausibility",
    "camera_coverage",
    "camera_innovation",
    "character_trajectory",
    "object_trajectory",
    "event_timing",
    "physical_realism",
    "motion_naturalness",
)


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
        "mean_luminance": round(mean_luminance, 6),
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


def score_sample_frames(frame_paths: list[str | Path]) -> dict[str, Any]:
    """Return low-level frame observations without semantic quality claims."""
    paths = [Path(path) for path in frame_paths]
    if Image is None:
        return {
            "evidence_version": VISUAL_EVIDENCE_VERSION,
            "status": "unavailable",
            "review_source": "frame_statistics",
            "method": "frame_statistics_only-v1",
            "scores": {
                **{name: None for name in FRAME_STATISTICS_UNOBSERVABLE_DIMENSIONS},
                **{name: None for name in FRAME_STATISTICS_MEASURABLE_DIMENSIONS},
            },
            "score": None,
            "artifact_health": {"readable": False, "all_black": None, "all_static": None, "status": "unavailable"},
            "frame_metrics": {"frame_count": 0, "requested_count": len(paths), "errors": ["Pillow is not installed"]},
        }
    metrics: list[dict[str, float | int]] = []
    images: list[Any] = []
    errors: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                images.append(image.copy())
                metrics.append({"path": str(path.resolve()), **_frame_metrics(image)})
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not metrics:
        return {
            "evidence_version": VISUAL_EVIDENCE_VERSION,
            "status": "unavailable",
            "review_source": "frame_statistics",
            "method": "frame_statistics_only-v1",
            "scores": {
                **{name: None for name in FRAME_STATISTICS_UNOBSERVABLE_DIMENSIONS},
                **{name: None for name in FRAME_STATISTICS_MEASURABLE_DIMENSIONS},
            },
            "score": None,
            "artifact_health": {"readable": False, "all_black": None, "all_static": None, "status": "unavailable"},
            "frame_metrics": {"frame_count": 0, "requested_count": len(paths), "errors": errors},
        }
    foreground = sum(float(item["foreground_fraction"]) for item in metrics) / len(metrics)
    edge = sum(float(item["edge_density"]) for item in metrics) / len(metrics)
    luma = sum(float(item["luminance_std"]) for item in metrics) / len(metrics)
    differences = [_frame_difference(images[index - 1], images[index]) for index in range(1, len(images))]
    mean_difference = sum(differences) / len(differences) if differences else 0.0
    variation = (
        math.sqrt(sum((value - mean_difference) ** 2 for value in differences) / len(differences))
        if differences
        else 0.0
    )
    smooth = _clamp(1.0 - variation / max(mean_difference, 0.02))
    foreground_variation = math.sqrt(
        sum((float(item["foreground_fraction"]) - foreground) ** 2 for item in metrics) / len(metrics)
    )
    occupancy = _clamp(1.0 - foreground_variation / max(foreground, 0.05))
    foreground_unit = _clamp(foreground / 0.35)
    edge_unit = _clamp(edge / 0.20)
    luma_unit = _clamp(luma / 55.0)
    scores = {
        "visual_clarity": 100.0 * _clamp(0.30 * foreground_unit + 0.45 * edge_unit + 0.25 * luma_unit),
        "temporal_smoothness": 100.0 * smooth,
        "appearance_detail": 100.0 * _clamp(0.55 * edge_unit + 0.45 * luma_unit),
        "spatial_consistency": 100.0 * occupancy,
        "visual_presentation": 100.0 * _clamp(0.45 * foreground_unit + 0.25 * edge_unit + 0.30 * occupancy),
    }
    all_black = all(float(item["mean_luminance"]) < 2.0 for item in metrics)
    all_static = bool(differences) and max(differences) < 0.001
    return {
        "evidence_version": VISUAL_EVIDENCE_VERSION,
        "status": "complete" if len(metrics) == len(paths) else "partial",
        "review_source": "frame_statistics",
        "method": "frame_statistics_only-v1",
        "scores": {
            **{name: None for name in FRAME_STATISTICS_UNOBSERVABLE_DIMENSIONS},
            **{name: round(value, 4) for name, value in scores.items()},
        },
        "score": None,
        "visible_evidence": [
            f"frame_statistics:{path.resolve()}" for path in paths if path.exists()
        ],
        "weaknesses": [
            "frame statistics cannot establish prompt compliance, physical plausibility, or choreography",
        ],
        "artifact_health": {
            "readable": len(metrics) == len(paths),
            "all_black": all_black,
            "all_static": all_static,
            "status": "complete" if len(metrics) == len(paths) else "partial",
        },
        "frame_metrics": {
            "frame_count": len(metrics),
            "requested_count": len(paths),
            "errors": errors,
            "mean_foreground_fraction": round(foreground, 6),
            "mean_edge_density": round(edge, 6),
            "mean_luminance_std": round(luma, 6),
            "mean_frame_difference": round(mean_difference, 6),
            "temporal_smoothness_observation": round(smooth * 100.0, 4),
            "occupancy_consistency_observation": round(occupancy * 100.0, 4),
        },
        "sampled_frames": [str(path.resolve()) for path in paths],
    }


def assess_render_frame_health(frame_paths: list[str | Path]) -> dict[str, Any]:
    """Detect unreadable or spatially blank render samples before VLM review.

    A uniform frame is not a semantic quality score. It is an artifact-level
    failure signal: a camera pointed at an empty background gives a visual
    judge no evidence to inspect, so the real inner loop should regenerate the
    candidate instead of recording an ambiguous VLM review.
    """

    paths = [Path(path) for path in frame_paths]
    if Image is None:
        return {
            "health_version": RENDER_FRAME_HEALTH_VERSION,
            "status": "unavailable",
            "reason": "Pillow is not installed in the evaluator runtime",
            "requested_count": len(paths),
            "readable_count": 0,
            "frame_metrics": [],
        }
    metrics: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.load()
                metrics.append({"path": str(path.resolve()), **_frame_metrics(image)})
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    if not metrics:
        return {
            "health_version": RENDER_FRAME_HEALTH_VERSION,
            "status": "unavailable",
            "reason": "no readable render frames",
            "requested_count": len(paths),
            "readable_count": 0,
            "errors": errors,
            "frame_metrics": [],
        }
    # A low-variance image with no foreground pixels and no edges is the
    # characteristic output of an incorrectly aimed or clipped camera. Keep
    # the thresholds deliberately conservative so ordinary low-detail scenes
    # remain available to the independent VLM judge.
    blank = len(metrics) == len(paths) and all(
        float(item["luminance_std"]) <= 1.5
        and float(item["foreground_fraction"]) <= 0.001
        and float(item["edge_density"]) <= 0.001
        for item in metrics
    )
    return {
        "health_version": RENDER_FRAME_HEALTH_VERSION,
        "status": "blank" if blank else ("visible" if len(metrics) == len(paths) else "partial"),
        "reason": "uniform_spatially_empty_frames" if blank else None,
        "requested_count": len(paths),
        "readable_count": len(metrics),
        "errors": errors,
        "frame_metrics": metrics,
    }


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
            "score": None,
            "score_kind": "artifact_health_only",
            "artifact_health": {
                "readable": False,
                "all_black": None,
                "all_static": None,
                "status": "unavailable",
            },
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
    all_black = all(float(item["mean_luminance"]) < 2.0 for item in frame_metrics)
    all_static = bool(differences) and max(differences) < 0.001
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
        "score": None,
        "score_kind": "artifact_health_only",
        "review_source": "frame_statistics",
        "artifact_health": {
            "readable": readable == requested,
            "all_black": all_black,
            "all_static": all_static,
            "status": "complete" if readable == requested else "partial",
        },
        "frame_metrics": frame_metrics,
        "source": "actual_sampled_png_frames",
    }
