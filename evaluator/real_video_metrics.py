"""Evidence-bound scores computed from a decoded real proxy video.

This module is intentionally separate from the declarative plan evaluators.
The plan describes what should happen; this evaluator reads the encoded MP4
and the transforms observed by Blender while rendering to judge what actually
happened.  It is a local Codex-compatible visual review path, not an external
VLM endpoint and not a numeric substitute for a missing video.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


REAL_VIDEO_METRICS_VERSION = "real-video-metrics-v1-mp4-runtime-evidence"
LOCAL_REVIEW_SOURCE = "deterministic_video_proxy_metrics"
_TARGET_SIZE = (64, 64)
_CHANNEL_NAMES = ("visual_score", "physical_score", "trajectory_score", "camera_score")
_HUMAN_WORDS = {
    "actor",
    "boy",
    "child",
    "girl",
    "human",
    "man",
    "person",
    "people",
    "woman",
}
_MOTION_ACTIONS = {
    "carry",
    "climb",
    "drag",
    "drop",
    "fall",
    "give",
    "grasp",
    "handoff",
    "jump",
    "move",
    "pick",
    "place",
    "pour",
    "press",
    "push",
    "reach",
    "release",
    "rotate",
    "run",
    "slide",
    "sweep",
    "walk",
    "write",
}
_CONTACT_ACTIONS = {"attach", "carry", "grasp", "handoff", "press", "push", "reach", "touch"}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _cap_observed(value: float, *, cap: float = 95.0) -> float:
    """Keep local evidence conservative; a perfect plan is not perfect video."""

    return round(_clamp(min(float(value), cap)), 4)


def _pixel_values(image: Image.Image) -> list[tuple[int, int, int]]:
    resized = image.convert("RGB").resize(_TARGET_SIZE)
    getter = getattr(resized, "get_flattened_data", None)
    return list(getter() if getter is not None else resized.getdata())


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _as_vec(value: Any, length: int = 3) -> tuple[float, ...] | None:
    if not isinstance(value, (list, tuple)) or len(value) < length:
        return None
    try:
        return tuple(float(value[index]) for index in range(length))
    except (TypeError, ValueError):
        return None


def _distance(first: Iterable[float], second: Iterable[float]) -> float:
    return math.sqrt(sum((float(left) - float(right)) ** 2 for left, right in zip(first, second)))


def _lerp(first: tuple[float, ...], second: tuple[float, ...], amount: float) -> tuple[float, ...]:
    return tuple(left + (right - left) * amount for left, right in zip(first, second))


def _frame_metric(image: Image.Image) -> dict[str, float | int]:
    width, height = _TARGET_SIZE
    pixels = _pixel_values(image)
    border = [
        pixels[index]
        for index in range(len(pixels))
        if index < width or index >= len(pixels) - width or index % width in {0, width - 1}
    ]
    background = tuple(sum(pixel[channel] for pixel in border) / max(1, len(border)) for channel in range(3))
    distances = [math.sqrt(sum((pixel[channel] - background[channel]) ** 2 for channel in range(3))) for pixel in pixels]
    foreground = sum(distance > 18.0 for distance in distances) / max(1, len(distances))
    luminance = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in pixels]
    mean_luma = sum(luminance) / max(1, len(luminance))
    luma_std = math.sqrt(sum((value - mean_luma) ** 2 for value in luminance) / max(1, len(luminance)))
    gradients: list[float] = []
    for y in range(height):
        for x in range(width):
            index = y * width + x
            if x + 1 < width:
                gradients.append(abs(luminance[index] - luminance[index + 1]))
            if y + 1 < height:
                gradients.append(abs(luminance[index] - luminance[index + width]))
    edge_density = sum(value > 12.0 for value in gradients) / max(1, len(gradients))
    color_bins = len({(r // 32, g // 32, b // 32) for r, g, b in pixels})
    return {
        "width": int(image.width),
        "height": int(image.height),
        "foreground_fraction": foreground,
        "mean_luminance": mean_luma,
        "luminance_std": luma_std,
        "edge_density": edge_density,
        "color_bin_count": color_bins,
    }


def _frame_difference(left: Image.Image, right: Image.Image) -> float:
    first = _pixel_values(left)
    second = _pixel_values(right)
    return sum(
        sum(abs(a[channel] - b[channel]) for channel in range(3)) / (3.0 * 255.0)
        for a, b in zip(first, second)
    ) / max(1, len(first))


def _decode_mp4(path: Path, max_frames: int) -> tuple[list[Image.Image], dict[str, Any]]:
    try:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(reader)
        size = metadata.get("size") or (0, 0)
        width, height = int(size[0]), int(size[1])
        decoded: list[Image.Image] = []
        try:
            for raw in reader:
                decoded.append(Image.frombytes("RGB", (width, height), raw))
        finally:
            close = getattr(reader, "close", None)
            if close:
                close()
    except (ImportError, OSError, RuntimeError, ValueError, StopIteration) as exc:
        raise RuntimeError(f"mp4_decode_failed:{type(exc).__name__}:{exc}") from exc
    if not decoded:
        raise RuntimeError("mp4_decode_failed:no_frames")
    if len(decoded) <= max_frames:
        selected = decoded
    else:
        indices = sorted({round(index * (len(decoded) - 1) / max(1, max_frames - 1)) for index in range(max_frames)})
        selected = [decoded[index] for index in indices]
    return selected, {
        "decoded_frame_count": len(decoded),
        "sampled_frame_count": len(selected),
        "fps": float(metadata.get("fps") or 0.0),
        "duration_s": float(metadata.get("duration") or 0.0),
        "width": width,
        "height": height,
    }


def _runtime_observations(telemetry: dict[str, Any]) -> list[dict[str, Any]]:
    values = telemetry.get("runtime_observations")
    if not isinstance(values, list):
        return []
    observations = [value for value in values if isinstance(value, dict) and isinstance(value.get("frame"), int)]
    return sorted(observations, key=lambda value: int(value["frame"]))


def _entity_samples(observations: list[dict[str, Any]], entity_id: str) -> list[dict[str, Any]]:
    samples = []
    for observation in observations:
        entities = observation.get("entities") or {}
        entity = entities.get(entity_id)
        if isinstance(entity, dict):
            samples.append({"frame": int(observation["frame"]), **entity})
    return samples


def _planned_position(states: list[dict[str, Any]], frame: int) -> tuple[float, ...] | None:
    points = []
    for state in states:
        position = _as_vec(state.get("position"))
        if position is not None:
            try:
                points.append((int(state.get("frame")), position))
            except (TypeError, ValueError):
                continue
    points.sort(key=lambda item: item[0])
    if not points:
        return None
    if frame <= points[0][0]:
        return points[0][1]
    if frame >= points[-1][0]:
        return points[-1][1]
    for (first_frame, first), (second_frame, second) in zip(points, points[1:]):
        if first_frame <= frame <= second_frame:
            amount = (frame - first_frame) / max(1, second_frame - first_frame)
            return _lerp(first, second, amount)
    return points[-1][1]


def _entity_kind_map(contract: dict[str, Any]) -> dict[str, str]:
    return {str(item.get("id")): str(item.get("kind") or "prop") for item in contract.get("entities", []) if item.get("id")}


def _visible_fraction(samples: list[dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    return sum(_clamp(float(sample.get("visible_fraction", 0.0) or 0.0), 0.0, 1.0) for sample in samples) / len(samples)


def _trajectory_dimension(
    entity_ids: list[str],
    trajectory_entities: dict[str, Any],
    observations: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    if not entity_ids:
        return 100.0, ["not_applicable:no_entities_of_this_kind"]
    scores: list[float] = []
    evidence: list[str] = []
    for entity_id in entity_ids:
        samples = _entity_samples(observations, entity_id)
        data = _as_dict(trajectory_entities.get(entity_id))
        states = [item for item in data.get("states", []) if isinstance(item, dict)]
        if not samples or not states:
            scores.append(0.0)
            evidence.append(f"trajectory_missing:{entity_id}")
            continue
        errors = []
        actual_positions: list[tuple[float, ...]] = []
        expected_positions: list[tuple[float, ...]] = []
        for sample in samples:
            actual = _as_vec(sample.get("root_location"))
            expected = _planned_position(states, int(sample["frame"]))
            if actual is None or expected is None:
                continue
            actual_positions.append(actual)
            expected_positions.append(expected)
            errors.append(_distance(actual, expected))
        if not errors:
            scores.append(0.0)
            evidence.append(f"trajectory_unreadable:{entity_id}")
            continue
        position_score = 100.0 * math.exp(-sum(errors) / len(errors) / 0.45)
        expected_motion = _distance(expected_positions[0], expected_positions[-1]) if len(expected_positions) > 1 else 0.0
        actual_motion = _distance(actual_positions[0], actual_positions[-1]) if len(actual_positions) > 1 else 0.0
        motion_score = 100.0 * math.exp(-abs(actual_motion - expected_motion) / max(0.35, expected_motion + 0.35))
        visibility_score = 100.0 * _visible_fraction(samples)
        score = 0.60 * position_score + 0.25 * motion_score + 0.15 * visibility_score
        scores.append(_cap_observed(score))
        evidence.append(
            f"trajectory_observed:{entity_id}:frames={len(samples)}:mean_position_error={sum(errors) / len(errors):.4f}"
        )
    return round(sum(scores) / len(scores), 4), evidence


def _camera_coverage(
    contract: dict[str, Any], plan: dict[str, Any], observations: list[dict[str, Any]]
) -> tuple[float, list[str]]:
    events = [item for item in contract.get("events", []) if isinstance(item, dict)]
    if not events:
        return 100.0, ["camera_coverage:not_applicable:no_events"]
    scores: list[float] = []
    evidence: list[str] = []
    plan_events = {str(item.get("id")): item for item in plan.get("events", []) if isinstance(item, dict)}
    for event in events:
        event_id = str(event.get("id") or "event")
        start = int(round(float(event.get("start", 0.0)) * float(contract.get("fps") or 24))) + 1
        end = int(round(float(event.get("end", 0.0)) * float(contract.get("fps") or 24)))
        window = [item for item in observations if start <= int(item["frame"]) <= max(start, end)]
        planned_event = plan_events.get(event_id, {})
        target_ids = list(dict.fromkeys([*(event.get("target_ids") or []), *(planned_event.get("participant_ids") or [])]))
        if not target_ids:
            target_ids = [str(item.get("id")) for item in plan.get("entities", []) if item.get("id")]
        target_scores = []
        for entity_id in target_ids:
            samples = []
            for observation in window:
                entity = (observation.get("entities") or {}).get(entity_id)
                if isinstance(entity, dict):
                    samples.append(entity)
            target_scores.append(_visible_fraction(samples))
        event_score = 100.0 * (sum(target_scores) / len(target_scores) if target_scores else 0.0)
        scores.append(_cap_observed(event_score))
        evidence.append(f"camera_coverage_observed:{event_id}:frames={len(window)}:targets={len(target_ids)}")
    return round(sum(scores) / len(scores), 4), evidence


def _camera_innovation(plan: dict[str, Any], observations: list[dict[str, Any]]) -> tuple[float, list[str]]:
    shots = [item for item in (plan.get("camera") or {}).get("shots", []) if isinstance(item, dict)]
    if not shots:
        return 100.0, ["camera_innovation:not_applicable:no_shots"]
    scores: list[float] = []
    evidence: list[str] = []
    for shot in shots:
        start = int(shot.get("start_frame", 1) or 1)
        end = int(shot.get("end_frame", start) or start)
        frames = [item for item in observations if start <= int(item["frame"]) <= max(start, end)]
        locations = [_as_vec((item.get("camera") or {}).get("location")) for item in frames]
        locations = [item for item in locations if item is not None]
        path = sum(_distance(first, second) for first, second in zip(locations, locations[1:])) if len(locations) > 1 else 0.0
        trajectory_type = str(shot.get("trajectory_type") or shot.get("camera_cue") or "static").lower()
        if trajectory_type == "orbit":
            angles = [math.atan2(location[1], location[0]) for location in locations]
            span = (max(angles) - min(angles)) if angles else 0.0
            score = 100.0 * min(abs(span) / math.radians(30.0), 1.0)
        elif trajectory_type in {"dolly", "zoom"}:
            distances = [math.sqrt(sum(value * value for value in location)) for location in locations]
            score = 100.0 * min(abs((distances[-1] - distances[0])) / 0.75, 1.0) if len(distances) > 1 else 0.0
        elif trajectory_type in {"pan", "tilt", "follow"}:
            score = 100.0 * min(path / 1.0, 1.0)
            if trajectory_type == "follow" and path < 0.2:
                score = 70.0
        else:
            score = 100.0 if path < 0.25 else 75.0
        scores.append(_cap_observed(score))
        evidence.append(f"camera_motion_observed:{shot.get('shot_id', 'shot')}:type={trajectory_type}:path={path:.4f}")
    return round(sum(scores) / len(scores), 4), evidence


def _motion_smoothness(observations: list[dict[str, Any]], entity_ids: list[str]) -> tuple[float, list[str]]:
    velocities: list[float] = []
    for entity_id in [*entity_ids, "__camera__"]:
        positions: list[tuple[int, tuple[float, ...]]] = []
        for observation in observations:
            if entity_id == "__camera__":
                position = _as_vec((observation.get("camera") or {}).get("location"))
            else:
                position = _as_vec(((observation.get("entities") or {}).get(entity_id) or {}).get("root_location"))
            if position is not None:
                positions.append((int(observation["frame"]), position))
        velocities.extend(
            _distance(first[1], second[1]) / max(1, second[0] - first[0])
            for first, second in zip(positions, positions[1:])
        )
    if len(velocities) < 2:
        return 0.0, ["temporal_smoothness:insufficient_runtime_positions"]
    median = sorted(velocities)[len(velocities) // 2]
    spikes = [value for value in velocities if value > max(0.05, median * 3.0)]
    score = 100.0 * max(0.0, 1.0 - len(spikes) / len(velocities) * 1.6)
    return _cap_observed(score), [f"temporal_smoothness_observed:velocity_samples={len(velocities)}:spikes={len(spikes)}"]


def _event_timing(contract: dict[str, Any], plan: dict[str, Any], observations: list[dict[str, Any]]) -> tuple[float, list[str]]:
    events = [item for item in contract.get("events", []) if isinstance(item, dict)]
    if not events:
        return 100.0, ["event_timing:not_applicable:no_events"]
    plan_events = {str(item.get("id")): item for item in plan.get("events", []) if isinstance(item, dict)}
    scores: list[float] = []
    evidence: list[str] = []
    for event in events:
        event_id = str(event.get("id") or "event")
        fps = float(contract.get("fps") or 24)
        start = int(round(float(event.get("start", 0.0)) * fps)) + 1
        end = int(round(float(event.get("end", 0.0)) * fps))
        window = [item for item in observations if start <= int(item["frame"]) <= max(start, end)]
        target_ids = list(event.get("target_ids") or [])
        planned = plan_events.get(event_id, {})
        action = str(planned.get("action") or event.get("description") or "observe").lower()
        motion: list[float] = []
        for entity_id in target_ids:
            samples = _entity_samples(observations, str(entity_id))
            samples = [sample for sample in samples if start <= sample["frame"] <= max(start, end)]
            positions = [_as_vec(sample.get("root_location")) for sample in samples]
            positions = [position for position in positions if position is not None]
            motion.extend(_distance(first, second) for first, second in zip(positions, positions[1:]))
        activity = sum(motion) / len(motion) if motion else 0.0
        if action in _MOTION_ACTIONS:
            score = 100.0 * min(activity / 0.35, 1.0)
        else:
            score = 100.0 * max(0.0, 1.0 - activity / 0.5)
        scores.append(_cap_observed(score))
        evidence.append(f"event_timing_observed:{event_id}:action={action}:activity={activity:.4f}")
    return round(sum(scores) / len(scores), 4), evidence


def _bbox_volume(bounds: Any) -> float:
    if not isinstance(bounds, list) or len(bounds) != 2:
        return 0.0
    first, second = _as_vec(bounds[0]), _as_vec(bounds[1])
    if first is None or second is None:
        return 0.0
    return max(0.0, second[0] - first[0]) * max(0.0, second[1] - first[1]) * max(0.0, second[2] - first[2])


def _bbox_overlap_ratio(first: Any, second: Any) -> float:
    if not isinstance(first, list) or not isinstance(second, list) or len(first) != 2 or len(second) != 2:
        return 0.0
    first_min, first_max = _as_vec(first[0]), _as_vec(first[1])
    second_min, second_max = _as_vec(second[0]), _as_vec(second[1])
    if None in (first_min, first_max, second_min, second_max):
        return 0.0
    overlap = 1.0
    for axis in range(3):
        overlap *= max(0.0, min(first_max[axis], second_max[axis]) - max(first_min[axis], second_min[axis]))
    denominator = min(_bbox_volume(first), _bbox_volume(second))
    return overlap / denominator if denominator > 1e-9 else 0.0


def _physical_scores(
    contract: dict[str, Any], telemetry: dict[str, Any], observations: list[dict[str, Any]], entity_kind: dict[str, str], smoothness: float
) -> tuple[float, float, list[str]]:
    contact_pairs: set[frozenset[str]] = set()
    for event in contract.get("events", []):
        if not isinstance(event, dict) or str(event.get("description") or "").lower() not in _CONTACT_ACTIONS:
            continue
        ids = [str(item) for item in event.get("target_ids") or []]
        for first in ids:
            for second in ids:
                if first < second:
                    contact_pairs.add(frozenset((first, second)))
    checked_pairs = 0
    collision_pairs = 0
    below_ground = 0
    for observation in observations:
        entities = observation.get("entities") or {}
        ids = [str(item) for item in entities if entity_kind.get(str(item)) not in {"support", "environment"}]
        for index, first in enumerate(ids):
            first_data = entities.get(first) or {}
            bounds = first_data.get("world_bbox")
            if isinstance(bounds, list) and len(bounds) == 2 and _as_vec(bounds[0]) is not None and _as_vec(bounds[0])[2] < -0.05:
                below_ground += 1
            for second in ids[index + 1 :]:
                checked_pairs += 1
                if frozenset((first, second)) in contact_pairs:
                    continue
                if _bbox_overlap_ratio(bounds, (entities.get(second) or {}).get("world_bbox")) > 0.12:
                    collision_pairs += 1
    collision_rate = collision_pairs / max(1, checked_pairs)
    below_rate = below_ground / max(1, len(observations) * max(1, len(entity_kind)))
    rig_values = [value for value in (telemetry.get("rigs") or {}).values() if isinstance(value, dict)]
    disconnected_rigs = sum(not bool(value.get("connected")) for value in rig_values)
    rig_rate = disconnected_rigs / max(1, len(rig_values))
    physical = 100.0 - 55.0 * collision_rate - 25.0 * min(below_rate, 1.0) - 25.0 * rig_rate
    physical = 0.75 * physical + 0.25 * smoothness
    realism = 0.70 * physical + 0.30 * smoothness
    evidence = [
        f"physical_collision_observed:checked_pairs={checked_pairs}:unexpected_overlaps={collision_pairs}",
        f"physical_ground_observed:below_ground_frames={below_ground}",
        f"physical_rig_observed:disconnected_rigs={disconnected_rigs}",
    ]
    return _cap_observed(physical), _cap_observed(realism), evidence


def _visual_scores(frames: list[Image.Image]) -> tuple[dict[str, float], list[str]]:
    metrics = [_frame_metric(frame) for frame in frames]
    differences = [_frame_difference(left, right) for left, right in zip(frames, frames[1:])]
    foreground = sum(float(item["foreground_fraction"]) for item in metrics) / len(metrics)
    edge = sum(float(item["edge_density"]) for item in metrics) / len(metrics)
    contrast = sum(float(item["luminance_std"]) for item in metrics) / len(metrics)
    colors = sum(float(item["color_bin_count"]) for item in metrics) / len(metrics)
    resolution = min(float(metrics[0]["width"]), float(metrics[0]["height"])) / 256.0
    motion = sum(differences) / len(differences) if differences else 0.0
    static_penalty = 30.0 if differences and max(differences) < 0.001 else 0.0
    clarity = 100.0 * (0.30 * min(foreground / 0.22, 1.0) + 0.35 * min(edge / 0.16, 1.0) + 0.20 * min(contrast / 45.0, 1.0) + 0.15 * min(resolution, 1.0))
    appearance = 100.0 * (0.40 * min(edge / 0.16, 1.0) + 0.35 * min(contrast / 45.0, 1.0) + 0.25 * min(colors / 80.0, 1.0))
    presentation = 100.0 * (0.45 * min(foreground / 0.22, 1.0) + 0.25 * min(edge / 0.16, 1.0) + 0.30 * min(contrast / 45.0, 1.0)) - static_penalty
    smooth = 100.0 * max(0.0, 1.0 - (math.sqrt(sum((value - motion) ** 2 for value in differences) / len(differences)) / max(motion, 0.02) if differences else 0.0))
    values = {
        "visual_clarity": _cap_observed(clarity),
        "appearance_detail": _cap_observed(appearance),
        "visual_presentation": _cap_observed(presentation),
        "frame_temporal_smoothness": _cap_observed(smooth),
    }
    evidence = [
        f"mp4_frame_metrics:frames={len(frames)}:mean_foreground={foreground:.4f}:edge_density={edge:.4f}:contrast={contrast:.4f}",
        f"mp4_temporal_metrics:frame_difference={motion:.6f}:static_penalty={static_penalty:.1f}",
    ]
    return values, evidence


def _spatial_consistency(entity_ids: list[str], observations: list[dict[str, Any]]) -> tuple[float, list[str]]:
    if not entity_ids:
        return 100.0, ["spatial_consistency:not_applicable:no_entities"]
    values: list[float] = []
    for entity_id in entity_ids:
        samples = _entity_samples(observations, entity_id)
        valid = 0
        clipped = 0
        for sample in samples:
            bbox = sample.get("screen_bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                valid += 1
                if min(float(value) for value in bbox) >= -0.05 and max(float(value) for value in bbox) <= 1.05:
                    clipped += 1
        values.append(100.0 * clipped / max(1, valid))
    return _cap_observed(sum(values) / len(values)), [f"screen_bbox_observed:entities={len(entity_ids)}"]


def evaluate_real_video(
    run_dir: str | Path,
    *,
    prompt: str,
    scene_contract: Any,
    trajectory_plan: Any,
    telemetry: dict[str, Any],
    max_frames: int = 8,
) -> dict[str, Any]:
    """Evaluate a real ``proxy.mp4`` plus Blender runtime observations.

    No numeric result is returned when either evidence source is missing.  In
    particular, a valid plan, PNG files, or a non-empty MP4 alone cannot
    create a visual/physics/trajectory score.
    """

    root = Path(run_dir)
    empty_channels = {name: None for name in _CHANNEL_NAMES}
    video_path = root / "proxy.mp4"
    observations = _runtime_observations(telemetry or {})
    if not observations:
        return {
            "evaluator_version": REAL_VIDEO_METRICS_VERSION,
            "status": "unavailable",
            "reason": "runtime_observations_missing",
            "source": "actual_proxy_mp4_and_runtime_observations",
            "channels": empty_channels,
            "dimensions": {},
        }
    try:
        frames, decoded = _decode_mp4(video_path, max_frames)
    except RuntimeError as exc:
        return {
            "evaluator_version": REAL_VIDEO_METRICS_VERSION,
            "status": "unavailable",
            "reason": str(exc),
            "source": "actual_proxy_mp4_and_runtime_observations",
            "channels": empty_channels,
            "dimensions": {},
        }
    contract = _as_dict(scene_contract)
    plan = _as_dict(trajectory_plan)
    telemetry = telemetry or {}
    entity_kind = _entity_kind_map(contract)
    trajectory_entities = _as_dict(plan.get("entities"))
    all_ids = [str(item) for item in entity_kind]
    actor_ids = [item for item in all_ids if entity_kind.get(item) in {"character", "actor"}]
    prop_ids = [item for item in all_ids if entity_kind.get(item) == "prop"]
    visual, visual_evidence = _visual_scores(frames)
    object_trajectory, object_evidence = _trajectory_dimension(prop_ids, trajectory_entities, observations)
    character_trajectory, character_evidence = _trajectory_dimension(actor_ids, trajectory_entities, observations)
    timing, timing_evidence = _event_timing(contract, plan, observations)
    smoothness, smoothness_evidence = _motion_smoothness(observations, all_ids)
    camera_coverage, coverage_evidence = _camera_coverage(contract, plan, observations)
    camera_innovation, camera_evidence = _camera_innovation(plan, observations)
    camera_score = _cap_observed((camera_coverage + camera_innovation) / 2.0)
    trajectory_score = _cap_observed((object_trajectory + character_trajectory + timing + smoothness) / 4.0)
    physical, physical_realism, physical_evidence = _physical_scores(contract, telemetry, observations, entity_kind, smoothness)
    spatial = _spatial_consistency(all_ids, observations)[0]
    visible_entities = [_visible_fraction(_entity_samples(observations, entity_id)) for entity_id in all_ids if entity_kind.get(entity_id) not in {"support", "environment"}]
    entity_visibility = 100.0 * (sum(visible_entities) / len(visible_entities) if visible_entities else 0.0)
    prompt_tokens = {token.strip(".,;:!?()[]{}\"'").lower() for token in prompt.split()}
    plan_actor_without_evidence = bool(actor_ids) and not (prompt_tokens & _HUMAN_WORDS)
    compliance = min(90.0, 0.65 * entity_visibility + 0.35 * timing)
    if plan_actor_without_evidence:
        compliance -= 30.0
    compliance = _cap_observed(compliance, cap=90.0)
    physical_plausibility = physical
    motion_naturalness = _cap_observed((smoothness + timing) / 2.0)
    dimensions = {
        "prompt_compliance": compliance,
        "physical_plausibility": physical_plausibility,
        "camera_coverage": camera_coverage,
        "camera_innovation": camera_innovation,
        "character_trajectory": character_trajectory,
        "object_trajectory": object_trajectory,
        "event_timing": timing,
        "temporal_smoothness": smoothness,
        "visual_clarity": visual["visual_clarity"],
        "appearance_detail": visual["appearance_detail"],
        "physical_realism": physical_realism,
        "spatial_consistency": spatial,
        "motion_naturalness": motion_naturalness,
        "visual_presentation": visual["visual_presentation"],
    }
    weaknesses: list[str] = []
    for name, value in dimensions.items():
        if float(value) < 55.0:
            weaknesses.append(f"{name}={value}")
    if plan_actor_without_evidence:
        weaknesses.append("object-only prompt appears to contain an ungrounded actor in the executable contract")
    evidence = [
        f"proxy_mp4:{video_path.resolve()}",
        f"runtime_observations:{len(observations)}:frames={observations[0]['frame']}-{observations[-1]['frame']}",
        *visual_evidence,
        *object_evidence,
        *character_evidence,
        *timing_evidence,
        *smoothness_evidence,
        *coverage_evidence,
        *camera_evidence,
        *physical_evidence,
    ]
    return {
        "evaluator_version": REAL_VIDEO_METRICS_VERSION,
        "status": "scored",
        "source": "actual_proxy_mp4_and_runtime_observations",
        "review_source": LOCAL_REVIEW_SOURCE,
        "review_method": "decoded-mp4-runtime-measurements-v2",
        "confidence": 0.78,
        "prompt": prompt,
        "decoded_video": decoded,
        "runtime_observations": {
            "count": len(observations),
            "first_frame": int(observations[0]["frame"]),
            "last_frame": int(observations[-1]["frame"]),
            "entities": all_ids,
        },
        "dimensions": dimensions,
        "channels": {
            "visual_score": _cap_observed((float(visual["visual_clarity"]) + float(visual["appearance_detail"]) + float(visual["visual_presentation"])) / 3.0),
            "physical_score": physical,
            "trajectory_score": trajectory_score,
            "camera_score": camera_score,
        },
        "visible_evidence": evidence,
        "weaknesses": weaknesses,
        "policy": {
            "plan_is_not_a_visual_score": True,
            "mp4_must_decode": True,
            "runtime_observations_must_exist": True,
            "deterministic_metrics_are_not_vlm": True,
            "local_score_cap": 95.0,
            "semantic_uncertainty_cap": 90.0,
        },
    }


def build_local_vlm_response(result: dict[str, Any]) -> Any:
    """Convert the evidence report to the shared VLM response contract."""

    from .schemas import VLMJudgeResponse

    dimensions = result.get("dimensions") or {}
    return VLMJudgeResponse(
        **{name: float(dimensions[name]) for name in (
            "prompt_compliance",
            "physical_plausibility",
            "camera_coverage",
            "camera_innovation",
            "character_trajectory",
            "object_trajectory",
            "event_timing",
            "temporal_smoothness",
            "visual_clarity",
            "appearance_detail",
            "physical_realism",
            "spatial_consistency",
            "motion_naturalness",
            "visual_presentation",
        )},
        visible_evidence=list(result.get("visible_evidence") or ["local video evidence"]),
        weaknesses=list(result.get("weaknesses") or []),
        confidence=float(result.get("confidence") or 0.0),
    )


__all__ = [
    "LOCAL_REVIEW_SOURCE",
    "REAL_VIDEO_METRICS_VERSION",
    "build_local_vlm_response",
    "evaluate_real_video",
]
