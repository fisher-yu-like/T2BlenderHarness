"""Pure constraint specifications consumed by the Blender compiler."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import register_primitive


@dataclass(frozen=True)
class ConstraintSpec:
    """Serializable description of a Blender constraint and its animation."""

    type: str
    object_id: str
    target: str
    subtarget: str | None = None
    influence_keyframes: list[tuple[int, float]] = field(default_factory=list)
    track_axis: str | None = None
    up_axis: str | None = None
    chain_length: int | None = None


def _validate_keyframes(keyframes: list[tuple[int, float]] | tuple[tuple[int, float], ...]) -> list[tuple[int, float]]:
    normalized = [(int(frame), float(value)) for frame, value in keyframes]
    if any(value < 0.0 or value > 1.0 for _frame, value in normalized):
        raise ValueError("constraint influence must be in [0, 1]")
    if any(normalized[index][0] > normalized[index + 1][0] for index in range(len(normalized) - 1)):
        raise ValueError("constraint keyframes must be ordered by frame")
    return normalized


@register_primitive(category="constraints", tags=["attachment", "handoff"], cost_estimate="low", example_usage="child_of_constraint('cup', 'rig', 'hand.R', [(1, 0), (10, 1)])")
def child_of_constraint(
    obj: str,
    target: str,
    subtarget: str,
    influence_keyframes: list[tuple[int, float]] | tuple[tuple[int, float], ...],
) -> ConstraintSpec:
    """Attach ``obj`` to ``target.subtarget`` with animated influence."""

    if not obj or not target or not subtarget:
        raise ValueError("object, target, and subtarget are required")
    return ConstraintSpec(
        type="CHILD_OF",
        object_id=obj,
        target=target,
        subtarget=subtarget,
        influence_keyframes=_validate_keyframes(influence_keyframes),
    )


@register_primitive(category="constraints", tags=["camera", "orientation"], cost_estimate="low", example_usage="track_to_constraint('camera', 'target')")
def track_to_constraint(obj: str, target: str, track_axis: str = "-Z", up_axis: str = "Y") -> ConstraintSpec:
    """Orient ``obj`` toward ``target`` using Track To-compatible axes."""

    if not obj or not target:
        raise ValueError("object and target are required")
    if track_axis not in {"X", "-X", "Y", "-Y", "Z", "-Z"}:
        raise ValueError("invalid track axis")
    if up_axis not in {"X", "-X", "Y", "-Y", "Z", "-Z"}:
        raise ValueError("invalid up axis")
    return ConstraintSpec(
        type="TRACK_TO",
        object_id=obj,
        target=target,
        track_axis=track_axis,
        up_axis=up_axis,
    )


__all__ = ["ConstraintSpec", "child_of_constraint", "track_to_constraint"]
