"""Independent physics and ownership checks over trusted raw observations.

The observer emits transforms, bounds, and pose-bone coordinates only.  This
module derives contact and ownership from those observations; it never reads
semantic booleans written by a generated Blender job.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


PHYSICS_ORACLE_VERSION = "physics-oracle-v2-obb-bvh-contact-ownership"
_CONTACT_DISTANCE = 0.85
_MAX_STEP_DISTANCE = 3.0
_MAX_ACCELERATION = 80.0
_PENETRATION_EPSILON = 0.04
_CONTACT_PENETRATION_LIMIT = 0.18


def _finding(failure_id: str, message: str, evidence: list[str], *, severity: str = "hard") -> dict[str, Any]:
    return {
        "failure_id": failure_id,
        "owner": "proxy_renderer",
        "category": "physics_ownership_oracle",
        "severity": severity,
        "root_cause_id": f"physics:{failure_id}",
        "message": message,
        "evidence": evidence,
        "repair_route": "runtime_repair",
    }


def _vec(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        result = tuple(float(value[index]) for index in range(3))
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _unit(value: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = math.sqrt(sum(item * item for item in value))
    if length <= 1e-9:
        return None
    return tuple(item / length for item in value)


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _obb(entity: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = entity.get("obb")
    if isinstance(raw, Mapping):
        center = _vec(raw.get("center"))
        half = _vec(raw.get("half_extents"))
        axes = raw.get("axes")
        if center and half and isinstance(axes, list) and len(axes) >= 3:
            normalised = [_unit(_vec(axis) or (0.0, 0.0, 0.0)) for axis in axes[:3]]
            if all(axis is not None for axis in normalised) and all(value >= 0 for value in half):
                return {"center": center, "axes": normalised, "half": half}
    bounds = entity.get("world_bounds")
    if isinstance(bounds, Mapping):
        lower = _vec(bounds.get("min"))
        upper = _vec(bounds.get("max"))
    elif isinstance(entity.get("world_bbox"), list) and len(entity["world_bbox"]) >= 2:
        lower = _vec(entity["world_bbox"][0])
        upper = _vec(entity["world_bbox"][1])
    else:
        lower = upper = None
    if lower and upper:
        low = tuple(min(a, b) for a, b in zip(lower, upper))
        high = tuple(max(a, b) for a, b in zip(lower, upper))
        return {
            "center": tuple((a + b) / 2.0 for a, b in zip(low, high)),
            "axes": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            "half": tuple((b - a) / 2.0 for a, b in zip(low, high)),
        }
    location = _vec(entity.get("location") or entity.get("root_location"))
    if location:
        return {"center": location, "axes": [(1.0, 0.0, 0.0)] * 3, "half": (0.0, 0.0, 0.0)}
    return None


def _obb_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Return the minimum separating-axis overlap depth, or zero."""

    axes_a = first["axes"]
    axes_b = second["axes"]
    axes: list[tuple[float, float, float]] = [*axes_a, *axes_b]
    for axis_a in axes_a:
        for axis_b in axes_b:
            cross = _unit(_cross(axis_a, axis_b))
            if cross is not None:
                axes.append(cross)
    delta = tuple(a - b for a, b in zip(first["center"], second["center"]))
    minimum = float("inf")
    for axis in axes:
        radius_a = sum(abs(_dot(axis, basis)) * extent for basis, extent in zip(axes_a, first["half"]))
        radius_b = sum(abs(_dot(axis, basis)) * extent for basis, extent in zip(axes_b, second["half"]))
        overlap = radius_a + radius_b - abs(_dot(delta, axis))
        if overlap <= 0.0:
            return 0.0
        minimum = min(minimum, overlap)
    return 0.0 if minimum == float("inf") else minimum


Triangle = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _mesh_triangles(entity: Mapping[str, Any] | None) -> list[Triangle] | None:
    if not isinstance(entity, Mapping) or "mesh_triangles" not in entity:
        return None
    raw = entity.get("mesh_triangles")
    if not isinstance(raw, list):
        return []
    triangles: list[Triangle] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        points = tuple(_vec(point) for point in item)
        if len(points) == 3 and all(point is not None for point in points):
            triangles.append((points[0], points[1], points[2]))  # type: ignore[arg-type]
    return triangles


def _triangle_bounds(triangle: Triangle) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        tuple(min(point[index] for point in triangle) for index in range(3)),
        tuple(max(point[index] for point in triangle) for index in range(3)),
    )


def _bounds_overlap(
    first: tuple[tuple[float, float, float], tuple[float, float, float]],
    second: tuple[tuple[float, float, float], tuple[float, float, float]],
    epsilon: float = 1e-7,
) -> bool:
    return all(first[0][index] <= second[1][index] + epsilon and second[0][index] <= first[1][index] + epsilon for index in range(3))


@dataclass(frozen=True)
class _BVHNode:
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    triangles: tuple[Triangle, ...] = ()
    left: "_BVHNode | None" = None
    right: "_BVHNode | None" = None

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.lower, self.upper


def _build_bvh(triangles: list[Triangle]) -> _BVHNode:
    if not triangles:
        raise ValueError("cannot build a BVH without triangles")
    bounds = [_triangle_bounds(triangle) for triangle in triangles]
    lower = tuple(min(item[0][index] for item in bounds) for index in range(3))
    upper = tuple(max(item[1][index] for item in bounds) for index in range(3))
    if len(triangles) <= 8:
        return _BVHNode(lower=lower, upper=upper, triangles=tuple(triangles))
    axis = max(range(3), key=lambda index: upper[index] - lower[index])
    ordered = sorted(
        triangles,
        key=lambda triangle: sum(point[axis] for point in triangle) / 3.0,
    )
    middle = len(ordered) // 2
    return _BVHNode(
        lower=lower,
        upper=upper,
        left=_build_bvh(ordered[:middle]),
        right=_build_bvh(ordered[middle:]),
    )


def _triangle_intersects(first: Triangle, second: Triangle) -> bool:
    """Triangle SAT test used only after OBB broad-phase overlap."""

    first_edges = [
        tuple(first[index][axis] - first[(index + 1) % 3][axis] for axis in range(3))
        for index in range(3)
    ]
    second_edges = [
        tuple(second[index][axis] - second[(index + 1) % 3][axis] for axis in range(3))
        for index in range(3)
    ]
    axes: list[tuple[float, float, float]] = [
        _cross(first_edges[0], first_edges[1]),
        _cross(second_edges[0], second_edges[1]),
    ]
    axes.extend(_cross(edge, normal) for edge in first_edges for normal in axes[:1])
    axes.extend(_cross(edge, normal) for edge in second_edges for normal in axes[1:2])
    axes.extend(_cross(left, right) for left in first_edges for right in second_edges)
    first_bounds = _triangle_bounds(first)
    second_bounds = _triangle_bounds(second)
    if not _bounds_overlap(first_bounds, second_bounds):
        return False
    for raw_axis in axes:
        axis = _unit(raw_axis)
        if axis is None:
            continue
        left_projection = [_dot(point, axis) for point in first]
        right_projection = [_dot(point, axis) for point in second]
        if max(left_projection) < min(right_projection) - _PENETRATION_EPSILON or max(right_projection) < min(left_projection) - _PENETRATION_EPSILON:
            return False
    return True


def _bvh_intersects(first: _BVHNode, second: _BVHNode) -> bool:
    if not _bounds_overlap(first.bounds, second.bounds):
        return False
    if first.triangles and second.triangles:
        return any(_triangle_intersects(left, right) for left in first.triangles for right in second.triangles)
    if first.triangles:
        return bool(
            (second.left and _bvh_intersects(first, second.left))
            or (second.right and _bvh_intersects(first, second.right))
        )
    if second.triangles:
        return bool(
            (first.left and _bvh_intersects(first.left, second))
            or (first.right and _bvh_intersects(first.right, second))
        )
    return bool(
        (first.left and second.left and _bvh_intersects(first.left, second.left))
        or (first.left and second.right and _bvh_intersects(first.left, second.right))
        or (first.right and second.left and _bvh_intersects(first.right, second.left))
        or (first.right and second.right and _bvh_intersects(first.right, second.right))
    )


def _mesh_bvh_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool | None:
    first_triangles = _mesh_triangles(first)
    second_triangles = _mesh_triangles(second)
    if not first_triangles or not second_triangles:
        return None
    return _bvh_intersects(_build_bvh(first_triangles), _build_bvh(second_triangles))


def _observations(telemetry: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = telemetry.get("observations")
    if not isinstance(values, list):
        return []
    return sorted(
        [value for value in values if isinstance(value, Mapping) and isinstance(value.get("frame"), int)],
        key=lambda value: int(value["frame"]),
    )


def _entity_at(observation: Mapping[str, Any], entity_id: str) -> Mapping[str, Any] | None:
    entities = observation.get("entities")
    value = entities.get(entity_id) if isinstance(entities, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _kind_map(record: Mapping[str, Any], telemetry: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for entity in record.get("entities", []) or []:
        if isinstance(entity, Mapping) and entity.get("id"):
            result[str(entity["id"])] = str(entity.get("kind") or "").lower()
    for entity in (record.get("proxy_scene") or {}).get("entities", []) or []:
        if isinstance(entity, Mapping) and entity.get("id"):
            result.setdefault(str(entity["id"]), str(entity.get("kind") or "").lower())
    for entity_id, value in (telemetry.get("objects") or {}).items():
        if isinstance(value, Mapping):
            result.setdefault(str(entity_id), str(value.get("kind") or "").lower())
    return {key: ("actor" if value in {"character", "human"} else value) for key, value in result.items()}


def _event_map(record: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    values = record.get("event_graph") or record.get("events") or []
    return {
        str(value.get("id")): value
        for value in values
        if isinstance(value, Mapping) and value.get("id")
    }


def _event_frames(event: Mapping[str, Any] | None, fps: float) -> range:
    if not event:
        return range(0, 0)
    start = round(float(event.get("start", 0.0)) * fps) + 1
    end = round(float(event.get("end", 0.0)) * fps) + 1
    return range(min(start, end), max(start, end) + 1)


def _hand_points(entity: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    bones = entity.get("pose_bones") or entity.get("bones") or {}
    if not isinstance(bones, Mapping):
        return []
    points: list[tuple[float, float, float]] = []
    for name, bone in bones.items():
        if "hand" not in str(name).lower():
            continue
        if isinstance(bone, Mapping):
            for field in ("head", "tail"):
                point = _vec(bone.get(field))
                if point:
                    points.append(point)
    return points


def _center(entity: Mapping[str, Any]) -> tuple[float, float, float] | None:
    box = _obb(entity)
    return tuple(box["center"]) if box else None


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _contact_distance(
    observations_by_frame: Mapping[int, Mapping[str, Any]],
    actor_id: str,
    prop_id: str,
    frames: range,
) -> tuple[float | None, list[int]]:
    distances: list[tuple[int, float]] = []
    for frame in frames:
        observation = observations_by_frame.get(frame)
        if not observation:
            continue
        actor = _entity_at(observation, actor_id)
        prop = _entity_at(observation, prop_id)
        prop_center = _center(prop) if prop else None
        if not actor or not prop_center:
            continue
        hands = _hand_points(actor)
        if not hands:
            continue
        distances.append((frame, min(_distance(prop_center, hand) for hand in hands)))
    if not distances:
        return None, []
    return min(value for _frame, value in distances), [frame for frame, value in distances if value <= _CONTACT_DISTANCE]


def _support_contact(prop: Mapping[str, Any], support: Mapping[str, Any]) -> bool:
    prop_box = _obb(prop)
    support_box = _obb(support)
    if not prop_box or not support_box:
        return False
    prop_center, support_center = prop_box["center"], support_box["center"]
    prop_half, support_half = prop_box["half"], support_box["half"]
    horizontal = (
        abs(prop_center[0] - support_center[0]) <= prop_half[0] + support_half[0]
        and abs(prop_center[1] - support_center[1]) <= prop_half[1] + support_half[1]
    )
    bottom = prop_center[2] - prop_half[2]
    top = support_center[2] + support_half[2]
    return horizontal and abs(bottom - top) <= 0.12


def _longest_consecutive_run(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(set(values))
    longest = current = 1
    for left, right in zip(ordered, ordered[1:]):
        if right == left + 1:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def evaluate_physics_oracle(
    record: Mapping[str, Any],
    telemetry: Mapping[str, Any] | None,
    *,
    contract: Any | None = None,
    trajectory_plan: Any | None = None,
) -> dict[str, Any]:
    """Derive physical findings from raw observed geometry and pose state."""

    del trajectory_plan  # Explicitly prevent plan-only acceptance.
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    observations = _observations(telemetry)
    negative = set(record.get("negative_constraints") or [])
    oracle = record.get("oracle_expectations") or {}
    negative.update(oracle.get("required_negative_constraints") or [])
    interactions = [item for item in record.get("interactions", []) or [] if isinstance(item, Mapping)]
    contract_constraints = set(getattr(contract, "physics_constraints", []) or []) if contract is not None else set()
    if isinstance(contract, Mapping):
        contract_constraints.update(contract.get("physics_constraints") or [])
    negative.update(str(item) for item in contract_constraints)
    needs_physics = bool(negative & {"no_prop_penetration", "support_before_grasp", "no_identity_swap"}) or bool(interactions)
    if not observations:
        findings = [_finding("physics_evidence_missing", "trusted raw observations are missing; physical claims are unavailable", ["telemetry.observations"])] if needs_physics else []
        return {
            "version": PHYSICS_ORACLE_VERSION,
            "status": "unavailable" if needs_physics else "pass",
            "observations": 0,
            "metrics": {},
            "findings": findings,
        }

    kinds = _kind_map(record, telemetry)
    by_frame = {int(item["frame"]): item for item in observations}
    fps = float(telemetry.get("fps") or record.get("fps") or 24.0)
    event_map = _event_map(record)
    expected_contact_windows: dict[tuple[str, str], set[int]] = {}
    for interaction in interactions:
        prop_id = str(interaction.get("prop_id") or "")
        for actor_field in ("giver_id", "receiver_id"):
            actor_id = str(interaction.get(actor_field) or "")
            if not actor_id or not prop_id:
                continue
            frames: set[int] = set()
            for event_field in ("attach_event_id", "transfer_event_id", "detach_event_id"):
                frames.update(_event_frames(event_map.get(str(interaction.get(event_field) or "")), fps))
            expected_contact_windows.setdefault((actor_id, prop_id), set()).update(frames)

    findings: list[dict[str, Any]] = []
    penetration_depths: list[float] = []
    mesh_bvh_queries = 0
    mesh_bvh_intersections = 0
    if "no_identity_swap" in negative:
        expected_ids = set(kinds)
        for entity_id in sorted(expected_ids):
            names = {
                str(entity.get("object_name"))
                for observation in observations
                if (entity := _entity_at(observation, entity_id)) is not None and entity.get("object_name")
            }
            if not names:
                findings.append(
                    _finding(
                        "physics_identity_evidence_missing",
                        f"raw observations do not expose a stable object name for {entity_id}",
                        [entity_id, "object_name"],
                    )
                )
            elif len(names) > 1:
                findings.append(
                    _finding(
                        "physics_identity_swap",
                        f"entity {entity_id} maps to multiple observed Blender objects",
                        [entity_id, *sorted(names)],
                    )
                )
    if "no_prop_penetration" in negative:
        entity_ids = sorted(kinds)
        for frame, observation in by_frame.items():
            for prop_id in entity_ids:
                if kinds.get(prop_id) != "prop":
                    continue
                prop = _entity_at(observation, prop_id)
                prop_box = _obb(prop) if prop else None
                if not prop_box:
                    continue
                for other_id in entity_ids:
                    if other_id == prop_id or kinds.get(other_id) not in {"actor", "support"}:
                        continue
                    other = _entity_at(observation, other_id)
                    other_box = _obb(other) if other else None
                    if not other_box:
                        continue
                    depth = _obb_overlap(prop_box, other_box)
                    if depth <= _PENETRATION_EPSILON:
                        continue
                    mesh_overlap = _mesh_bvh_overlap(prop, other)
                    if mesh_overlap is not None:
                        mesh_bvh_queries += 1
                        if not mesh_overlap:
                            # OBBs are deliberately conservative.  When the
                            # observer supplied mesh triangles, use the BVH
                            # narrow phase to avoid calling separated meshes a
                            # penetration merely because their broad bounds
                            # overlap.
                            continue
                        mesh_bvh_intersections += 1
                    allowed_window = frame in expected_contact_windows.get((other_id, prop_id), set())
                    if kinds.get(other_id) == "support" and _support_contact(prop, other):
                        continue
                    if allowed_window and depth <= _CONTACT_PENETRATION_LIMIT:
                        continue
                    penetration_depths.append(depth)
                    findings.append(
                        _finding(
                            "physics_penetration",
                            f"{prop_id} penetrates {other_id} at frame {frame}; overlap depth={depth:.4f}",
                            [prop_id, other_id, str(frame), f"depth={depth:.4f}"],
                        )
                    )
        if findings:
            # One finding per offending frame is useful for diagnosis, but the
            # aggregate is kept bounded for report size and patch attribution.
            findings = findings[:50]

    contact_checks = 0
    contact_successes = 0
    ownership_checks = 0
    for interaction in interactions:
        prop_id = str(interaction.get("prop_id") or "")
        giver_id = str(interaction.get("giver_id") or "")
        receiver_id = str(interaction.get("receiver_id") or interaction.get("final_owner_id") or "")
        attach_frames = _event_frames(event_map.get(str(interaction.get("attach_event_id") or "")), fps)
        transfer_frames = _event_frames(event_map.get(str(interaction.get("transfer_event_id") or "")), fps)
        giver_min, giver_contact = _contact_distance(by_frame, giver_id, prop_id, attach_frames)
        receiver_min, receiver_contact = _contact_distance(by_frame, receiver_id, prop_id, transfer_frames)
        attach_event = event_map.get(str(interaction.get("attach_event_id") or ""))
        detach_event = event_map.get(str(interaction.get("detach_event_id") or ""))
        transfer_event = event_map.get(str(interaction.get("transfer_event_id") or ""))
        if "handoff_requires_same_window_detach_attach" in negative and transfer_event:
            attach_end = float(attach_event.get("end", 0.0)) if attach_event else float("inf")
            transfer_start = float(transfer_event.get("start", 0.0))
            transfer_end = float(transfer_event.get("end", 0.0))
            detach_start = float(detach_event.get("start", 0.0)) if detach_event else float("inf")
            if attach_event is None or detach_event is None or attach_end > transfer_end or transfer_start > detach_start:
                findings.append(
                    _finding(
                        "physics_transfer_window_invalid",
                        f"interaction {interaction.get('id', 'unknown')} has no ordered attach/transfer/detach windows",
                        [str(interaction.get("id") or "interaction")],
                    )
                )
        contact_checks += 1
        if giver_min is not None and giver_contact:
            contact_successes += 1
        else:
            findings.append(
                _finding(
                    "interaction_giver_contact_missing",
                    f"{giver_id} has no observed hand contact with {prop_id} in the attach window",
                    [str(interaction.get("id") or "interaction"), giver_id, prop_id],
                )
            )
        ownership_checks += 1
        if receiver_min is None or not receiver_contact:
            findings.append(
                _finding(
                    "interaction_receiver_contact_missing",
                    f"{receiver_id} has no observed hand contact with {prop_id} in the transfer window",
                    [str(interaction.get("id") or "interaction"), receiver_id, prop_id],
                )
            )
            findings.append(
                _finding(
                    "interaction_final_owner_unobserved",
                    f"final owner {receiver_id} cannot be established from raw hand/object proximity",
                    [str(interaction.get("id") or "interaction"), receiver_id, prop_id],
                )
            )
        elif _longest_consecutive_run(receiver_contact) < min(2, len(list(transfer_frames))):
            findings.append(
                _finding(
                    "interaction_contact_discontinuous",
                    f"receiver contact for {prop_id} is not continuous across the transfer window",
                    [str(interaction.get("id") or "interaction"), receiver_id, prop_id],
                    severity="error",
                )
            )

        final_support_id = str(interaction.get("final_support_id") or "")
        if final_support_id and final_support_id in kinds:
            support_samples = [
                (_entity_at(by_frame[frame], prop_id), _entity_at(by_frame[frame], final_support_id))
                for frame in _event_frames(detach_event, fps)
                if frame in by_frame
            ]
            observed_support_samples = [
                (prop, support) for prop, support in support_samples if support is not None
            ]
            if observed_support_samples and not any(
                prop is not None and support is not None and _support_contact(prop, support)
                for prop, support in observed_support_samples
            ):
                findings.append(
                    _finding(
                        "interaction_final_support_missing",
                        f"{prop_id} does not contact declared final support {final_support_id} after detach",
                        [prop_id, final_support_id],
                    )
                )

    if "support_before_grasp" in negative:
        support_ids = [entity_id for entity_id, kind in kinds.items() if kind in {"support", "environment"}]
        prop_ids = [entity_id for entity_id, kind in kinds.items() if kind == "prop"]
        grasp_events = [
            event
            for event in event_map.values()
            if str(event.get("action") or "").lower() in {"grasp", "pick", "attach"}
        ]
        for prop_id in prop_ids:
            event = next(
                (
                    candidate
                    for candidate in grasp_events
                    if prop_id in [str(value) for value in candidate.get("target_ids", []) or []]
                ),
                None,
            )
            if event is None:
                continue
            start_frame = min(_event_frames(event, fps), default=None)
            pre_frames = [frame for frame in by_frame if start_frame is not None and frame < start_frame]
            if not support_ids or not pre_frames:
                findings.append(
                    _finding(
                        "physics_support_evidence_missing",
                        f"support-before-grasp cannot be verified for {prop_id}",
                        [prop_id, "support", "pre_grasp_frames"],
                    )
                )
                continue
            for frame in pre_frames:
                prop = _entity_at(by_frame[frame], prop_id)
                supports = [_entity_at(by_frame[frame], support_id) for support_id in support_ids]
                if prop is None or not any(support is not None and _support_contact(prop, support) for support in supports):
                    findings.append(
                        _finding(
                            "physics_support_before_grasp_violated",
                            f"{prop_id} is not supported before grasp at frame {frame}",
                            [prop_id, str(frame), *support_ids],
                        )
                    )
                    break

    teleport_count = 0
    max_velocity = 0.0
    max_acceleration = 0.0
    for entity_id in sorted(kinds):
        points = [(frame, _center(_entity_at(observation, entity_id) or {})) for frame, observation in by_frame.items()]
        points = [(frame, point) for frame, point in points if point is not None]
        velocities: list[tuple[int, float]] = []
        for (left_frame, left), (right_frame, right) in zip(points, points[1:]):
            dt = max(1.0 / max(fps, 1.0), (right_frame - left_frame) / max(fps, 1.0))
            distance = _distance(left, right)
            velocity = distance / dt
            max_velocity = max(max_velocity, velocity)
            velocities.append((right_frame, velocity))
            if distance > _MAX_STEP_DISTANCE:
                teleport_count += 1
                findings.append(_finding("physics_teleport", f"{entity_id} jumps {distance:.3f} units between frames {left_frame} and {right_frame}", [entity_id, str(left_frame), str(right_frame)]))
        for (left_frame, left_velocity), (right_frame, right_velocity) in zip(velocities, velocities[1:]):
            dt = max(1.0 / max(fps, 1.0), (right_frame - left_frame) / max(fps, 1.0))
            acceleration = abs(right_velocity - left_velocity) / dt
            max_acceleration = max(max_acceleration, acceleration)
            if acceleration > _MAX_ACCELERATION:
                findings.append(_finding("physics_acceleration_discontinuity", f"{entity_id} acceleration discontinuity={acceleration:.3f}", [entity_id, str(left_frame), str(right_frame)], severity="error"))

    hard = any(item["severity"] == "hard" for item in findings)
    return {
        "version": PHYSICS_ORACLE_VERSION,
        "status": "fail" if findings else "pass",
        "observations": len(observations),
        "metrics": {
            "penetration_count": len(penetration_depths),
            "max_penetration_depth": round(max(penetration_depths), 6) if penetration_depths else 0.0,
            "interaction_contact_rate": round(contact_successes / contact_checks, 6) if contact_checks else 1.0,
            "ownership_checks": ownership_checks,
            "teleport_count": teleport_count,
            "max_velocity": round(max_velocity, 6),
            "max_acceleration": round(max_acceleration, 6),
            "mesh_bvh_queries": mesh_bvh_queries,
            "mesh_bvh_intersections": mesh_bvh_intersections,
        },
        "hard_gate_failed": hard,
        "findings": findings,
    }


__all__ = ["PHYSICS_ORACLE_VERSION", "evaluate_physics_oracle"]
