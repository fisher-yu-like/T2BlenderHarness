"""Pure geometry primitives returning vertices and polygon faces."""

from __future__ import annotations

import math
from typing import TypeAlias

from . import register_primitive


Vec3: TypeAlias = tuple[float, float, float]
Face: TypeAlias = tuple[int, ...]
Mesh: TypeAlias = tuple[list[Vec3], list[Face]]


def _positive(values: tuple[float, ...], label: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{label} values must be positive")


def _segments(value: int, label: str, minimum: int = 3) -> int:
    if int(value) < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return int(value)


@register_primitive(category="geometry", tags=["primitive", "rigid"], cost_estimate="low", example_usage="box((0, 0, 0), (1, 1, 1))")
def box(center: Vec3, size: Vec3) -> Mesh:
    """Axis-aligned box centered at ``center``; every size component must be > 0."""

    _positive(tuple(float(item) for item in size), "size")
    cx, cy, cz = (float(item) for item in center)
    sx, sy, sz = (float(item) / 2.0 for item in size)
    vertices = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ]
    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (4, 0, 3, 7),
    ]
    return vertices, faces


@register_primitive(category="geometry", tags=["organic", "smooth"], cost_estimate="medium", example_usage="ellipsoid((0, 0, 1), (1, 0.5, 1.2))")
def ellipsoid(center: Vec3, radii: Vec3, segments: int = 16, rings: int = 8) -> Mesh:
    """UV ellipsoid with positive radii and at least three angular segments."""

    _positive(tuple(float(item) for item in radii), "radii")
    segments = _segments(segments, "segments")
    rings = _segments(rings, "rings", minimum=2)
    cx, cy, cz = (float(item) for item in center)
    rx, ry, rz = (float(item) for item in radii)
    vertices: list[Vec3] = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(
                (
                    cx + rx * math.sin(phi) * math.cos(theta),
                    cy + ry * math.sin(phi) * math.sin(theta),
                    cz + rz * math.cos(phi),
                )
            )
    faces: list[Face] = []
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + next_segment
            c = (ring + 1) * segments + next_segment
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    return vertices, faces


@register_primitive(category="geometry", tags=["organic", "limb"], cost_estimate="medium", example_usage="capsule((0, 0, 0), (0, 0, 2), 0.2)")
def capsule(start: Vec3, end: Vec3, radius: float, rings: int = 8, segments: int = 12) -> Mesh:
    """Capsule-like tube between two points with spherical end rings."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    segments = _segments(segments, "segments")
    rings = _segments(rings, "rings", minimum=2)
    sx, sy, sz = (float(item) for item in start)
    ex, ey, ez = (float(item) for item in end)
    dx, dy, dz = ex - sx, ey - sy, ez - sz
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length <= 1e-9:
        raise ValueError("capsule endpoints must be distinct")
    axis = (dx / length, dy / length, dz / length)
    reference = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (0.0, 1.0, 0.0)
    ux = axis[1] * reference[2] - axis[2] * reference[1]
    uy = axis[2] * reference[0] - axis[0] * reference[2]
    uz = axis[0] * reference[1] - axis[1] * reference[0]
    u_length = math.sqrt(ux * ux + uy * uy + uz * uz)
    ux, uy, uz = ux / u_length, uy / u_length, uz / u_length
    vx, vy, vz = (
        axis[1] * uz - axis[2] * uy,
        axis[2] * ux - axis[0] * uz,
        axis[0] * uy - axis[1] * ux,
    )
    vertices: list[Vec3] = []
    for ring in range(rings + 1):
        t = ring / rings
        center = (sx + dx * t, sy + dy * t, sz + dz * t)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(
                (
                    center[0] + radius * (ux * math.cos(theta) + vx * math.sin(theta)),
                    center[1] + radius * (uy * math.cos(theta) + vy * math.sin(theta)),
                    center[2] + radius * (uz * math.cos(theta) + vz * math.sin(theta)),
                )
            )
    faces = []
    for ring in range(rings):
        for segment in range(segments):
            a = ring * segments + segment
            b = ring * segments + (segment + 1) % segments
            c = (ring + 1) * segments + (segment + 1) % segments
            d = (ring + 1) * segments + segment
            faces.append((a, b, c, d))
    return vertices, faces


@register_primitive(category="geometry", tags=["rigid", "beveled"], cost_estimate="medium", example_usage="rounded_box((0, 0, 0), (1, 2, 1), 0.1)")
def rounded_box(center: Vec3, size: Vec3, radius: float = 0.08) -> Mesh:
    """Box proxy with a validated bevel radius; compiler may apply a bevel modifier."""

    _positive(tuple(float(item) for item in size), "size")
    if radius <= 0 or radius * 2 >= min(float(item) for item in size):
        raise ValueError("radius must be positive and smaller than half the smallest size")
    return box(center, size)


def _radial_mesh(center: Vec3, rings: list[tuple[float, float]], segments: int) -> Mesh:
    cx, cy, cz = (float(item) for item in center)
    vertices: list[Vec3] = []
    for z, radius in rings:
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta), cz + z))
    faces: list[Face] = []
    for ring in range(len(rings) - 1):
        for segment in range(segments):
            faces.append(
                (
                    ring * segments + segment,
                    ring * segments + (segment + 1) % segments,
                    (ring + 1) * segments + (segment + 1) % segments,
                    (ring + 1) * segments + segment,
                )
            )
    return vertices, faces


@register_primitive(category="geometry", tags=["rigid", "rotational"], cost_estimate="low", example_usage="cylinder((0, 0, 0), 0.5, 1.0)")
def cylinder(center: Vec3, radius: float, depth: float, segments: int = 16) -> Mesh:
    """Open-sided cylindrical surface around the Z axis."""

    _positive((float(radius), float(depth)), "radius/depth")
    segments = _segments(segments, "segments")
    return _radial_mesh(center, [(-depth / 2.0, radius), (depth / 2.0, radius)], segments)


@register_primitive(category="geometry", tags=["rigid", "tapered"], cost_estimate="low", example_usage="cone((0, 0, 0), 0.5, 0.1, 1.0)")
def cone(center: Vec3, radius1: float, radius2: float, depth: float, segments: int = 16) -> Mesh:
    """Open-sided conical frustum around the Z axis."""

    _positive((float(radius1), float(radius2), float(depth)), "radius/depth")
    segments = _segments(segments, "segments")
    return _radial_mesh(center, [(-depth / 2.0, radius1), (depth / 2.0, radius2)], segments)


@register_primitive(category="geometry", tags=["organic", "rotational"], cost_estimate="medium", example_usage="torus((0, 0, 0), 1.0, 0.2)")
def torus(center: Vec3, major_radius: float, minor_radius: float, segments: int = 24, rings: int = 12) -> Mesh:
    """Torus centered on the Z axis with two validated angular resolutions."""

    _positive((float(major_radius), float(minor_radius)), "radii")
    segments = _segments(segments, "segments")
    rings = _segments(rings, "rings")
    cx, cy, cz = (float(item) for item in center)
    vertices: list[Vec3] = []
    for ring in range(rings):
        phi = 2.0 * math.pi * ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            radial = major_radius + minor_radius * math.cos(phi)
            vertices.append((cx + radial * math.cos(theta), cy + radial * math.sin(theta), cz + minor_radius * math.sin(phi)))
    faces: list[Face] = []
    for ring in range(rings):
        for segment in range(segments):
            faces.append(
                (
                    ring * segments + segment,
                    ring * segments + (segment + 1) % segments,
                    ((ring + 1) % rings) * segments + (segment + 1) % segments,
                    ((ring + 1) % rings) * segments + segment,
                )
            )
    return vertices, faces


@register_primitive(category="geometry", tags=["architectural", "extruded"], cost_estimate="low", example_usage="extruded_polygon([(0, 0), (1, 0), (1, 1)], 0.2)")
def extruded_polygon(polygon: list[tuple[float, float]], depth: float) -> Mesh:
    """Extrude a simple XY polygon symmetrically along Z."""

    if len(polygon) < 3:
        raise ValueError("polygon must contain at least three points")
    if depth <= 0:
        raise ValueError("depth must be positive")
    half = depth / 2.0
    vertices = [(float(x), float(y), -half) for x, y in polygon]
    vertices.extend((float(x), float(y), half) for x, y in polygon)
    count = len(polygon)
    faces: list[Face] = [tuple(range(count - 1, -1, -1)), tuple(range(count, 2 * count))]
    for index in range(count):
        next_index = (index + 1) % count
        faces.append((index, next_index, count + next_index, count + index))
    return vertices, faces


__all__ = [
    "Face",
    "Mesh",
    "Vec3",
    "box",
    "capsule",
    "cone",
    "cylinder",
    "ellipsoid",
    "extruded_polygon",
    "rounded_box",
    "torus",
]

