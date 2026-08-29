from __future__ import annotations

import pytest

from blender.lib.geometry import (
    box,
    capsule,
    cone,
    cylinder,
    ellipsoid,
    extruded_polygon,
    rounded_box,
    torus,
)


@pytest.mark.parametrize(
    "builder",
    [
        lambda: box((0, 0, 0), (1, 2, 3)),
        lambda: ellipsoid((0, 0, 0), (1, 2, 3), segments=12, rings=6),
        lambda: capsule((0, 0, 0), (0, 0, 2), 0.25, segments=12, rings=6),
        lambda: rounded_box((0, 0, 0), (1, 2, 3), radius=0.1),
        lambda: cylinder((0, 0, 0), radius=0.5, depth=1.0, segments=12),
        lambda: cone((0, 0, 0), radius1=0.5, radius2=0.1, depth=1.0, segments=12),
        lambda: torus((0, 0, 0), major_radius=1.0, minor_radius=0.2, segments=12, rings=6),
        lambda: extruded_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], depth=0.2),
    ],
)
def test_geometry_primitives_return_non_empty_meshes(builder) -> None:
    vertices, faces = builder()

    assert vertices
    assert faces
    assert all(len(vertex) == 3 for vertex in vertices)
    assert all(len(face) >= 3 for face in faces)


def test_geometry_primitives_reject_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        box((0, 0, 0), (1, 0, 1))
    with pytest.raises(ValueError):
        cylinder((0, 0, 0), radius=0.0, depth=1.0)

