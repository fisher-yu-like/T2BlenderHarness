from __future__ import annotations

import pytest

from blender.lib.constraints import child_of_constraint, track_to_constraint


def test_child_of_constraint_preserves_monotonic_influence_keyframes() -> None:
    constraint = child_of_constraint(
        "red_cup",
        "Alice_armature",
        "hand.R",
        [(10, 0.0), (20, 1.0), (30, 1.0)],
    )

    assert constraint.type == "CHILD_OF"
    assert constraint.subtarget == "hand.R"
    assert constraint.influence_keyframes == [(10, 0.0), (20, 1.0), (30, 1.0)]


def test_constraint_rejects_invalid_influence() -> None:
    with pytest.raises(ValueError):
        child_of_constraint("prop", "rig", "hand.R", [(1, 1.2)])
    with pytest.raises(ValueError):
        track_to_constraint("camera", "target", track_axis="bad")

