from __future__ import annotations

import pytest

from blender.lib.rigging import add_ik_constraint, bind_mesh_to_armature, minimal_humanoid_armature


def test_minimal_humanoid_armature_has_independent_named_bones() -> None:
    armature = minimal_humanoid_armature("Alice", (1.0, 2.0, 0.0))

    assert armature.name == "Alice"
    assert {"hips", "chest", "hand.L", "hand.R", "thigh.L", "thigh.R"} <= set(armature.bones)
    assert armature.parent_map["hips"] == "root"
    assert armature.bones["hand.R"].position[0] > armature.bones["shoulder.R"].position[0]


def test_bind_mesh_to_armature_assigns_each_vertex_to_a_bone() -> None:
    armature = minimal_humanoid_armature("Alice", (0.0, 0.0, 0.0))
    weights = bind_mesh_to_armature([(0.0, 0.0, 0.0), (1.0, 0.0, 2.0), (-1.0, 0.0, 0.5)], armature)

    assigned = {index for entries in weights.values() for index, weight in entries if weight > 0}
    assert assigned == {0, 1, 2}
    assert all(abs(sum(weight for index, weight in entries if index == vertex)) - 1.0 < 1e-6 for vertex in assigned for entries in weights.values())


def test_ik_constraint_requires_positive_chain_length() -> None:
    with pytest.raises(ValueError):
        add_ik_constraint("hand.R", "target", chain_length=0)

