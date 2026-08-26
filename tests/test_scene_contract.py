import pytest


def test_builder_normalizes_walk_pickup_and_closeup_prompt():
    from videoact.scene_contract import SceneContractBuilder

    contract = SceneContractBuilder().build(
        "A character should walk to the table, pick up the red cup, and show the grasp closeup."
    )

    assert [event.id for event in contract.events] == ["walk", "reach", "grasp"]
    assert contract.entities[0].id == "character"
    assert {entity.id for entity in contract.entities} >= {"table", "red_cup"}
    assert contract.must_show == ["walk", "reach", "grasp"]
    assert "target_visible_before_grasp" in contract.camera_constraints
    assert "grasp_in_closeup" in contract.camera_constraints


def test_builder_emits_explicit_physics_and_relation_predicates():
    from videoact.scene_contract import SceneContractBuilder

    contract = SceneContractBuilder().build("Walk to a table and pick up a cup.")

    assert {relation.type for relation in contract.relations} == {"on"}
    assert "support_before_grasp" in contract.physics_constraints
    assert "contact_before_attachment" in contract.physics_constraints
    assert contract.events[-1].target_ids == ["cup"]


def test_builder_rejects_empty_prompt_before_creating_scene_code():
    from videoact.scene_contract import SceneContractBuilder

    with pytest.raises(ValueError, match="prompt"):
        SceneContractBuilder().build("   ")


def test_builder_accepts_explicit_duration_and_fps():
    from videoact.scene_contract import SceneContractBuilder

    contract = SceneContractBuilder().build(
        "The character walks to the table and picks up the cup.",
        duration_s=12.0,
        fps=30,
    )

    assert contract.duration_s == 12.0
    assert contract.fps == 30
    assert contract.events[-1].end <= 12.0


def test_builder_recognizes_third_person_walk_verb():
    from videoact.scene_contract import SceneContractBuilder

    contract = SceneContractBuilder().build("The character walks to the table and picks up the cup.")

    assert [event.id for event in contract.events] == ["walk", "reach", "grasp"]


def test_builder_recognizes_support_synonyms_and_common_proxy_props():
    from videoact.scene_contract import SceneContractBuilder

    contract = SceneContractBuilder().build("Observe a red cup on a plain support.")
    cube_contract = SceneContractBuilder().build("A character walks to the blue cube.")

    assert {entity.id for entity in contract.entities} >= {"red_cup", "table"}
    assert contract.relations[0].type == "on"
    assert {entity.id for entity in cube_contract.entities} >= {"blue_cube"}
