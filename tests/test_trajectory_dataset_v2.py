import json


def complex_prompt():
    return (
        "Start with a wide establishing shot. The character walks to the table, reaches for the red cup, "
        "grasps it, lifts it, carries it to the drop zone, places it down, and releases it. "
        "The camera follows the approach, orbits during the carry, then dollies into a close-up of the release "
        "while keeping the cup visible before grasp."
    )


def test_complex_prompt_produces_ordered_events_and_camera_intents():
    from videoact.scene_contract import SceneContractBuilder

    contract = SceneContractBuilder().build(complex_prompt(), duration_s=16.0, fps=24)

    assert [event.id for event in contract.events] == [
        "walk",
        "reach",
        "grasp",
        "lift",
        "carry",
        "place",
        "release",
    ]
    assert {
        "camera_follow",
        "camera_orbit",
        "camera_dolly",
        "grasp_in_closeup",
    } <= set(contract.camera_constraints)


def test_complex_prompt_plan_contains_phase_states_motion_and_camera_observability():
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build(complex_prompt(), duration_s=16.0, fps=24)
    plan = TrajectoryPlanner().plan(contract)
    character = plan.entities["character"]

    assert len(character.states) >= 6
    assert {primitive.type for primitive in character.motion_primitives} >= {"ease_in_out", "linear"}
    assert [event.action for event in character.attachment_events] == ["attach", "detach"]
    assert {shot.trajectory_type for shot in plan.camera.shots} >= {"follow", "orbit", "dolly"}
    assert {item.event_id for item in plan.event_observability} == set(contract.must_show)
    assert all(item.covered_by_shots for item in plan.event_observability)


def test_trajectory_dataset_builder_emits_80_cases_and_split_metadata(tmp_path):
    from scripts.build_trajectory_dataset import build_dataset

    summary = build_dataset(tmp_path)
    records = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = json.loads((tmp_path / "splits.json").read_text(encoding="utf-8"))

    assert summary["cases"] == 80
    assert {name: len(values) for name, values in splits.items()} == {
        "train": 50,
        "dev": 20,
        "test": 10,
    }
    assert len({record["prompt"] for record in records}) == 80
    assert all(record["trajectory_expectations"]["event_order"] for record in records)
    assert all(record["trajectory_expectations"]["camera_types"] for record in records)
