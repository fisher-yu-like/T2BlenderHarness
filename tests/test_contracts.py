import pytest
from pydantic import ValidationError


def valid_scene_payload():
    return {
        "scene_id": "walk_to_table_pickup_cup",
        "duration_s": 10.0,
        "fps": 24,
        "entities": [
            {"id": "character", "kind": "character", "role": "actor"},
            {"id": "table", "kind": "support", "role": "environment"},
            {"id": "red_cup", "kind": "prop", "role": "target_object"},
        ],
        "events": [
            {"id": "walk", "start": 0.0, "end": 4.0},
            {"id": "reach", "start": 4.0, "end": 6.0},
            {"id": "grasp", "start": 6.0, "end": 8.0},
        ],
        "relations": [
            {"type": "on", "subject": "red_cup", "object": "table"},
        ],
        "must_show": ["walk", "grasp"],
        "physics_constraints": ["no_penetration", "support_before_grasp"],
        "camera_constraints": ["target_visible_before_grasp", "grasp_in_closeup"],
    }


def test_scene_contract_requires_event_times_and_entities():
    from videoact.contracts import SceneContract

    contract = SceneContract.model_validate(valid_scene_payload())

    assert contract.events[0].end == 4.0
    assert contract.entities[2].id == "red_cup"


@pytest.mark.parametrize(
    "field,value",
    [("duration_s", 0), ("duration_s", -1), ("fps", 0), ("fps", -24)],
)
def test_scene_contract_rejects_non_positive_timebase(field, value):
    from videoact.contracts import SceneContract

    payload = valid_scene_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        SceneContract.model_validate(payload)


def test_scene_contract_rejects_duplicate_ids_and_unknown_relation_entities():
    from videoact.contracts import SceneContract

    duplicate = valid_scene_payload()
    duplicate["entities"].append({"id": "table", "kind": "prop", "role": "actor"})
    with pytest.raises(ValidationError, match="unique"):
        SceneContract.model_validate(duplicate)

    unknown_relation = valid_scene_payload()
    unknown_relation["relations"][0]["object"] = "missing"
    with pytest.raises(ValidationError, match="unknown entity"):
        SceneContract.model_validate(unknown_relation)


def test_scene_contract_rejects_out_of_bounds_or_reversed_events():
    from videoact.contracts import SceneContract

    out_of_bounds = valid_scene_payload()
    out_of_bounds["events"][0]["end"] = 10.1
    with pytest.raises(ValidationError, match="duration"):
        SceneContract.model_validate(out_of_bounds)

    reversed_event = valid_scene_payload()
    reversed_event["events"][0]["start"] = 4.0
    reversed_event["events"][0]["end"] = 3.0
    with pytest.raises(ValidationError, match="end"):
        SceneContract.model_validate(reversed_event)


def test_trajectory_plan_requires_monotonic_frame_ranges():
    from videoact.contracts import TrajectoryPlan

    payload = {
        "timebase": {"fps": 24, "frame_start": 1, "frame_end": 240},
        "entities": {
            "character": {
                "states": [
                    {"frame": 1, "position": [0, 0, 0]},
                    {"frame": 24, "position": [1, 0, 0]},
                ],
                "motion_primitives": [],
                "attachment_events": [],
            }
        },
        "camera": {"shots": []},
        "event_observability": [],
        "validation_intents": [],
    }
    plan = TrajectoryPlan.model_validate(payload)
    assert plan.timebase.frame_end == 240

    payload["timebase"]["frame_end"] = 0
    with pytest.raises(ValidationError, match="frame_end"):
        TrajectoryPlan.model_validate(payload)


def test_run_manifest_hash_is_stable_for_same_payload():
    from videoact.contracts import RunManifest

    payload = {
        "run_id": "run-001",
        "scene_id": "demo",
        "prompt_hash": "abc",
        "harness_version": "h1",
        "evaluator_version": "e1",
        "plan_hash": "p1",
        "backend": "fake",
        "frame_start": 1,
        "frame_end": 24,
        "artifacts": {"plan": "plan.json"},
    }
    first = RunManifest.model_validate(payload)
    second = RunManifest.model_validate(payload)

    assert first.content_hash() == second.content_hash()
    assert len(first.content_hash()) == 64
