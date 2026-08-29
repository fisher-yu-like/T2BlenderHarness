from __future__ import annotations

from videoact.case_coverage import validate_case_coverage


def _record() -> dict:
    return {
        "case_id": "case-coverage-01",
        "prompt": "Alice carries the red cup and places it on the table.",
        "required_events": ["carry_01", "place_01"],
        "oracle_expectations": {
            "required_entity_ids": ["actor_a", "red_cup", "table"],
            "required_camera_events": ["carry_01", "place_01"],
        },
    }


def test_missing_required_event_fails_closed() -> None:
    report = validate_case_coverage(
        record=_record(),
        director_plan={
            "entities": [{"id": "actor_a"}, {"id": "red_cup"}, {"id": "table"}],
            "events": [{"id": "carry_01"}],
        },
        director_trajectories={"entities": {"actor_a": {}, "red_cup": {}, "table": {}}},
        director_camera={"shots": [{"shot_id": "shot_01", "required_event_ids": ["carry_01"]}]},
        generated_code="actor_a red_cup table carry_01",
    )

    assert report.status == "coverage_failed"
    assert "missing_plan_events:place_01" in report.hard_failures


def test_source_without_required_entity_or_event_is_rejected() -> None:
    report = validate_case_coverage(
        record=_record(),
        director_plan={
            "entities": [{"id": "actor_a"}, {"id": "red_cup"}, {"id": "table"}],
            "events": [{"id": "carry_01"}, {"id": "place_01"}],
        },
        director_trajectories={"entities": {"actor_a": {}, "red_cup": {}, "table": {}}},
        director_camera={"shots": [{"shot_id": "shot_01", "required_event_ids": ["carry_01", "place_01"]}]},
        generated_code="actor_a red_cup carry_01 place_01",
    )

    assert report.status == "coverage_failed"
    assert "missing_source_tokens:table" in report.hard_failures


def test_complete_case_coverage_passes() -> None:
    report = validate_case_coverage(
        record=_record(),
        director_plan={
            "entities": [{"id": "actor_a"}, {"id": "red_cup"}, {"id": "table"}],
            "events": [{"id": "carry_01"}, {"id": "place_01"}],
        },
        director_trajectories={
            "entities": {
                "actor_a": {"motion_primitives": [{"parameters": {"event_id": "carry_01"}}]},
                "red_cup": {"motion_primitives": [{"parameters": {"event_id": "place_01"}}]},
                "table": {},
            }
        },
        director_camera={"shots": [{"shot_id": "shot_01", "required_event_ids": ["carry_01", "place_01"]}]},
        generated_code=(
            "DIRECTOR_PLAN = {'actor_a': 'actor_a', 'red_cup': 'red_cup', 'table': 'table', "
            "'carry_01': 'carry_01', 'place_01': 'place_01'}"
        ),
    )

    assert report.status == "pass"
    assert report.coverage_ratio == 1.0


def test_same_code_hash_cannot_be_reused_for_different_prompt_cases() -> None:
    report = validate_case_coverage(
        record=_record(),
        director_plan={
            "entities": [{"id": "actor_a"}, {"id": "red_cup"}, {"id": "table"}],
            "events": [{"id": "carry_01"}, {"id": "place_01"}],
        },
        director_trajectories={"entities": {"actor_a": {}, "red_cup": {}, "table": {}}},
        director_camera={"shots": [{"shot_id": "shot_01", "required_event_ids": ["carry_01", "place_01"]}]},
        generated_code="actor_a red_cup table carry_01 place_01",
        existing_code_hashes={"other-case": "actor_a red_cup table carry_01 place_01"},
    )

    assert report.status == "coverage_failed"
    assert any(item.startswith("duplicate_source_hash:") for item in report.hard_failures)


def test_coverage_gate_rejects_comment_only_tokens() -> None:
    record = _record()
    plan = {
        "entities": [{"id": "actor_a"}, {"id": "red_cup"}, {"id": "table"}],
        "events": [{"id": "carry_01"}, {"id": "place_01"}],
    }
    trajectories = {
        "entities": {
            "actor_a": {"motion_primitives": [{"parameters": {"event_id": "carry_01"}}]},
            "red_cup": {"motion_primitives": [{"parameters": {"event_id": "place_01"}}]},
            "table": {"motion_primitives": []},
        }
    }
    camera = {"shots": [{"shot_id": "shot_01", "required_event_ids": ["carry_01", "place_01"]}]}

    report = validate_case_coverage(
        record=record,
        director_plan=plan,
        director_trajectories=trajectories,
        director_camera=camera,
        generated_code="""from blender.lib.geometry import box
# actor_a red_cup table carry_01 place_01
mesh = box((0, 0, 0), (1, 1, 1))
""",
    )

    assert report.status == "coverage_failed"
    assert any(item.startswith("missing_source_tokens:") for item in report.hard_failures)


def test_coverage_gate_requires_event_to_reach_trajectory_and_camera() -> None:
    record = _record()
    plan = {
        "entities": [{"id": "actor_a"}, {"id": "red_cup"}, {"id": "table"}],
        "events": [{"id": "carry_01"}, {"id": "place_01"}],
    }
    trajectories = {
        "entities": {
            "actor_a": {"motion_primitives": []},
            "red_cup": {"motion_primitives": [{"parameters": {"event_id": "place_01"}}]},
            "table": {"motion_primitives": []},
        }
    }
    camera = {"shots": [{"shot_id": "shot_01", "required_event_ids": ["place_01"]}]}

    report = validate_case_coverage(
        record=record,
        director_plan=plan,
        director_trajectories=trajectories,
        director_camera=camera,
        generated_code=(
            "from blender.lib.geometry import box\n"
            "DIRECTOR_PLAN = {'actor_a': 'actor_a', 'red_cup': 'red_cup', 'table': 'table', "
            "'carry_01': 'carry_01', 'place_01': 'place_01'}\n"
            "mesh = box((0, 0, 0), (1, 1, 1))\n"
        ),
    )

    assert report.status == "coverage_failed"
    assert "missing_trajectory_events:carry_01" in report.hard_failures
    assert "missing_camera_events:carry_01" in report.hard_failures


def test_benchmark_prompt_only_uses_director_plan_as_internal_coverage_contract() -> None:
    report = validate_case_coverage(
        record={
            "case_id": "vbench2-train-01-01",
            "prompt": "A person walks through a room while the camera follows.",
            "benchmark_prompt_only": True,
        },
        director_plan={
            "entities": [{"id": "actor_a"}, {"id": "room"}],
            "events": [{"id": "walk_01"}],
        },
        director_trajectories={
            "entities": {
                "actor_a": {"motion_primitives": [{"parameters": {"event_id": "walk_01"}}]},
                "room": {},
            }
        },
        director_camera={"shots": [{"shot_id": "shot_01", "required_event_ids": ["walk_01"]}]},
        generated_code="DIRECTOR_PLAN = {'actor_a': 'actor_a', 'room': 'room', 'walk_01': 'walk_01'}",
    )

    assert report.status == "pass"
    assert report.required_entities == ["actor_a", "room"]
    assert report.required_events == ["walk_01"]
    assert report.required_camera_events == ["walk_01"]
