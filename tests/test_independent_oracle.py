from __future__ import annotations


def test_independent_oracle_checks_multi_entity_ids_and_camera_event_coverage():
    from evaluator.independent_oracle import evaluate_independent_oracle
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build("A character walks to the table and picks up the red cup.")
    plan = TrajectoryPlanner().plan(contract)
    event_ids = [event.id for event in contract.events]
    record = {
        "case_id": "oracle-multi",
        "oracle_expectations": {
            "required_entity_ids": ["character", "red_cup"],
            "event_order": event_ids,
            "required_camera_events": [event_ids[0]],
        },
    }

    broken_plan = plan.model_copy(
        update={
            "camera": plan.camera.model_copy(
                update={
                    "shots": [
                        shot.model_copy(update={"required_event_ids": []})
                        for shot in plan.camera.shots
                    ]
                }
            )
        }
    )

    findings = evaluate_independent_oracle(record, contract, broken_plan)

    assert any(item.failure_id == "oracle_camera_event_missing" for item in findings)


def test_real_evaluator_does_not_skip_oracle_when_director_plan_exists():
    from evaluator.independent_oracle import evaluate_independent_oracle
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner
    from videoact.director import DirectorAgent

    result = DirectorAgent().plan(
        "Alice carries the red cube, then Alice hands the red cube to Bob.",
        scene_id="oracle-director",
        duration_s=10.0,
        fps=24,
    )
    record = {
        "case_id": "oracle-director",
        "oracle_expectations": {"event_order": ["missing_event"]},
    }
    findings = evaluate_independent_oracle(record, result.scene_contract, result.trajectory_plan)

    assert any(item.failure_id == "oracle_event_order_mismatch" for item in findings)


def test_independent_oracle_checks_declared_negative_constraints_from_runtime_evidence():
    from evaluator.independent_oracle import evaluate_independent_oracle
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build("A character walks to the table and picks up the red cup.")
    plan = TrajectoryPlanner().plan(contract)
    record = {
        "case_id": "oracle-negative",
        "oracle_expectations": {
            "event_order": [event.id for event in contract.events],
            "required_negative_constraints": ["no_prop_penetration"],
        },
    }

    findings = evaluate_independent_oracle(
        record,
        contract,
        plan,
        telemetry={
            "attachment_penetration": [
                {"failure_id": "no_prop_penetration", "severity": "hard", "message": "intersects"}
            ]
        },
    )

    assert any(item.failure_id == "oracle_negative_constraint_violated" for item in findings)


def test_independent_oracle_detects_unplanned_actor_lane_crossing_from_trajectory():
    from evaluator.independent_oracle import evaluate_independent_oracle
    from videoact.contracts import EntityState, EntityTrajectory
    from videoact.director import DirectorAgent

    result = DirectorAgent().plan(
        "Alice carries the red cube, then Alice hands the red cube to Bob.",
        scene_id="oracle-crossing",
        duration_s=10.0,
        fps=24,
    )
    actor_a = result.trajectory_plan.entities["actor_a"]
    actor_b = result.trajectory_plan.entities["actor_b"]
    crossing_a = actor_a.model_copy(
        update={
            "states": [
                EntityState(frame=1, position=(-1.0, -1.0, 0.0)),
                EntityState(frame=120, position=(1.0, 1.0, 0.0)),
            ]
        }
    )
    crossing_b = actor_b.model_copy(
        update={
            "states": [
                EntityState(frame=1, position=(-1.0, 1.0, 0.0)),
                EntityState(frame=120, position=(1.0, -1.0, 0.0)),
            ]
        }
    )
    broken_plan = result.trajectory_plan.model_copy(
        update={"entities": {**result.trajectory_plan.entities, "actor_a": crossing_a, "actor_b": crossing_b}}
    )
    record = {
        "case_id": "oracle-crossing",
        "proxy_scene": {"layout": {"actor_start_positions": {"actor_a": [-1, -1, 0], "actor_b": [-1, 1, 0]}}},
        "oracle_expectations": {
            "event_order": [event.id for event in result.scene_contract.events],
            "required_negative_constraints": ["no_unplanned_actor_crossing"],
        },
    }

    findings = evaluate_independent_oracle(record, result.scene_contract, broken_plan)

    assert any(item.failure_id == "oracle_negative_constraint_violated" for item in findings)


def test_independent_oracle_does_not_assume_legacy_character_for_multi_entity_trajectories():
    from evaluator.independent_oracle import evaluate_independent_oracle
    from videoact.director import DirectorAgent

    result = DirectorAgent().plan(
        "Alice carries the red cube, then Alice hands the red cube to Bob.",
        scene_id="oracle-multi-trajectory",
        duration_s=10.0,
        fps=24,
    )
    record = {
        "case_id": "oracle-multi-trajectory",
        "oracle_expectations": {
            "event_order": [event.id for event in result.scene_contract.events],
            "required_motion_primitives": ["s_curve", "arc"],
            "required_attachment_actions": ["attach", "transfer"],
        },
    }

    findings = evaluate_independent_oracle(record, result.scene_contract, result.trajectory_plan)

    assert not any(item.failure_id in {"oracle_motion_primitive_missing", "oracle_attachment_lifecycle_mismatch"} for item in findings)
