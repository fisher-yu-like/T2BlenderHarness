from __future__ import annotations

import pytest


def _result():
    from videoact.director import DirectorAgent

    return DirectorAgent().plan(
        "Alice carries the red cube while Bob carries the blue cup, then Alice hands the red cube to Bob.",
        scene_id="director-eval",
        duration_s=12.0,
        fps=24,
    )


def test_director_evaluator_returns_independent_perfect_score_for_valid_plan():
    from evaluator.director_metrics import evaluate_director_plan

    result = _result()
    report = evaluate_director_plan(
        result.director_plan,
        result.trajectory_plan,
        telemetry={"objects": {"actor_a": {}, "actor_b": {}, "red_cube": {}, "blue_cup": {}}},
    )

    assert report.director_plan_score == 100.0
    assert report.findings == []


@pytest.mark.parametrize(
    ("mutate", "failure_id"),
    [
        (
            lambda plan: plan.model_copy(
                update={
                    "assumptions": [
                        {
                            "id": "assumption-unsupported",
                            "statement": "the unseen room is a studio",
                            "supported_by_evidence_ids": ["missing-evidence"],
                        }
                    ]
                }
            ),
            "director_unsupported_assumption",
        ),
        (
            lambda plan: plan.model_copy(update={"evidence": []}),
            "director_evidence_missing",
        ),
        (
            lambda plan: plan.model_copy(
                update={
                    "events": [
                        plan.events[0].model_copy(update={"depends_on": ["missing-event"]}),
                        *plan.events[1:],
                    ]
                }
            ),
            "director_dependency_mismatch",
        ),
    ],
)
def test_director_evaluator_finds_plan_integrity_errors(mutate, failure_id):
    from evaluator.director_metrics import evaluate_director_plan

    result = _result()
    report = evaluate_director_plan(mutate(result.director_plan), result.trajectory_plan)

    assert failure_id in {finding.failure_id for finding in report.findings}
    assert report.director_plan_score < 100.0
    assert all(finding.owner in {
        "director_prompt_interpreter",
        "director_event_scheduler",
        "director_trajectory",
        "director_camera",
        "blender_code_agent",
        "blender_executor",
        "proxy_renderer",
        "evaluator",
    } for finding in report.findings)


def test_director_evaluator_detects_identity_swap_path_collision_and_invisible_target():
    from evaluator.director_metrics import evaluate_director_plan

    result = _result()
    actor_a = result.trajectory_plan.entities["actor_a"]
    actor_b = result.trajectory_plan.entities["actor_b"].model_copy(
        update={"states": actor_a.states}
    )
    swapped_telemetry = {
        "objects": {
            "actor_a": {"source_entity_id": "actor_b"},
            "actor_b": {"source_entity_id": "actor_a"},
        }
    }
    trajectory = result.trajectory_plan.model_copy(
        update={"entities": {**result.trajectory_plan.entities, "actor_b": actor_b}}
    )
    camera = result.director_plan
    bad_camera = result.trajectory_plan.camera.model_copy(
        update={
            "shots": [
                result.trajectory_plan.camera.shots[0].model_copy(
                    update={"visibility_predicates": {}}
                )
            ]
        }
    )
    trajectory = trajectory.model_copy(update={"camera": bad_camera})
    report = evaluate_director_plan(camera, trajectory, telemetry=swapped_telemetry)
    failure_ids = {finding.failure_id for finding in report.findings}

    assert "director_identity_swap" in failure_ids
    assert "director_path_collision" in failure_ids
    assert "director_target_invisible" in failure_ids


def test_director_plan_score_is_not_folded_into_deterministic_score():
    from evaluator.deterministic import DeterministicEvaluator

    result = _result()
    with_director = DeterministicEvaluator().evaluate(
        result.scene_contract,
        result.trajectory_plan,
        director_plan=result.director_plan,
    )
    without_director = DeterministicEvaluator().evaluate(
        result.scene_contract,
        result.trajectory_plan,
    )

    assert with_director.director_plan_score == 100.0
    assert with_director.score == without_director.score
