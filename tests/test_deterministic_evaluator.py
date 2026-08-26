from videoact.contracts import CameraPlan, EntityState, EntityTrajectory
from videoact.scene_contract import SceneContractBuilder
from videoact.trajectory import TrajectoryPlanner


def planned_scene():
    contract = SceneContractBuilder().build(
        "A character walks to the table, picks up the red cup, and shows the grasp closeup."
    )
    return contract, TrajectoryPlanner().plan(contract)


def test_deterministic_evaluator_accepts_valid_proxy_plan():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan = planned_scene()
    report = DeterministicEvaluator().evaluate(contract, plan)

    assert report.terminal_status == "pass"
    assert report.hard_gate_failed is False
    assert report.score >= 90


def test_evaluator_flags_missing_required_event_as_hard_failure():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan = planned_scene()
    plan = plan.model_copy(
        update={
            "camera": CameraPlan(shots=[plan.camera.shots[0]]),
            "event_observability": [plan.event_observability[0]],
        }
    )

    report = DeterministicEvaluator().evaluate(contract, plan)

    assert report.terminal_status == "fail"
    assert report.hard_gate_failed is True
    assert any(f.failure_id == "missing_required_event" for f in report.findings)


def test_evaluator_flags_support_before_grasp_violation():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan = planned_scene()
    invalid_contract = contract.model_copy(update={"relations": []})

    report = DeterministicEvaluator().evaluate(invalid_contract, plan)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "support_before_grasp" for f in report.findings)


def test_evaluator_flags_attachment_when_contact_distance_is_too_large():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan = planned_scene()
    cup = plan.entities["red_cup"]
    broken_cup = cup.model_copy(
        update={
            "states": [
                state.model_copy(update={"position": (50.0, 0.0, 50.0)})
                if state.frame == 145
                else state
                for state in cup.states
            ]
        }
    )
    plan = plan.model_copy(update={"entities": {**plan.entities, "red_cup": broken_cup}})

    report = DeterministicEvaluator().evaluate(contract, plan)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "attachment_without_contact" for f in report.findings)


def test_evaluator_flags_velocity_spike():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan = planned_scene()
    character = plan.entities["character"]
    broken_character = character.model_copy(
        update={
            "states": [
                state.model_copy(update={"position": (100.0, 0.0, 0.0)})
                if state.frame == 97
                else state
                for state in character.states
            ]
        }
    )
    plan = plan.model_copy(update={"entities": {**plan.entities, "character": broken_character}})

    report = DeterministicEvaluator().evaluate(contract, plan)

    assert report.terminal_status == "pass"
    finding = next(f for f in report.findings if f.failure_id == "velocity_spike")
    assert finding.severity == "error"
    assert report.hard_gate_failed is False
    assert report.score == 82


def test_evaluator_hard_fails_incomplete_execution():
    from evaluator.deterministic import DeterministicEvaluator
    from videoact.contracts import ExecutionResult

    contract, plan = planned_scene()
    execution = ExecutionResult(status="failed", backend="fake", error="render failed")

    report = DeterministicEvaluator().evaluate(contract, plan, execution=execution)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "incomplete_proxy" for f in report.findings)


def test_support_gate_applies_only_when_contract_declares_support_constraint():
    from evaluator.deterministic import DeterministicEvaluator

    contract = SceneContractBuilder().build("A character picks up the red cup and places it down.")
    plan = TrajectoryPlanner().plan(contract)

    report = DeterministicEvaluator().evaluate(contract, plan)

    assert report.terminal_status == "pass"
    assert not any(f.failure_id == "support_before_grasp" for f in report.findings)
