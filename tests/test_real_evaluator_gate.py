from videoact.real_artifacts import RealArtifactReport
from videoact.scene_contract import SceneContractBuilder
from videoact.trajectory import TrajectoryPlanner


def real_inputs():
    contract = SceneContractBuilder().build(
        "A character walks to the table, picks up the red cup, and shows the grasp closeup."
    )
    plan = TrajectoryPlanner().plan(contract)
    telemetry = {
        "blender_version": "5.1.2",
        "frame_start": 1,
        "frame_end": 240,
        "fps": 24,
        "objects": {entity_id: {"keyframe_count": len(trajectory.states)} for entity_id, trajectory in plan.entities.items()},
        "camera": {"active": True},
        "event_observability": [item.model_dump(mode="json") for item in plan.event_observability],
    }
    artifacts = RealArtifactReport(artifact_status="complete", readable_frame_count=3)
    return contract, plan, telemetry, artifacts


def test_real_evaluator_accepts_complete_telemetry_and_artifacts():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, artifacts = real_inputs()
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert report.terminal_status == "pass"
    assert report.hard_gate_failed is False


def test_real_evaluator_flags_missing_entity_in_telemetry():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, artifacts = real_inputs()
    telemetry["objects"].pop("red_cup")
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "telemetry_missing_entity" for f in report.findings)


def test_real_evaluator_hard_fails_incomplete_artifacts():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, _ = real_inputs()
    artifacts = RealArtifactReport(
        artifact_status="incomplete",
        hard_failures=["missing_artifact:proxy.mp4"],
        readable_frame_count=3,
    )
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "incomplete_real_artifacts" for f in report.findings)


def test_real_evaluator_flags_semantic_entity_kind_mismatch():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, artifacts = real_inputs()
    telemetry["objects"]["table"]["kind"] = "prop"
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "telemetry_entity_kind_mismatch" for f in report.findings)
