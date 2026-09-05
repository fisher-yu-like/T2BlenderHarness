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


def test_real_evaluator_accepts_actor_alias_for_character_entity_kind():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, artifacts = real_inputs()
    telemetry["objects"]["character"]["kind"] = "actor"
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert not any(f.failure_id == "telemetry_entity_kind_mismatch" for f in report.findings)


def test_real_evaluator_surfaces_runtime_camera_findings_from_telemetry():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, artifacts = real_inputs()
    telemetry["camera_findings"] = [{
        "failure_id": "camera_occlusion_exceeded",
        "owner": "director_camera",
        "category": "camera_coverage",
        "severity": "error",
        "message": "handoff target is occluded",
        "evidence": ["handoff", "red_cup"],
    }]
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert any(f.failure_id == "camera_occlusion_exceeded" for f in report.findings)
    assert report.metrics["error_count"] >= 1


def test_real_evaluator_hard_fails_runtime_penetration_finding():
    from evaluator.deterministic import DeterministicEvaluator

    contract, plan, telemetry, artifacts = real_inputs()
    telemetry["attachment_penetration"] = [{
        "failure_id": "no_prop_penetration",
        "owner": "director_trajectory",
        "category": "interaction_geometry",
        "severity": "hard",
        "message": "prop intersects torso",
        "evidence": ["red_cup", "character", "42"],
    }]
    report = DeterministicEvaluator().evaluate_real(contract, plan, telemetry, artifacts)

    assert report.terminal_status == "fail"
    assert any(f.failure_id == "no_prop_penetration" for f in report.findings)


def test_evaluate_real_run_is_fail_closed_when_render_artifacts_are_missing(tmp_path):
    import json

    from scripts.evaluate_real_runs import evaluate_real_run
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run

    contract, plan, _telemetry, _artifacts = real_inputs()
    fingerprint = fingerprint_real_run(
        prompt_hash="p",
        plan_hash="t",
        harness_version="h",
        evaluator_version="e",
        blender_version="pending-mcp",
        render_settings={"resolution": [256, 256]},
    )
    manifest = RealRunManifest(
        run_id="failed-run",
        case_id="failed-run",
        split="train",
        prompt_hash="p",
        plan_hash="t",
        harness_version="h",
        evaluator_version="e",
        blender_version="pending-mcp",
        fps=plan.timebase.fps,
        frame_start=plan.timebase.frame_start,
        frame_end=plan.timebase.frame_end,
        render_settings={"resolution": [256, 256]},
        fingerprint=fingerprint,
        state="prepared",
    )
    for name, payload in (
        ("run_manifest.json", manifest.model_dump(mode="json")),
        ("scene_contract.json", contract.model_dump(mode="json")),
        ("trajectory.json", plan.model_dump(mode="json")),
        ("camera_plan.json", plan.camera.model_dump(mode="json")),
    ):
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "blender_job.py").write_text("# failed before execution", encoding="utf-8")

    result = evaluate_real_run(tmp_path, record={"case_id": "failed-run"})

    assert result["status"] == "fail"
    assert result["artifact_status"] == "incomplete"
    assert "missing_artifact:telemetry.json" in result["hard_failures"]
