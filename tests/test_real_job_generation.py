import json


def real_plan():
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build(
        "A character walks to the table, picks up the red cup, and shows the grasp closeup."
    )
    return contract, TrajectoryPlanner().plan(contract)


def test_real_job_contains_render_telemetry_and_proxy_artifact_steps(tmp_path):
    from blender.real_proxy_job import compile_real_proxy_job
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run

    contract, plan = real_plan()
    fingerprint = fingerprint_real_run(
        prompt_hash="p",
        plan_hash="t",
        harness_version="h1",
        evaluator_version="e1",
        blender_version="pending-mcp",
        render_settings={"resolution": [256, 256]},
    )
    manifest = RealRunManifest(
        run_id="run-001",
        case_id="case-01",
        split="calibration",
        prompt_hash="p",
        plan_hash="t",
        harness_version="h1",
        evaluator_version="e1",
        blender_version="pending-mcp",
        fps=24,
        frame_start=1,
        frame_end=240,
        render_settings={"resolution": [256, 256]},
        fingerprint=fingerprint,
        state="prepared",
    )

    script = compile_real_proxy_job(plan, manifest, tmp_path)

    compile(script, "real_proxy_job.py", "exec")
    assert "BLENDER_EEVEE_NEXT" in script
    assert "animation" in script
    assert "telemetry.json" in script
    assert "bpy.ops.wm.save_as_mainfile" in script
    assert "bpy.ops.render.render(animation=True)" in script
    assert "ProxyWhiteMaterial" in script
    assert 'camera.data.keyframe_insert(data_path="lens"' in script
    assert '"id": "table", "kind": "support"' in script
    assert '"kind": obj.get("entity_kind", "unknown")' in script


def test_render_engine_falls_back_to_engine_supported_by_connected_blender():
    from blender.real_proxy_job import choose_render_engine

    assert choose_render_engine("BLENDER_EEVEE_NEXT", ["BLENDER_EEVEE", "CYCLES"]) == "BLENDER_EEVEE"
    assert choose_render_engine("BLENDER_EEVEE_NEXT", ["CYCLES"]) == "CYCLES"


def test_real_job_preserves_complex_camera_trajectory_and_support_semantics(tmp_path):
    from blender.real_proxy_job import compile_real_proxy_job
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run

    contract = SceneContractBuilder().build(
        "The character walks to the table, reaches, grasps, lifts, carries the red cup to the drop zone, places, and releases it. "
        "The camera follows, orbits during carry, and dollies into a close-up.",
        duration_s=16.0,
        fps=24,
    )
    plan = TrajectoryPlanner().plan(contract)
    fingerprint = fingerprint_real_run(
        prompt_hash="p",
        plan_hash="t",
        harness_version="h1",
        evaluator_version="e1",
        blender_version="pending-mcp",
        render_settings={"resolution": [256, 256]},
    )
    manifest = RealRunManifest(
        run_id="run-complex",
        case_id="case-complex",
        split="dev",
        prompt_hash="p",
        plan_hash="t",
        harness_version="h1",
        evaluator_version="e1",
        blender_version="pending-mcp",
        fps=24,
        frame_start=1,
        frame_end=384,
        render_settings={"resolution": [256, 256]},
        fingerprint=fingerprint,
        state="prepared",
    )

    script = compile_real_proxy_job(plan, manifest, tmp_path)

    assert 'trajectory_type == "orbit"' in script
    assert 'trajectory_type == "dolly"' in script
    assert '"camera_shots"' in script
    assert '"drop_zone"' in script


def test_prepare_real_jobs_writes_immutable_job_index(tmp_path):
    from scripts.prepare_real_jobs import prepare_jobs

    index = prepare_jobs("calibration", tmp_path, dataset_root="dataset", harness_version="h1")

    assert index["split"] == "calibration"
    assert index["case_count"] == 10
    assert len(index["jobs"]) == 10
    for job in index["jobs"]:
        run_dir = tmp_path / job["case_id"]
        assert (run_dir / "blender_job.py").exists()
        assert (run_dir / "run_manifest.json").exists()
        assert json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["state"] == "prepared"
