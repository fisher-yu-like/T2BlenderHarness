from __future__ import annotations


def test_generated_job_contains_ground_lighting_and_per_kind_principled_materials(tmp_path):
    from blender.real_proxy_job import compile_real_proxy_job
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build("A character observes a red cup on a table.", duration_s=2.0, fps=3)
    plan = TrajectoryPlanner().plan(contract)
    settings = {"engine": "BLENDER_EEVEE_NEXT", "resolution": [512, 512], "samples": 16}
    manifest = RealRunManifest(
        run_id="quality-01",
        case_id="quality-01",
        split="calibration",
        prompt_hash="p",
        plan_hash="t",
        harness_version="h",
        evaluator_version="e",
        blender_version="pending-mcp",
        fps=3,
        frame_start=1,
        frame_end=6,
        render_settings=settings,
        fingerprint=fingerprint_real_run(
            prompt_hash="p", plan_hash="t", harness_version="h", evaluator_version="e",
            blender_version="pending-mcp", render_settings=settings,
        ),
        state="prepared",
    )

    source = compile_real_proxy_job(
        plan,
        manifest,
        tmp_path,
        proxy_spec={"entities": [{"id": "character", "kind": "character"}, {"id": "red_cup", "kind": "prop"}, {"id": "table", "kind": "support"}]},
    )

    assert "ground_plane" in source
    assert source.count("light_add") >= 3
    assert "Principled BSDF" in source
    assert "entity_kind" in source
    assert "roughness" in source

