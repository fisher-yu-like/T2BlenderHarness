from __future__ import annotations


def test_minimal_rig_has_independent_armature_bones_and_skinning_contract():
    from blender.character_rig import MINIMAL_BONES, rig_contract

    expected = {
        "root",
        "hips",
        "spine",
        "chest",
        "neck",
        "head",
        "shoulder.L",
        "upper_arm.L",
        "forearm.L",
        "hand.L",
        "shoulder.R",
        "upper_arm.R",
        "forearm.R",
        "hand.R",
        "thigh.L",
        "shin.L",
        "foot.L",
        "thigh.R",
        "shin.R",
        "foot.R",
    }
    assert expected.issubset(set(MINIMAL_BONES))
    contract = rig_contract("actor_custom")
    assert contract["armature_name"] == "actor_custom__armature"
    assert contract["independent_per_character"] is True
    assert contract["ik_targets"] == ["hand.L", "hand.R"]
    assert "ARMATURE" in contract["required_operations"]


def test_generated_job_exposes_ik_targets_and_anatomical_weight_assignment(tmp_path):
    from blender.real_proxy_job import compile_real_proxy_job
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build("A character observes a red cup on a table.", duration_s=2.0, fps=3)
    plan = TrajectoryPlanner().plan(contract)
    settings = {"engine": "BLENDER_EEVEE_NEXT", "resolution": [128, 128]}
    manifest = RealRunManifest(
        run_id="rig-job", case_id="rig-job", split="calibration", prompt_hash="p", plan_hash="t",
        harness_version="h", evaluator_version="e", blender_version="pending-mcp", fps=3,
        frame_start=1, frame_end=6, render_settings=settings,
        fingerprint=fingerprint_real_run(
            prompt_hash="p", plan_hash="t", harness_version="h", evaluator_version="e",
            blender_version="pending-mcp", render_settings=settings,
        ), state="prepared",
    )
    source = compile_real_proxy_job(plan, manifest, tmp_path)
    assert "add_hand_ik_constraint" in source
    assert "constraints.new(type=\"IK\")" in source
    assert "nearest_bone" in source
    assert "vertex_groups" in source
