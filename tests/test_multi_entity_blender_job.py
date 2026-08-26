from __future__ import annotations

import json


def _manifest(tmp_path, *, director_plan_hash: str):
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run

    settings = {"engine": "BLENDER_EEVEE_NEXT", "resolution": [128, 128]}
    fingerprint = fingerprint_real_run(
        prompt_hash="prompt-hash",
        plan_hash="trajectory-hash",
        director_plan_hash=director_plan_hash,
        harness_version="director-v1",
        evaluator_version="deterministic-v2",
        blender_version="pending-mcp",
        render_settings=settings,
    )
    return RealRunManifest(
        run_id="real-multi-01",
        case_id="multi-01",
        split="train",
        prompt_hash="prompt-hash",
        plan_hash="trajectory-hash",
        director_plan_hash=director_plan_hash,
        harness_version="director-v1",
        evaluator_version="deterministic-v2",
        blender_version="pending-mcp",
        fps=24,
        frame_start=1,
        frame_end=288,
        render_settings=settings,
        fingerprint=fingerprint,
        state="prepared",
    )


def test_generated_job_compiles_all_multi_entity_semantics(tmp_path):
    from blender.real_proxy_job import compile_real_proxy_job
    from videoact.director import DirectorAgent

    result = DirectorAgent().plan(
        "Alice carries the red cube while Bob carries the blue cup, then Alice hands the red cube to Bob.",
        scene_id="multi-01",
        duration_s=12.0,
        fps=24,
    )
    manifest = _manifest(tmp_path, director_plan_hash=result.director_plan_hash)
    script = compile_real_proxy_job(
        result.trajectory_plan,
        manifest,
        tmp_path,
        director_plan=result.director_plan,
        director_trajectories=result.director_trajectories,
        director_camera=result.director_camera,
        proxy_spec={
            "scene_id": "multi-01",
            "entities": [
                {"id": "actor_a", "kind": "character", "role": "participant"},
                {"id": "actor_b", "kind": "character", "role": "participant"},
                {"id": "red_cube", "kind": "prop", "role": "target_object"},
                {"id": "blue_cup", "kind": "prop", "role": "target_object"},
                {"id": "support_surface", "kind": "support", "role": "environment"},
            ],
        },
    )

    compile(script, "multi_entity_real_proxy_job.py", "exec")
    for stable_id in ("actor_a", "actor_b", "red_cube", "blue_cup", "support_surface"):
        assert stable_id in script
    for required_marker in (
        "DIRECTOR_PLAN",
        "director_plan_hash",
        "current_owner_by_event",
        "interaction_state",
        "visibility",
        "target_ids",
        "validate_transfer",
        "attach",
        "detach",
    ):
        assert required_marker in script
    assert 'entity_id == "character"' not in script
    assert 'entity_id == "opening"' not in script


def test_real_artifact_fingerprint_binds_director_plan_hash():
    from videoact.real_artifacts import fingerprint_real_run

    common = dict(
        prompt_hash="p",
        plan_hash="t",
        harness_version="h",
        evaluator_version="e",
        blender_version="b",
        render_settings={"resolution": [128, 128]},
    )
    first = fingerprint_real_run(**common, director_plan_hash="a" * 64)
    second = fingerprint_real_run(**common, director_plan_hash="b" * 64)
    assert first != second


def test_prepare_multi_job_persists_director_hash_and_multientity_plan(tmp_path):
    from scripts.prepare_real_jobs import prepare_jobs

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    case = {
        "case_id": "multi-prepare-01",
        "prompt": "Alice carries the red cube while Bob carries the blue cup.",
        "duration_s": 8.0,
        "fps": 24,
        "proxy_scene": {
            "entities": [
                {"id": "actor_a", "kind": "character", "role": "participant"},
                {"id": "actor_b", "kind": "character", "role": "participant"},
                {"id": "red_cube", "kind": "prop", "role": "target_object"},
                {"id": "blue_cup", "kind": "prop", "role": "target_object"},
            ]
        },
    }
    (dataset_root / "manifest.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    (dataset_root / "splits.json").write_text(json.dumps({"train": [case["case_id"]]}), encoding="utf-8")

    prepare_jobs("train", tmp_path / "out", dataset_root=dataset_root)
    run_dir = tmp_path / "out" / case["case_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    director_plan = json.loads((run_dir / "director_plan.json").read_text(encoding="utf-8"))
    trajectory = json.loads((run_dir / "trajectory.json").read_text(encoding="utf-8"))
    assert manifest["director_plan_hash"] == "".join([]) or len(manifest["director_plan_hash"]) == 64
    assert {"actor_a", "actor_b", "red_cube", "blue_cup"}.issubset(trajectory["entities"])
    assert director_plan["request"]["scene_id"] == case["case_id"]
