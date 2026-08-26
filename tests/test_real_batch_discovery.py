import json
from pathlib import Path

from PIL import Image


def test_real_run_discovery_accepts_dataset_case_ids(tmp_path: Path):
    from scripts.evaluate_real_runs import discover_run_dirs

    for case_id in ("hard-01-01", "hard-02-10"):
        case_dir = tmp_path / case_id
        case_dir.mkdir()
        (case_dir / "run_manifest.json").write_text("{}", encoding="utf-8")

    assert [path.name for path in discover_run_dirs(tmp_path)] == ["hard-01-01", "hard-02-10"]


def test_discovery_ignores_non_run_directories(tmp_path: Path):
    from scripts.evaluate_real_runs import discover_run_dirs

    (tmp_path / "frames").mkdir()
    (tmp_path / "job_index.json").write_text("{}", encoding="utf-8")

    assert discover_run_dirs(tmp_path) == []


def test_successful_cli_job_records_rendered_state(tmp_path: Path):
    from scripts.render_proxy_jobs_parallel import mark_render_state

    job_dir = tmp_path / "hard-01-01"
    job_dir.mkdir()

    mark_render_state(job_dir, return_code=0, blender_version="Blender 5.1.2")

    state = (job_dir / "state.json").read_text(encoding="utf-8")
    response = (job_dir / "mcp_response.json").read_text(encoding="utf-8")
    assert '"state": "rendered"' in state
    assert '"status": "success"' in response


def test_cli_command_uses_absolute_job_path_and_requires_video(tmp_path: Path):
    from scripts.render_proxy_jobs_parallel import build_blender_command, classify_render_status

    job_dir = tmp_path / "hard-01-01"
    job_dir.mkdir()

    command = build_blender_command(job_dir, "D:/blender/blender.exe")

    assert Path(command[-1]).is_absolute()
    assert classify_render_status(0, False) == "failed"
    assert classify_render_status(0, True) == "success"


def test_cli_render_retries_failed_blender_execution_and_records_attempts(tmp_path: Path, monkeypatch):
    import subprocess
    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "hard-01-01"
    job_dir.mkdir()
    (job_dir / "blender_job.py").write_text("# fake job", encoding="utf-8")
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            frames = job_dir / "frames" / "animation"
            frames.mkdir(parents=True)
            for number in (1, 2, 3):
                Image.new("RGB", (8, 8), (number, 0, 0)).save(frames / f"frame_{number:06d}.png")
            (job_dir / "frames" / "index.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
            (job_dir / "proxy.blend").write_bytes(b"blend")
            (job_dir / "telemetry.json").write_text("{}", encoding="utf-8")
        return subprocess.CompletedProcess(args=args[0], returncode=0 if calls["count"] == 2 else 1, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    result = renderer._run_one(job_dir, "blender", timeout_s=10, max_retries=2)

    assert result["status"] == "success"
    assert result["retry_count"] == 1
    assert len(result["attempts"]) == 2
    assert result["video_probe"]["playable"] is True
    saved = json.loads((job_dir / "render_attempts.json").read_text(encoding="utf-8"))
    assert saved[0]["return_code"] == 1
    assert saved[1]["return_code"] == 0


def test_unified_report_uses_real_vlm_score_when_available(tmp_path: Path):
    from scripts.train_real_harness import merge_real_scores

    report = merge_real_scores(
        run_root=tmp_path,
        deterministic_results=[
            {"case_id": "hard-01-01", "status": "pass", "score": 100.0, "artifact_status": "complete"}
        ],
        vlm_results=[
            {"case_id": "hard-01-01", "status": "scored", "aggregate": {"final_score": 91.5}}
        ],
    )

    assert report["scoring_mode"] == "real_blender_video_vlm"
    assert report["aggregate"]["mean_final_score"] == 91.5
    assert report["cases"][0]["video_score"] == 91.5


def test_task_and_realism_scores_are_separate_channels(tmp_path: Path):
    from scripts.train_real_harness import merge_real_scores

    report = merge_real_scores(
        run_root=tmp_path,
        deterministic_results=[
            {
                "case_id": "hard-01-01",
                "status": "pass",
                "score": 100.0,
                "artifact_status": "complete",
                "realism": {
                    "score": 71.25,
                    "score_kind": "artifact_only_proxy",
                    "band": "artifact_only_strong",
                    "realism_claim": "not_established",
                    "requires_independent_review": True,
                    "evaluator_version": "realism-v3-independent-artifact-review",
                },
            }
        ],
        vlm_results=[
            {"case_id": "hard-01-01", "status": "scored", "aggregate": {"final_score": 91.5}}
        ],
    )

    assert report["cases"][0]["task_final_score"] == 91.5
    assert report["cases"][0]["realism_score"] == 71.25
    assert report["aggregate"]["mean_task_final_score"] == 91.5
    assert report["aggregate"]["mean_artifact_only_realism_score"] == 71.25
    assert report["score_channels"]["task_final_score"] != report["score_channels"]["artifact_only_realism_score"]


def test_six_round_batches_pair_disjoint_train_and_dev_cases():
    from scripts.train_real_harness import build_six_round_batches

    train_ids = [f"hard-{family:02d}-{variant:02d}" for family in range(1, 7) for variant in range(1, 11)]
    dev_ids = [f"hard-{family:02d}-{variant:02d}" for family in range(7, 13) for variant in range(1, 11)]

    batches = build_six_round_batches(train_ids, dev_ids)

    assert len(batches) == 6
    assert all(len(batch["train"]) == 10 and len(batch["dev"]) == 10 for batch in batches)
    assert [batch["train"][0] for batch in batches] == [f"hard-{family:02d}-01" for family in range(1, 7)]
    assert [batch["dev"][0] for batch in batches] == [f"hard-{family:02d}-01" for family in range(7, 13)]
    assert len({case_id for batch in batches for case_id in batch["train"]}) == 60
    assert len({case_id for batch in batches for case_id in batch["dev"]}) == 60


def test_harness_patch_scope_rejects_blender_dataset_and_evaluator_edits():
    import pytest
    from scripts.train_real_harness import validate_harness_patch_paths

    validate_harness_patch_paths(["src/videoact/scene_contract.py", "src/videoact/trajectory.py"])
    with pytest.raises(ValueError, match="Harness-only"):
        validate_harness_patch_paths(["blender/real_proxy_job.py"])
    with pytest.raises(ValueError, match="Harness-only"):
        validate_harness_patch_paths(["dataset/trajectory-v3-hard/manifest.jsonl"])
    with pytest.raises(ValueError, match="Harness-only"):
        validate_harness_patch_paths(["evaluator/deterministic.py"])


def test_training_memory_markdown_contains_required_natural_language_columns(tmp_path: Path):
    from scripts.train_real_harness import write_training_memory_markdown

    destination = tmp_path / "training.md"
    write_training_memory_markdown(
        destination,
        [
            {
                "round": 1,
                "prompt": "A character carries a red cup to a marked destination.",
                "proxy_video": "C:/runs/hard-01-01/proxy.mp4",
                "score": 82.0,
                "detected_problem": "camera intent missing",
                "fix_location": "src/videoact/scene_contract.py",
                "fix_method": "normalize rotate to orbit",
                "delta": 7.0,
                "handling": "The patch was retained because train improved and paired dev did not regress.",
            }
        ],
    )
    content = destination.read_text(encoding="utf-8")
    for column in ("轮数", "Prompt", "Proxy 视频地址", "打分", "检测出的 Harness 问题", "修复位置/方法", "提升或下降", "自然语言处理"):
        assert column in content
    assert "camera intent missing" in content


def test_anti_overfit_gate_requires_train_gain_and_paired_and_overall_dev_non_regression():
    from scripts.train_real_harness import anti_overfit_gate

    assert anti_overfit_gate(70, 72, 68, 68, 65, 65)["accepted"] is True
    assert anti_overfit_gate(70, 72, 68, 69, 65, 64)["accepted"] is False
    assert anti_overfit_gate(70, 70, 68, 69, 65, 65)["accepted"] is False


def test_protocol_manifest_declares_six_rounds_and_cumulative_overall_evaluation():
    from scripts.train_real_harness import build_protocol_manifest

    train_ids = [f"hard-{family:02d}-{variant:02d}" for family in range(1, 7) for variant in range(1, 11)]
    dev_ids = [f"hard-{family:02d}-{variant:02d}" for family in range(7, 13) for variant in range(1, 11)]
    test_ids = [f"hard-{family:02d}-{variant:02d}" for family in range(13, 15) for variant in range(1, 11)]

    manifest = build_protocol_manifest(train_ids, dev_ids, test_ids, dataset_fingerprint="fp")

    assert manifest["round_count"] == 6
    assert manifest["train_count"] == 60
    assert manifest["dev_count"] == 60
    assert manifest["test_count"] == 20
    assert manifest["attempts_per_round_max"] == 5
    assert manifest["videos_per_round_max"] == 220
    assert manifest["videos_total_max"] == 1320
    assert all(round_info["overall_evaluation"]["scope"] == "cumulative_train_and_dev" for round_info in manifest["rounds"])


def test_outer_attempt_has_no_inner_loop_retry_and_caps_at_five():
    from scripts.train_real_harness import build_attempt_policy

    policy = build_attempt_policy(5)

    assert policy["mode"] == "outer_loop_only"
    assert policy["inner_case_retries"] == 0
    assert policy["render_retries_per_case"] == 2
    assert policy["max_attempts"] == 5
    assert policy["videos_per_attempt"] == 20
    assert policy["overall_videos_per_round"] == 120
    assert policy["videos_total_max"] == 1320
