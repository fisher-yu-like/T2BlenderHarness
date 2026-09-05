import json
import re
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


def test_exhausted_inner_loop_is_not_reported_as_local_visual_review(tmp_path: Path):
    from scripts.train_real_harness import (
        _final_batch_status,
        _inner_not_rendered_result,
        merge_real_scores,
    )

    inner_case = {
        "case_id": "case-exhausted",
        "status": "exhausted",
        "reason": "max_inner_attempts_exhausted",
        "attempts": [
            {
                "attempt": 1,
                "status": "evaluation_failed",
                "reason": "Blender telemetry unavailable",
            }
        ],
    }
    deterministic = _inner_not_rendered_result(
        case_id="case-exhausted",
        run_root=tmp_path,
        split="test",
        inner_case=inner_case,
    )
    merged = merge_real_scores(
        run_root=tmp_path,
        deterministic_results=[deterministic],
        vlm_results=[],
    )

    assert deterministic["vlm_status"] == "not_run"
    assert deterministic["vlm_reason"] == "inner_loop_exhausted"
    assert merged["cases"][0]["vlm_status"] == "not_run"
    assert merged["cases"][0]["vlm_reason"] == "inner_loop_exhausted"
    assert _final_batch_status(
        inner={"pending_case_ids": ["case-exhausted"]},
        vlm_scored_count=0,
        real_video_count=0,
    ) == "incomplete_inner_loop"


def test_batch_eval_uses_protocol_inner_attempt_budget():
    from scripts import run_batch_eval

    assert run_batch_eval.DEFAULT_INNER_ATTEMPTS == 3


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


def test_blender_version_extraction_ignores_leading_warnings():
    from scripts.render_proxy_jobs_parallel import _extract_blender_version

    stdout = (
        "00:00.953  reports | WARNING Path 'D:/blender/datafiles/missing.blend' cannot be found\n"
        "Blender 5.1.2 (hash ec6e62d40fa9 built 2026-05-19 01:37:34)\n"
    )

    assert _extract_blender_version(stdout, "") == "5.1.2"
    assert _extract_blender_version("warning only", "") is None


def test_geometry_audit_decodes_blender_output_as_utf8_with_replacement(tmp_path: Path, monkeypatch):
    import subprocess

    import scripts.evaluate_proxy_realism as evaluator

    run_dir = tmp_path / "case-geometry"
    run_dir.mkdir()
    (run_dir / "proxy.blend").write_bytes(b"blend")
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        (run_dir / "geometry_raw.json").write_text(
            json.dumps({"audit_available": True, "mesh_count": 0, "meshes": []}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    assert evaluator._inspect("blender", run_dir, timeout_s=10)["audit_available"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"


def test_cli_render_retries_failed_blender_execution_and_records_attempts(tmp_path: Path, monkeypatch):
    import subprocess
    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "hard-01-01"
    job_dir.mkdir()
    (job_dir / "blender_job.py").write_text("# fake job", encoding="utf-8")
    calls = {"count": 0, "kwargs": []}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        calls["kwargs"].append(kwargs)
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
    assert all(kwargs["encoding"] == "utf-8" for kwargs in calls["kwargs"])
    assert all(kwargs["errors"] == "replace" for kwargs in calls["kwargs"])


def test_cli_renderer_caps_parallel_workers_at_twelve(tmp_path: Path):
    from scripts.render_proxy_jobs_parallel import render_jobs

    report = render_jobs(tmp_path, blender_bin="missing-blender", workers=64)

    assert report["workers"] == 12


def test_cli_renderer_can_select_one_serial_group_of_case_ids(tmp_path: Path, monkeypatch):
    import scripts.render_proxy_jobs_parallel as renderer

    for case_id in ("case-01", "case-02", "case-03"):
        case_dir = tmp_path / case_id
        case_dir.mkdir()
        (case_dir / "blender_job.py").write_text("# frozen job", encoding="utf-8")

    monkeypatch.setattr(renderer, "_run_one", lambda job_dir, *args: {
        "case_id": Path(job_dir).name,
        "status": "success",
    })
    report = renderer.render_jobs(
        tmp_path,
        blender_bin="missing-blender",
        workers=2,
        case_ids=["case-01", "case-03"],
    )

    assert report["job_count"] == 2
    assert [item["case_id"] for item in report["results"]] == ["case-01", "case-03"]
    assert report["workers"] == 2


def test_cli_renderer_fails_closed_if_frozen_job_source_changes_during_retry(tmp_path: Path, monkeypatch):
    import subprocess
    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "case-source-change"
    job_dir.mkdir()
    job_path = job_dir / "blender_job.py"
    job_path.write_text("# immutable source", encoding="utf-8")
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            job_path.write_text("# changed source", encoding="utf-8")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    result = renderer._run_one(job_dir, "blender", timeout_s=10, max_retries=1)

    assert result["status"] == "failed"
    assert result["attempts"][0]["source_unchanged"] is False


def test_cli_renderer_fails_closed_if_rendered_manifest_loses_source_hash(tmp_path: Path, monkeypatch):
    import hashlib
    import subprocess
    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "manifest-source-hash"
    job_dir.mkdir()
    job_path = job_dir / "blender_job.py"
    job_path.write_text("# immutable job", encoding="utf-8")
    source_hash = hashlib.sha256(job_path.read_bytes()).hexdigest()
    (job_dir / "run_manifest.json").write_text(json.dumps({"code_hash": source_hash}), encoding="utf-8")

    def fake_run(*args, **kwargs):
        frames = job_dir / "frames" / "animation"
        frames.mkdir(parents=True)
        for number in (1, 2, 3):
            Image.new("RGB", (8, 8), (number, 0, 0)).save(frames / f"frame_{number:06d}.png")
        (job_dir / "frames" / "index.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
        (job_dir / "proxy.blend").write_bytes(b"blend")
        (job_dir / "telemetry.json").write_text("{}", encoding="utf-8")
        (job_dir / "run_manifest.json").write_text(json.dumps({"code_hash": None}), encoding="utf-8")
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    result = renderer._run_one(job_dir, "blender", timeout_s=10, max_retries=0)

    assert result["status"] == "failed"
    assert "manifest_code_hash_mismatch" in result["attempts"][0]["error"]


def test_cli_renderer_exposes_project_library_on_pythonpath(tmp_path: Path, monkeypatch):
    import subprocess
    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "agent-case"
    job_dir.mkdir()
    (job_dir / "blender_job.py").write_text("# generated agent job", encoding="utf-8")
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    renderer._run_one(job_dir, "blender", timeout_s=10, max_retries=0)

    assert captured["env"]["PYTHONPATH"].split(renderer.os.pathsep)[0] == str(renderer.ROOT)


def test_real_evaluator_cli_passes_blender_binary_to_geometry_audit(tmp_path: Path, monkeypatch):
    import sys
    import scripts.evaluate_real_runs as evaluator

    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_real_runs.py",
            "--run-root",
            str(tmp_path),
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--blender-bin",
            r"D:\blender\blender.exe",
        ],
    )
    monkeypatch.setattr(
        evaluator,
        "evaluate_real_split",
        lambda root, dataset_root, *, blender_bin: captured.update(
            root=str(root), dataset_root=str(dataset_root), blender_bin=blender_bin
        ) or [],
    )

    assert evaluator.main() == 0
    assert captured["blender_bin"] == r"D:\blender\blender.exe"


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


def test_preparation_failures_are_retained_as_not_rendered_cases(tmp_path: Path):
    import json

    from scripts.train_real_harness import merge_real_scores, preparation_failure_results

    run_root = tmp_path / "run"
    run_root.mkdir()
    failure_path = run_root / "case-02" / "codegen_failure.json"
    failure_path.parent.mkdir()
    failure_path.write_text(json.dumps({"status": "codegen_failed", "reason": "provider unavailable"}), encoding="utf-8")
    (run_root / "job_index.json").write_text(
        json.dumps(
            {
                "jobs": [
                    {"case_id": "case-01", "status": "prepared", "run_dir": str(run_root / "case-01")},
                    {
                        "case_id": "case-02",
                        "status": "codegen_failed",
                        "run_dir": str(run_root / "case-02"),
                        "failure_path": str(failure_path),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    failures = preparation_failure_results(run_root)
    report = merge_real_scores(
        run_root=run_root,
        deterministic_results=[
            {"case_id": "case-01", "status": "pass", "score": 90.0, "artifact_status": "complete"},
            *failures,
        ],
        vlm_results=[],
    )

    assert failures[0]["status"] == "not_rendered"
    assert failures[0]["preparation_status"] == "codegen_failed"
    assert report["case_count"] == 2
    assert report["cases"][1]["case_id"] == "case-02"
    assert report["cases"][1]["video_score"] is None
    assert report["preparation_failed_count"] == 1


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
    prompt = "A character carries a red cup to a marked destination."
    proxy_video = tmp_path / "single-01-01" / "proxy.mp4"
    proxy_video.parent.mkdir()
    proxy_video.write_bytes(b"real proxy video")
    handling = "Accepted because train improved while paired and overall dev did not regress."
    write_training_memory_markdown(
        destination,
        [
            {
                "round": 1,
                "attempt": 3,
                "split": "train",
                "case_id": "single-01-01",
                "prompt": prompt,
                "proxy_video": str(proxy_video),
                "director_plan_score": 88,
                "task_score": 82,
                "realism_score": 71,
                "review": "gpt-5.6-Luna confidence=0.83",
                "detected_problem": "camera intent missing",
                "owner": "director_camera",
                "fix_location": "src/videoact/director_camera.py",
                "fix_method": "add a handoff two-shot",
                "delta": "train +7.0; paired dev +1.0; overall dev +0.2",
                "handling": handling,
            }
        ],
    )
    content = destination.read_text(encoding="utf-8")
    assert (
        "| 轮数 | Attempt | Split | Case ID | Prompt | Proxy 视频地址 | Director plan 分 | "
        "Task score | Realism score | Review | 检测出的 Harness 问题 | Owner | 修复位置/方法 | "
        "提升或下降 | 自然语言处理 |"
    ) in content
    assert (
        f"| 1 | 3 | train | single-01-01 | {prompt} | {proxy_video} | 88 | 82 | 71 | "
        "gpt-5.6-Luna confidence=0.83 | camera intent missing | director_camera | "
        "src/videoact/director_camera.py: add a handoff two-shot | "
        f"train +7.0; paired dev +1.0; overall dev +0.2 | {handling} |"
    ) in content


def test_training_memory_markdown_escapes_pipes_and_newlines(tmp_path: Path):
    from scripts.train_real_harness import write_training_memory_markdown

    destination = tmp_path / "training.md"
    write_training_memory_markdown(
        destination,
        [
            {
                "prompt": "carry | cup\nto marker",
                "detected_problem": "camera | intent\nis missing",
                "fix_location": "src/videoact/director|camera.py\nmodule",
                "fix_method": "add | handoff\ntwo-shot",
                "handling": "accepted | retain\npatch",
            }
        ],
    )

    content = destination.read_text(encoding="utf-8")
    assert "carry \\| cup to marker" in content
    assert "camera \\| intent is missing" in content
    assert "src/videoact/director\\|camera.py module: add \\| handoff two-shot" in content
    assert "accepted \\| retain patch" in content
    data_row = content.splitlines()[-1]
    assert len(re.split(r"(?<!\\)\|", data_row)[1:-1]) == 15


def test_training_memory_markdown_renders_missing_and_none_scores_as_unavailable(tmp_path: Path):
    from scripts.train_real_harness import write_training_memory_markdown

    destination = tmp_path / "training.md"
    write_training_memory_markdown(
        destination,
        [
            {"case_id": "none", "director_plan_score": None, "task_score": None, "realism_score": None},
            {"case_id": "missing"},
            {"case_id": "blank", "director_plan_score": "", "task_score": "   ", "realism_score": "\t"},
        ],
    )

    data_rows = destination.read_text(encoding="utf-8").splitlines()[-3:]
    cells_by_case = {cells[3]: cells for cells in (row[2:-2].split(" | ") for row in data_rows)}
    for case_id in ("none", "missing", "blank"):
        assert cells_by_case[case_id][6:9] == ["unavailable", "unavailable", "unavailable"]
        assert "0" not in cells_by_case[case_id][6:9]


def test_training_memory_markdown_uses_deterministic_legacy_task_score_precedence(tmp_path: Path):
    from scripts.train_real_harness import write_training_memory_markdown

    destination = tmp_path / "training.md"
    write_training_memory_markdown(
        destination,
        [
            {"case_id": "explicit", "task_score": 91, "video_score": 82, "score": 73},
            {"case_id": "video", "task_final_score": 99, "video_score": 82, "score": 73},
            {"case_id": "score", "score": 73},
        ],
    )

    data_rows = destination.read_text(encoding="utf-8").splitlines()[-3:]
    cells_by_case = {cells[3]: cells for cells in (row[2:-2].split(" | ") for row in data_rows)}
    assert cells_by_case["explicit"][6:9] == ["unavailable", "91", "unavailable"]
    assert cells_by_case["video"][6:9] == ["unavailable", "82", "unavailable"]
    assert cells_by_case["score"][6:9] == ["unavailable", "73", "unavailable"]


def test_training_memory_markdown_composes_review_and_prefers_explicit_review(tmp_path: Path):
    from scripts.train_real_harness import write_training_memory_markdown

    destination = tmp_path / "training.md"
    write_training_memory_markdown(
        destination,
        [
            {"case_id": "composed", "review_source": "gpt-5.6-Luna", "review_confidence": 0.83},
            {
                "case_id": "explicit",
                "review": "manual review confidence=0.91",
                "review_source": "gpt-5.6-Luna",
                "review_confidence": 0.83,
            },
        ],
    )

    data_rows = destination.read_text(encoding="utf-8").splitlines()[-2:]
    cells_by_case = {cells[3]: cells for cells in (row[2:-2].split(" | ") for row in data_rows)}
    assert cells_by_case["composed"][9] == "gpt-5.6-Luna confidence=0.83"
    assert cells_by_case["explicit"][9] == "manual review confidence=0.91"


def test_update_training_memory_table_preserves_real_report_traceability_and_scores(tmp_path: Path):
    from scripts.train_real_harness import update_training_memory_table

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text(
        json.dumps({"case_id": "single-01-01", "prompt": "Carry the red cup to the marker."}) + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "runs"
    round_root = output_root / "round-01"
    report_root = round_root / "attempt-03" / "real" / "train"
    report_root.mkdir(parents=True)
    (round_root / "patch_manifest.json").write_text(
        json.dumps(
            {
                "owner": "director_camera",
                "detected_problem": "fallback problem",
                "fix_location": "src/videoact/director_camera.py",
                "fix_method": "add a handoff two-shot",
                "delta": "train +7.0; paired dev +1.0; overall dev +0.2",
                "handling": "Accepted after the paired holdout gate passed.",
            }
        ),
        encoding="utf-8",
    )
    proxy_video = report_root / "single-01-01" / "proxy.mp4"
    (report_root / "real_unified_score.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "single-01-01",
                        "proxy_video": str(proxy_video),
                        "director_plan_score": 88,
                        "task_final_score": 82,
                        "video_score": 80,
                        "realism_score": 71,
                        "review": "manual review confidence=0.91",
                        "review_source": "gpt-5.6-Luna",
                        "review_confidence": 0.83,
                        "deterministic_findings": ["camera intent missing"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    destination = tmp_path / "training.md"

    update_training_memory_table(output_root, dataset_root, destination)

    content = destination.read_text(encoding="utf-8")
    assert (
        f"| 1 | 3 | train | single-01-01 | Carry the red cup to the marker. | {proxy_video} | "
        "88 | 82 | 71 | manual review confidence=0.91 | camera intent missing | director_camera | "
        "src/videoact/director_camera.py: add a handoff two-shot | "
        "train +7.0; paired dev +1.0; overall dev +0.2 | Accepted after the paired holdout gate passed. |"
    ) in content


def test_update_training_memory_table_records_low_artifact_realism_as_diagnostic_issue(tmp_path: Path):
    from scripts.train_real_harness import update_training_memory_table

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text(
        json.dumps({"case_id": "case-low", "prompt": "A cup rotates slowly."}) + "\n",
        encoding="utf-8",
    )
    report_root = tmp_path / "runs" / "round-01" / "attempt-01" / "real" / "train"
    case_root = report_root / "case-low"
    case_root.mkdir(parents=True)
    (case_root / "run_manifest.json").write_text(json.dumps({"director_plan_hash": "a" * 64}), encoding="utf-8")
    (report_root / "real_unified_score.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-low",
                        "proxy_video": str(case_root / "proxy.mp4"),
                        "director_plan_score": 100,
                        "realism_score": 32.5,
                        "realism_score_kind": "artifact_only_proxy",
                        "realism_band": "artifact_only_weak",
                        "review_source": "assistant_local_review",
                        "review_confidence": None,
                        "deterministic_findings": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "memory.md"
    update_training_memory_table(tmp_path / "runs", dataset_root, destination)

    content = destination.read_text(encoding="utf-8")
    assert "artifact-only realism is weak" in content
    assert "independent visual review is pending" in content


def test_update_training_memory_table_keeps_sibling_diagnostic_roots(tmp_path: Path):
    from scripts.train_real_harness import update_training_memory_table

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text(
        "".join(
            json.dumps({"case_id": case_id, "prompt": prompt}) + "\n"
            for case_id, prompt in (("old-case", "A red cup rotates."), ("new-case", "A blue cup slides."))
        ),
        encoding="utf-8",
    )
    training_root = tmp_path / "out" / "training"
    old_root = training_root / "diagnostic-old"
    new_root = training_root / "diagnostic-new"
    for root, case_id, round_number in ((old_root, "old-case", 1), (new_root, "new-case", 2)):
        report_root = root / f"round-{round_number:02d}" / "attempt-01" / "real" / "train"
        case_root = report_root / case_id
        case_root.mkdir(parents=True)
        (report_root / "real_unified_score.json").write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "case_id": case_id,
                            "proxy_video": str(case_root / "proxy.mp4"),
                            "deterministic_score": 100,
                            "realism_score": 35,
                            "realism_score_kind": "artifact_only_proxy",
                            "realism_band": "artifact_only_weak",
                            "deterministic_findings": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    destination = tmp_path / "training.md"
    update_training_memory_table(new_root, dataset_root, destination)
    content = destination.read_text(encoding="utf-8")

    assert "old-case" in content
    assert "new-case" in content


def test_canonical_memory_path_is_not_overwritten_by_single_run_summary(tmp_path: Path):
    from scripts.train_real_harness import write_unified_outputs

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text("", encoding="utf-8")
    destination = tmp_path / "t2blendercodeharness-agent-training-memory-v1.md"
    write_unified_outputs(
        {
            "scoring_mode": "real_blender_video_vlm",
            "case_count": 0,
            "real_video_count": 0,
            "vlm_scored_count": 0,
            "aggregate": {
                "mean_final_score": None,
                "mean_task_final_score": None,
                "mean_artifact_only_realism_score": None,
            },
            "cases": [],
        },
        dataset_root=dataset_root,
        report_root=tmp_path / "run",
        markdown_path=destination,
    )

    content = destination.read_text(encoding="utf-8")
    assert "append-only" in content
    assert "Real Blender Video Evaluation" not in content


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
    assert all(
        len(round_info["overall_evaluation"]["train_cases"]) == 60
        and len(round_info["overall_evaluation"]["dev_cases"]) == 60
        for round_info in manifest["rounds"]
    )


def test_outer_attempt_has_bounded_inner_regeneration_and_caps_outer_attempts_at_five():
    from scripts.train_real_harness import build_attempt_policy

    policy = build_attempt_policy(5)

    assert policy["mode"] == "outer_loop_with_bounded_case_regeneration"
    assert policy["inner_case_attempts_max"] == 3
    assert policy["render_retries_per_case"] == 0
    assert policy["candidate_videos_per_case_max"] == 3
    # 1320 protocol case slots × at most 3 complete candidate generations.
    assert policy["candidate_videos_total_max"] == 3960
    assert policy["max_attempts"] == 5
    assert policy["videos_per_attempt"] == 20
    assert policy["overall_videos_per_round"] == 120
    assert policy["videos_total_max"] == 1320
