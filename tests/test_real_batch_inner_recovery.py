from __future__ import annotations

import json
import hashlib
from pathlib import Path


def test_real_batch_recovery_regenerates_plan_and_code_after_render_failure(tmp_path, monkeypatch):
    import scripts.train_real_harness as training

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    record = {
        "case_id": "case-a",
        "prompt": "A benchmark prompt.",
        "split": "train",
        "duration_s": 1.0,
        "fps": 3,
    }
    (dataset / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    root = tmp_path / "runs"
    calls: list[tuple[str, int]] = []

    def fake_prepare(split, out_dir, **kwargs):
        attempt = len([item for item in calls if item[0] == "prepare"]) + 1
        calls.append(("prepare", attempt))
        case_dir = root / "case-a"
        case_dir.mkdir(parents=True, exist_ok=True)
        if attempt == 1:
            failure_path = case_dir / "director_failure.json"
            failure_path.write_text(json.dumps({"status": "director_failed", "reason": "plan_gap"}), encoding="utf-8")
            index = {
                "generation_mode": "agent",
                "jobs": [{"case_id": "case-a", "run_dir": str(case_dir), "status": "director_failed", "failure_path": str(failure_path)}],
            }
        else:
            source = case_dir / "blender_job.py"
            source.write_text(
                'CASE_SCENE_PROFILE = {"profile_version": "codex-local-case-profile-v2", "case_signature": "aaaaaaaaaaaaaaaa"}\n',
                encoding="utf-8",
            )
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            plan_hash = "a" * 64
            (case_dir / "run_manifest.json").write_text(
                json.dumps({"case_id": "case-a", "director_plan_hash": plan_hash, "code_hash": source_hash}),
                encoding="utf-8",
            )
            index = {
                "generation_mode": "agent",
                "jobs": [
                    {
                        "case_id": "case-a",
                        "run_dir": str(case_dir),
                        "status": "prepared",
                        "codegen_call_id": "codex-local:case-a",
                        "director_plan_hash": plan_hash,
                        "job_path": str(source),
                        "code_hash": source_hash,
                    }
                ],
            }
        (root / "job_index.json").write_text(json.dumps(index), encoding="utf-8")
        return {"generation_mode": index["generation_mode"], "jobs": index["jobs"]}

    def fake_render(run_root, **kwargs):
        attempt = len([item for item in calls if item[0] == "render"]) + 1
        calls.append(("render", attempt))
        return {
            "status": "completed",
            "results": [{"case_id": "case-a", "status": "failed" if attempt == 1 else "success", "reason": "render_gap" if attempt == 1 else None}],
        }

    def fake_evaluate(run_dir, **kwargs):
        calls.append(("evaluate", len([item for item in calls if item[0] == "evaluate"]) + 1))
        return {
            "case_id": "case-a",
            "status": "pass",
            "score": 88.0,
            "director_plan_score": 90.0,
            "realism": {"score": 35.0, "score_kind": "artifact_only_proxy"},
            "proxy_video": str(Path(run_dir) / "proxy.mp4"),
        }

    monkeypatch.setattr(training, "prepare_jobs", fake_prepare)
    monkeypatch.setattr(training, "render_jobs", fake_render)
    monkeypatch.setattr(training, "evaluate_real_run", fake_evaluate)
    monkeypatch.setattr(training, "evaluate_split", lambda *args, **kwargs: [])

    result = training.run_real_batch_with_inner_loop(
        root,
        split="train",
        case_ids=["case-a"],
        dataset_root=dataset,
        harness_version="h1",
        evaluator_version="e1",
        blender_bin="blender",
        workers=4,
        timeout_s=30,
        vlm_model="gpt-5.6-luna",
        director_agent=object(),
        code_agent=object(),
        max_inner_attempts=3,
    )

    assert result["inner_loop"]["status"] == "completed"
    assert result["inner_loop"]["cases"]["case-a"]["selected_attempt"] == 3
    assert [kind for kind, _ in calls] == ["prepare", "prepare", "render", "prepare", "render", "evaluate"]
