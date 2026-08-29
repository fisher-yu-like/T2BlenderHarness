from __future__ import annotations

import pytest


def test_rollouts_aggregate_mean_variance_and_pass_rate():
    from videoact.run_manifest import aggregate_rollouts

    result = aggregate_rollouts(
        "case-01",
        [
            {"seed": 11, "score": 80.0, "passed": True},
            {"seed": 12, "score": 60.0, "passed": False},
        ],
    )

    assert result["case_id"] == "case-01"
    assert result["rollout_count"] == 2
    assert result["seeds"] == [11, 12]
    assert result["mean_score"] == 70.0
    assert result["score_std"] == 10.0
    assert result["pass_rate"] == 0.5


def test_rollouts_reject_duplicate_seed_and_fingerprint_changes_with_seed():
    from videoact.run_manifest import aggregate_rollouts, rollout_fingerprint
    from videoact.real_artifacts import fingerprint_real_run

    with pytest.raises(ValueError, match="unique seeds"):
        aggregate_rollouts(
            "case-01",
            [{"seed": 7, "score": 80, "passed": True}, {"seed": 7, "score": 80, "passed": True}],
        )
    assert rollout_fingerprint("case-01", seed=7, harness_version="h1") != rollout_fingerprint(
        "case-01", seed=8, harness_version="h1"
    )
    common = {
        "prompt_hash": "p",
        "plan_hash": "t",
        "harness_version": "h1",
        "evaluator_version": "e1",
        "blender_version": "b",
        "render_settings": {},
    }
    assert fingerprint_real_run(**common, rollout_seed=7) != fingerprint_real_run(**common, rollout_seed=8)


def test_renderer_report_keeps_rollout_seed(monkeypatch, tmp_path):
    import sys

    import scripts.render_proxy_jobs_parallel as renderer

    case_dir = tmp_path / "case-01"
    case_dir.mkdir()
    (case_dir / "blender_job.py").write_text("# fake", encoding="utf-8")
    captured = {}

    def fake_run_one(job_dir, blender_bin, timeout_s, max_retries=2, rollout_seed=None):
        captured["seed"] = rollout_seed
        return {"case_id": job_dir.name, "status": "success"}

    monkeypatch.setattr(renderer, "_run_one", fake_run_one)
    report = renderer.render_jobs(tmp_path, blender_bin=sys.executable, workers=1, rollout_seed=23)

    assert captured["seed"] == 23
    assert report["rollout_seed"] == 23
