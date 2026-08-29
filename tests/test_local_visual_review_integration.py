from __future__ import annotations

import json

from test_real_video_metrics import _write_video_run


def test_assistant_local_mode_uses_real_mp4_evidence_when_runtime_observations_exist(tmp_path):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    root, contract, trajectory = _write_video_run(tmp_path / "review")
    (root / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 100, "findings": []}),
        encoding="utf-8",
    )
    (root / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
    (root / "geometry_report.json").write_text(json.dumps({"hard_gate_failed": False, "score": 80}), encoding="utf-8")
    (root / "visual_evidence.json").write_text(json.dumps({"review_source": "frame_statistics"}), encoding="utf-8")

    result = evaluate_vlm_run(
        root,
        prompt="A person walks while carrying a cup.",
        scene_contract=contract,
        assistant_local=True,
    )

    assert result["status"] == "unavailable"
    assert result["review_source"] == "deterministic_video_proxy_metrics"
    assert result["reason"] == "deterministic_video_proxy_metrics_not_a_visual_judge"
    assert result["deterministic_video_proxy_metrics"]["source"] == "actual_proxy_mp4_and_runtime_observations"
    assert result["task_score"] is None
    assert result["realism_score"] is None
