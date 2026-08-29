from __future__ import annotations


def test_frame_statistics_review_cannot_enter_patch_acceptance_gate():
    from videoact.outer_loop import evaluate_candidate

    decision = evaluate_candidate(
        {"train_score": 60, "dev_score": 60},
        {"train_score": 70, "dev_score": 60},
        {
            "paired_train_improvement": True,
            "paired_dev_non_regression": True,
            "review_source": "frame_statistics",
        },
        {
            "paired_dev_non_regression": True,
            "review_source": "frame_statistics",
        },
    )

    assert decision.accepted is False
    assert "independent_visual_review" in decision.failed_checks


def test_frame_statistics_review_payload_is_not_accepted_as_human_review(tmp_path):
    import json

    from PIL import Image

    from videoact.real_video import assemble_mp4_from_pngs

    from scripts.evaluate_real_videos import evaluate_vlm_run

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frames = []
    for number in (1, 2, 3):
        path = frames_dir / f"frame_{number:06d}.png"
        Image.new("RGB", (8, 8), (number * 30, 20, 10)).save(path)
        frames.append(path)
    (frames_dir / "index.json").write_text(
        json.dumps({"frames": [{"frame": n, "path": f"frame_{n:06d}.png"} for n in (1, 2, 3)]}),
        encoding="utf-8",
    )
    assemble_mp4_from_pngs(frames, tmp_path / "proxy.mp4", fps=3)
    (tmp_path / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps({"case_id": "case-01"}), encoding="utf-8")

    result = evaluate_vlm_run(
        tmp_path,
        prompt="Observe a table.",
        scene_contract={"events": [{"id": "observe"}], "fps": 3},
        assistant_review={"review_source": "frame_statistics"},
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "frame_statistics_not_eligible"
