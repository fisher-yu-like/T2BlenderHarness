import json

from PIL import Image

from videoact.real_video import assemble_mp4_from_pngs


def make_review_run(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_paths = []
    for number in (1, 2, 3):
        path = frames_dir / f"frame_{number:06d}.png"
        Image.new("RGB", (8, 8), (number, 0, 0)).save(path)
        frame_paths.append(path)
    (frames_dir / "index.json").write_text(
        json.dumps({"frames": [{"frame": n, "path": f"frame_{n:06d}.png"} for n in (1, 2, 3)]}),
        encoding="utf-8",
    )
    assemble_mp4_from_pngs(frame_paths, tmp_path / "proxy.mp4", fps=3)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"case_id": "case-local-01"}), encoding="utf-8")
    (tmp_path / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )
    return frame_paths


def review_scores():
    return {
        "prompt_compliance": 90,
        "physical_plausibility": 90,
        "camera_coverage": 80,
        "camera_innovation": 80,
        "character_trajectory": 90,
        "object_trajectory": 90,
        "event_timing": 90,
        "temporal_smoothness": 90,
        "visual_clarity": 90,
        "appearance_detail": 90,
        "physical_realism": 90,
        "spatial_consistency": 90,
        "motion_naturalness": 90,
        "visual_presentation": 90,
        "visible_evidence": ["chronological sampled frames show the reviewed scene"],
        "weaknesses": ["sample is intentionally minimal"],
        "confidence": 0.8,
    }


def test_assistant_local_mode_creates_review_request_without_score(tmp_path):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    frame_paths = make_review_run(tmp_path)
    result = evaluate_vlm_run(
        tmp_path,
        prompt="Observe a table.",
        scene_contract={"events": [{"id": "observe"}]},
        assistant_local=True,
    )

    assert result["status"] == "awaiting_assistant_review"
    assert result["review_source"] == "assistant_local_review"
    request = json.loads((tmp_path / "assistant_review_request.json").read_text(encoding="utf-8"))
    assert request["sampled_frames"] == [str(path.resolve()) for path in frame_paths]
    assert not result.get("aggregate")


def test_assistant_local_review_is_scored_by_the_same_aggregate(tmp_path):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    frame_paths = make_review_run(tmp_path)
    request = {
        "review_version": "assistant-local-v1",
        "reviewer": "codex-assistant",
        "sampled_frames": [str(path.resolve()) for path in frame_paths],
        "scores": review_scores(),
    }
    result = evaluate_vlm_run(
        tmp_path,
        prompt="Observe a table.",
        scene_contract={"events": [{"id": "observe"}]},
        assistant_review=request,
    )

    assert result["status"] == "scored"
    assert result["review_source"] == "assistant_local_review"
    assert result["vlm_model"] is None
    assert result["aggregate"]["final_score"] == 88.147
