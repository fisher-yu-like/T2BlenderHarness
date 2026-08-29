from __future__ import annotations

import json

from PIL import Image


def _run(tmp_path):
    from videoact.real_artifacts import RealRunManifest, fingerprint_real_run

    root = tmp_path / "case-01"
    (root / "frames" / "animation").mkdir(parents=True)
    for frame in (1, 2, 3):
        Image.new("RGB", (8, 8), (frame, 20, 30)).save(root / "frames" / "animation" / f"frame_{frame:06d}.png")
    (root / "frames" / "index.json").write_text(
        json.dumps({"frames": [{"frame": 1, "path": "animation/frame_000001.png"}]}),
        encoding="utf-8",
    )
    (root / "proxy.mp4").write_bytes(b"not-a-video")
    manifest = RealRunManifest(
        run_id="run-01",
        case_id="case-01",
        split="train",
        prompt_hash="p",
        plan_hash="t",
        harness_version="h",
        evaluator_version="e",
        blender_version="b",
        fps=24,
        frame_start=1,
        frame_end=3,
        render_settings={"resolution": [8, 8]},
        fingerprint=fingerprint_real_run(
            prompt_hash="p", plan_hash="t", harness_version="h", evaluator_version="e",
            blender_version="b", render_settings={"resolution": [8, 8]},
        ),
        state="rendered",
    )
    (root / "run_manifest.json").write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")
    (root / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )
    (root / "scene_contract.json").write_text(json.dumps({"events": [], "fps": 24}), encoding="utf-8")
    return root


def _response():
    from evaluator.schemas import VLMJudgeResponse

    return VLMJudgeResponse(
        prompt_compliance=80, physical_plausibility=70, camera_coverage=90,
        camera_innovation=60, character_trajectory=75, object_trajectory=65,
        event_timing=80, temporal_smoothness=85, visual_clarity=90,
        appearance_detail=50, physical_realism=60, spatial_consistency=70,
        motion_naturalness=55, visual_presentation=65,
        visible_evidence=["frame 1"], weaknesses=["coarse proxy"], confidence=0.9,
    )


def test_visual_primary_runner_emits_independent_task_and_realism_channels(tmp_path, monkeypatch):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    root = _run(tmp_path)
    monkeypatch.setattr(
        "scripts.evaluate_real_videos.probe_mp4",
        lambda *_args, **_kwargs: {"playable": True, "frame_count": 3, "fps": 24.0, "duration_s": 0.125},
    )

    class Provider:
        model_alias = "gpt-5.6-luna"

        def evaluate(self, **_kwargs):
            return _response(), {"id": "review-1"}

    result = evaluate_vlm_run(
        root,
        prompt="Alice carries the red cup.",
        scene_contract={"events": [], "fps": 24},
        provider=Provider(),
        scoring_policy="visual-primary-v6",
    )

    assert result["status"] == "scored"
    assert result["task_score"] is not None
    assert result["task_score"] != result["realism_score"]
    assert result["score_channels"]["combined"] is False
    assert result["visual_primary"]["source"] == "gpt-5.6-luna"


def test_codex_local_visual_review_preserves_local_review_provenance(tmp_path, monkeypatch):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    root = _run(tmp_path)
    monkeypatch.setattr(
        "scripts.evaluate_real_videos.probe_mp4",
        lambda *_args, **_kwargs: {"playable": True, "frame_count": 3, "fps": 24.0, "duration_s": 0.125},
    )
    review = _response().model_dump(mode="json")

    result = evaluate_vlm_run(
        root,
        prompt="Alice carries the red cup.",
        scene_contract={"events": [], "fps": 24},
        assistant_review={"review_source": "codex_local_visual_review", "scores": review},
        scoring_policy="visual-primary-v6",
    )

    assert result["status"] == "scored"
    assert result["review_source"] == "codex_local_visual_review"
    assert result["realism_score_kind"] == "independent_review_fused"
