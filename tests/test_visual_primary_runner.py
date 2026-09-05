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


def test_missing_deterministic_report_is_unavailable_without_visual_review(tmp_path):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    calls = []

    class Provider:
        def evaluate(self, **kwargs):
            calls.append(kwargs)
            raise AssertionError("visual review must not run without deterministic evidence")

    result = evaluate_vlm_run(
        tmp_path,
        prompt="Alice carries the red cup.",
        scene_contract={"events": [], "fps": 24},
        provider=Provider(),
        scoring_policy="visual-primary-v6",
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "deterministic_report_missing"
    assert result["review_source"] == "not_evaluated"
    assert calls == []
    assert json.loads((tmp_path / "vlm_report.json").read_text(encoding="utf-8")) == result


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


def test_formal_visual_result_includes_three_layer_status_from_real_artifacts(tmp_path, monkeypatch):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    root = _run(tmp_path)
    monkeypatch.setattr(
        "scripts.evaluate_real_videos.probe_mp4",
        lambda *_args, **_kwargs: {"playable": True, "frame_count": 3, "fps": 24.0, "duration_s": 0.125},
    )
    (root / "artifact_report.json").write_text(json.dumps({"artifact_status": "complete"}), encoding="utf-8")
    (root / "telemetry.json").write_text(
        json.dumps({"observations": [{"frame": 1}, {"frame": 2}, {"frame": 3}]}), encoding="utf-8"
    )

    class Provider:
        model_alias = "gpt-5.6-luna"

        def evaluate(self, **_kwargs):
            return _response_with_event_score(), {"id": "review-event-gate"}

    result = evaluate_vlm_run(
        root,
        prompt="Alice hands the red cup to Bob.",
        scene_contract={
            "events": [{"id": "handoff_01", "start": 0, "end": 1}],
            "must_show": ["handoff_01"],
            "fps": 24,
            "entities": [
                {"id": "actor_a", "kind": "character"},
                {"id": "actor_b", "kind": "character"},
                {"id": "red_cup", "kind": "prop"},
            ],
        },
        provider=Provider(),
        scoring_policy="scoring-v7-independent-channels",
    )

    assert result["status"] == "scored"
    assert result["evaluation_result"]["execution_status"] == "valid"
    assert result["evaluation_result"]["semantic_status"] == "failed_required_event"
    assert result["evaluation_result"]["task_score"] == 49


def test_formal_visual_result_does_not_trust_generated_telemetry(tmp_path, monkeypatch):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    root = _run(tmp_path)
    monkeypatch.setattr(
        "scripts.evaluate_real_videos.probe_mp4",
        lambda *_args, **_kwargs: {"playable": True, "frame_count": 3, "fps": 24.0, "duration_s": 0.125},
    )
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["trusted_observer_required"] = True
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "artifact_report.json").write_text(json.dumps({"artifact_status": "complete"}), encoding="utf-8")
    (root / "telemetry.json").write_text(
        json.dumps({"observations": [{"frame": 1}, {"frame": 2}, {"frame": 3}]}),
        encoding="utf-8",
    )

    class Provider:
        model_alias = "gpt-5.6-luna"

        def evaluate(self, **_kwargs):
            return _response_with_event_score(), {"id": "generated-telemetry-must-not-count"}

    result = evaluate_vlm_run(
        root,
        prompt="Alice carries the red cup.",
        scene_contract={"events": [], "fps": 24},
        provider=Provider(),
        scoring_policy="scoring-v7-independent-channels",
    )

    assert result["status"] == "scored"
    assert result["evaluation_result"]["execution_status"] == "invalid"
    assert "runtime_observations_missing" in result["evaluation_result"]["reasons"]


def test_observe_only_event_does_not_require_discrete_event_timing_evidence(tmp_path, monkeypatch):
    from scripts.evaluate_real_videos import evaluate_vlm_run

    root = _run(tmp_path)
    monkeypatch.setattr(
        "scripts.evaluate_real_videos.probe_mp4",
        lambda *_args, **_kwargs: {"playable": True, "frame_count": 3, "fps": 24.0, "duration_s": 0.125},
    )
    response = _response_with_event_score()
    evidence = dict(response.dimension_evidence or {})
    evidence.pop("event_timing", None)
    response = response.model_copy(update={"dimension_evidence": evidence})

    class Provider:
        model_alias = "codex_local_visual_review"
        provider_kind = "codex_exec_visual_review"

        def evaluate(self, **_kwargs):
            return response, {"id": "observe-only-review"}

    result = evaluate_vlm_run(
        root,
        prompt="A garden remains visible.",
        scene_contract={
            "events": [{"id": "observe", "start": 0.0, "end": 1.0, "description": "observe"}],
            "must_show": ["observe"],
            "duration_s": 1.0,
            "fps": 24,
        },
        provider=Provider(),
        scoring_policy="scoring-v7-independent-channels",
    )

    assert result["status"] == "scored"
    assert result["visual_primary"]["applicability"]["event_timing"] is False
    assert result["visual_primary"]["dimension_evidence"]["event_timing"]["applicability"] is False


def _response_with_event_score():
    from evaluator.schemas import VLMJudgeResponse

    evidence = {
        name: {
            "confidence": 0.9,
            "evidence_completeness": 1.0,
            "evidence_refs": ["frame:1", "frame:2"],
        }
        for name in (
            "prompt_compliance",
            "physical_plausibility",
            "camera_coverage",
            "camera_innovation",
            "character_trajectory",
            "object_trajectory",
            "event_timing",
            "temporal_smoothness",
            "visual_clarity",
            "appearance_detail",
            "physical_realism",
            "spatial_consistency",
            "motion_naturalness",
            "visual_presentation",
        )
    }
    return VLMJudgeResponse(
        prompt_compliance=80,
        physical_plausibility=80,
        camera_coverage=80,
        camera_innovation=80,
        character_trajectory=80,
        object_trajectory=80,
        event_timing=80,
        temporal_smoothness=80,
        visual_clarity=80,
        appearance_detail=80,
        physical_realism=80,
        spatial_consistency=80,
        motion_naturalness=80,
        visual_presentation=80,
        event_scores={"handoff_01": 10},
        dimension_evidence=evidence,
        visible_evidence=["frame 1"],
        weaknesses=["handoff is not visible"],
        confidence=0.9,
    )
