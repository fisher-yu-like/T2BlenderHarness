import json

from PIL import Image

from videoact.real_video import assemble_mp4_from_pngs


def test_evaluate_real_run_builds_report_and_promotes_state(tmp_path):
    from scripts.evaluate_real_runs import evaluate_real_run
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner
    from videoact.real_artifacts import fingerprint_real_run

    contract = SceneContractBuilder().build("Observe a table.", duration_s=1.0, fps=3)
    plan = TrajectoryPlanner().plan(contract)
    (tmp_path / "frames").mkdir()
    frame_paths = []
    for frame in (1, 2, 3):
        path = tmp_path / "frames" / f"frame_{frame:06d}.png"
        Image.new("RGB", (8, 8), (frame, 0, 0)).save(path)
        frame_paths.append(path)
    (tmp_path / "frames" / "index.json").write_text(
        json.dumps({"frames": [{"frame": frame, "path": f"frame_{frame:06d}.png"} for frame in (1, 2, 3)]}),
        encoding="utf-8",
    )
    for name in ["scene_contract.json", "trajectory.json", "camera_plan.json", "blender_job.py"]:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "proxy.blend").write_bytes(b"blend")
    assemble_mp4_from_pngs(frame_paths, tmp_path / "proxy.mp4", fps=3)
    (tmp_path / "telemetry.json").write_text(
        json.dumps({
            "blender_version": "5.1.2",
            "frame_start": 1,
            "frame_end": 3,
            "fps": 3,
            "objects": {entity_id: {"keyframe_count": len(entity.states)} for entity_id, entity in plan.entities.items()},
            "camera": {"active": True},
        }),
        encoding="utf-8",
    )
    prompt_hash, plan_hash = "p", "t"
    manifest = {
        "run_id": "real-run",
        "case_id": "case-01",
        "split": "calibration",
        "prompt_hash": prompt_hash,
        "plan_hash": plan_hash,
        "harness_version": "h1",
        "evaluator_version": "deterministic-v1",
        "blender_version": "5.1.2",
        "fps": 3,
        "frame_start": 1,
        "frame_end": 3,
        "render_settings": {"resolution": [8, 8]},
        "fingerprint": fingerprint_real_run(
            prompt_hash=prompt_hash,
            plan_hash=plan_hash,
            harness_version="h1",
            evaluator_version="deterministic-v1",
            blender_version="5.1.2",
            render_settings={"resolution": [8, 8]},
        ),
        "state": "prepared",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "state.json").write_text(
        json.dumps({"case_id": "case-01", "state": "executing", "history": [{"state": "prepared", "metadata": {}}, {"state": "executing", "metadata": {}}]}),
        encoding="utf-8",
    )
    (tmp_path / "scene_contract.json").write_text(json.dumps(contract.model_dump(mode="json")), encoding="utf-8")
    (tmp_path / "trajectory.json").write_text(json.dumps(plan.model_dump(mode="json")), encoding="utf-8")
    (tmp_path / "camera_plan.json").write_text(json.dumps(plan.camera.model_dump(mode="json")), encoding="utf-8")

    result = evaluate_real_run(tmp_path)

    assert result["status"] == "pass"
    assert result["state"] == "evaluated"
    assert (tmp_path / "deterministic_report.json").exists()


def test_vlm_evaluation_runs_only_after_deterministic_pass(tmp_path):
    from evaluator.schemas import VLMJudgeResponse
    from scripts.evaluate_real_videos import evaluate_vlm_run

    (tmp_path / "frames").mkdir()
    frame_paths = []
    for frame_number in (1, 2, 3):
        frame = tmp_path / "frames" / f"frame_{frame_number:06d}.png"
        Image.new("RGB", (4, 4), (255, 255, 255)).save(frame)
        frame_paths.append(frame)
    (tmp_path / "frames" / "index.json").write_text(
        json.dumps({"frames": [{"frame": n, "path": f"frame_{n:06d}.png"} for n in (1, 2, 3)]}),
        encoding="utf-8",
    )
    (tmp_path / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )
    assemble_mp4_from_pngs(frame_paths, tmp_path / "proxy.mp4", fps=3)
    (tmp_path / "run_manifest.json").write_text(json.dumps({"case_id": "case-01"}), encoding="utf-8")

    class FakeProvider:
        def evaluate(self, **kwargs):
            return VLMJudgeResponse(
                prompt_compliance=100,
                physical_plausibility=100,
                camera_coverage=100,
                camera_innovation=100,
                character_trajectory=100,
                object_trajectory=100,
                event_timing=100,
                temporal_smoothness=100,
                visual_clarity=100,
                visible_evidence=["frame"],
                weaknesses=[],
                confidence=1.0,
            ), {"id": "fake-response"}

    result = evaluate_vlm_run(
        tmp_path,
        prompt="Observe a table.",
        scene_contract={"events": [{"id": "observe"}]},
        provider=FakeProvider(),
        scoring_policy="legacy-aggregate",
    )

    assert result["status"] == "scored"
    assert result["aggregate"]["final_score"] == 98
    assert result["video_probe"]["playable"] is True
    assert result["video_probe"]["frame_count"] >= 3
    assert (tmp_path / "vlm_report.json").exists()


def test_vlm_evaluation_requires_a_playable_proxy_video(tmp_path):
    from evaluator.schemas import VLMJudgeResponse
    from scripts.evaluate_real_videos import evaluate_vlm_run

    (tmp_path / "frames").mkdir()
    Image.new("RGB", (4, 4), (255, 255, 255)).save(tmp_path / "frames" / "frame_000001.png")
    (tmp_path / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )
    (tmp_path / "proxy.mp4").write_bytes(b"not-a-real-video")

    class ExplodingProvider:
        def evaluate(self, **kwargs):
            raise AssertionError("VLM must not run before the real video gate")

    result = evaluate_vlm_run(
        tmp_path,
        prompt="Observe a table.",
        scene_contract={"events": [{"id": "observe"}]},
        provider=ExplodingProvider(),
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "unplayable_proxy_video"


def test_low_confidence_visual_review_does_not_enter_score(tmp_path):
    from evaluator.schemas import VLMJudgeResponse
    from scripts.evaluate_real_videos import evaluate_vlm_run

    (tmp_path / "frames").mkdir()
    frame_paths = []
    for number in (1, 2, 3):
        path = tmp_path / "frames" / f"frame_{number:06d}.png"
        Image.new("RGB", (4, 4), (255, 255, 255)).save(path)
        frame_paths.append(path)
    assemble_mp4_from_pngs(frame_paths, tmp_path / "proxy.mp4", fps=3)
    (tmp_path / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )

    class LowConfidenceProvider:
        def evaluate(self, **kwargs):
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
                visible_evidence=["frames are ambiguous"],
                weaknesses=["critical event is not clear"],
                confidence=0.4,
            ), {"id": "low-confidence"}

    result = evaluate_vlm_run(
        tmp_path,
        prompt="Observe a table.",
        scene_contract={"events": [{"id": "observe", "start": 0, "end": 1}], "fps": 3},
        provider=LowConfidenceProvider(),
    )

    assert result["status"] == "needs_human_review"
    assert result["reason"] == "low_visual_review_confidence"
    assert "aggregate" not in result
