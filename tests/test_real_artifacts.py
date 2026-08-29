import json

from PIL import Image

from videoact.real_video import assemble_mp4_from_pngs


def manifest_payload():
    return {
        "run_id": "real-run-001",
        "case_id": "case-01",
        "split": "calibration",
        "prompt_hash": "prompt-hash",
        "plan_hash": "plan-hash",
        "harness_version": "h1",
        "evaluator_version": "deterministic-v1",
        "blender_version": "4.3.0",
        "fps": 24,
        "frame_start": 1,
        "frame_end": 24,
        "render_settings": {"engine": "BLENDER_EEVEE_NEXT", "resolution": [256, 256]},
        "fingerprint": "fingerprint",
        "state": "rendered",
    }


def make_complete_run(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    frame_paths = []
    for frame in (1, 12, 24):
        path = frames / f"frame_{frame:06d}.png"
        Image.new("RGB", (8, 8), (frame, 0, 0)).save(path)
        frame_paths.append(path)
    (frames / "index.json").write_text(
        json.dumps({"frames": [{"frame": frame, "path": f"frame_{frame:06d}.png"} for frame in (1, 12, 24)]}),
        encoding="utf-8",
    )
    for name, content in {
        "proxy.blend": b"blend",
        "telemetry.json": b"{}",
        "scene_contract.json": b"{}",
        "trajectory.json": b"{}",
        "camera_plan.json": b"{}",
        "blender_job.py": b"# bpy job",
    }.items():
        (tmp_path / name).write_bytes(content)
    assemble_mp4_from_pngs(frame_paths, tmp_path / "proxy.mp4", fps=3)
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest_payload()), encoding="utf-8")


def test_artifact_gate_accepts_complete_readable_run(tmp_path):
    from videoact.real_artifacts import RealArtifactGate

    make_complete_run(tmp_path)
    report = RealArtifactGate().validate(tmp_path)

    assert report.artifact_status == "complete"
    assert report.readable_frame_count == 3
    assert report.video_frame_count >= 3
    assert report.video_fps > 0
    assert report.video_duration_s > 0
    assert report.hard_failures == []


def test_artifact_gate_hard_fails_missing_video_and_invalid_frame(tmp_path):
    from videoact.real_artifacts import RealArtifactGate

    make_complete_run(tmp_path)
    (tmp_path / "proxy.mp4").unlink()
    (tmp_path / "frames" / "frame_000012.png").write_bytes(b"not-an-image")

    report = RealArtifactGate().validate(tmp_path)

    assert report.artifact_status == "incomplete"
    assert "missing_artifact:proxy.mp4" in report.hard_failures


def test_artifact_gate_rejects_non_playable_mp4_even_when_file_exists(tmp_path):
    from videoact.real_artifacts import RealArtifactGate

    fixture = tmp_path / "case"
    fixture.mkdir()
    for name in [
        "run_manifest.json",
        "scene_contract.json",
        "trajectory.json",
        "camera_plan.json",
        "blender_job.py",
        "proxy.blend",
        "telemetry.json",
    ]:
        (fixture / name).write_text("{}", encoding="utf-8")
    (fixture / "proxy.mp4").write_bytes(b"not-a-real-video")
    (fixture / "frames").mkdir()
    (fixture / "frames" / "index.json").write_text(json.dumps({"frames": []}), encoding="utf-8")

    report = RealArtifactGate().validate(fixture)

    assert report.artifact_status == "incomplete"
    assert "unplayable_video:proxy.mp4" in report.hard_failures
    assert "insufficient_readable_frames:0" in report.hard_failures


def test_real_run_fingerprint_and_resume_match_are_stable():
    from videoact.real_artifacts import fingerprint_real_run, resume_matches
    from videoact.real_artifacts import RealRunManifest

    fingerprint = fingerprint_real_run(
        prompt_hash="p",
        plan_hash="t",
        harness_version="h1",
        evaluator_version="e1",
        blender_version="4.3.0",
        render_settings={"resolution": [256, 256]},
    )
    manifest = RealRunManifest.model_validate({**manifest_payload(), "fingerprint": fingerprint})

    assert fingerprint == fingerprint_real_run(
        prompt_hash="p",
        plan_hash="t",
        harness_version="h1",
        evaluator_version="e1",
        blender_version="4.3.0",
        render_settings={"resolution": [256, 256]},
    )
    assert resume_matches(manifest, {"fingerprint": fingerprint}) is True
    assert resume_matches(manifest, {"fingerprint": "different"}) is False


def test_real_artifact_gate_detects_job_source_hash_mismatch(tmp_path):
    from videoact.real_artifacts import RealArtifactGate

    make_complete_run(tmp_path)
    payload = manifest_payload()
    payload["code_hash"] = "0" * 64
    (tmp_path / "run_manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    report = RealArtifactGate().validate(tmp_path)

    assert "job_source_hash_mismatch" in report.hard_failures


def test_real_artifact_gate_exposes_stable_aggregate_artifact_hash(tmp_path):
    from videoact.real_artifacts import RealArtifactGate

    make_complete_run(tmp_path)
    first = RealArtifactGate().validate(tmp_path)
    (tmp_path / "frames" / "frame_000012.png").write_bytes(
        (tmp_path / "frames" / "frame_000024.png").read_bytes()
    )
    second = RealArtifactGate().validate(tmp_path)

    assert first.artifact_hash
    assert second.artifact_hash
    assert first.artifact_hash != second.artifact_hash
