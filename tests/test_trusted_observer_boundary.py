from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_trusted_fixture(root: Path, *, source_hash: str | None = None) -> None:
    from videoact.observer_contract import OBSERVER_SCHEMA_VERSION, write_observer_request

    candidate = root / "candidate.blend"
    candidate.write_bytes(b"candidate-blend-v1")
    observer = root / "trusted_observer.py"
    observer.write_text("# fixed observer source\n", encoding="utf-8")
    telemetry = {
        "schema_version": OBSERVER_SCHEMA_VERSION,
        "frame_start": 1,
        "frame_end": 24,
        "fps": 24,
        "entities": {"actor_a": {"kind": "character"}},
        "observations": [],
        "camera_observations": [],
    }
    telemetry_path = root / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")
    request = write_observer_request(
        root / "observer_request.json",
        candidate_blend_hash=_sha256(candidate),
        observer_source_hash=source_hash or _sha256(observer),
    )
    manifest = {
        "schema_version": OBSERVER_SCHEMA_VERSION,
        "trusted": True,
        "request_nonce": request["nonce"],
        "candidate_blend_hash": _sha256(candidate),
        "observer_source_hash": _sha256(observer),
        "telemetry_hash": _sha256(telemetry_path),
    }
    (root / "telemetry_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_generated_self_report_is_not_accepted_as_trusted_telemetry(tmp_path: Path) -> None:
    from videoact.observer_contract import read_trusted_observer_output

    (tmp_path / "telemetry.json").write_text(
        json.dumps({"handoff_success": True, "objects": {"actor_a": {}}}), encoding="utf-8"
    )
    report = read_trusted_observer_output(tmp_path, observer_source_path=tmp_path / "trusted_observer.py")

    assert report["status"] == "fail"
    assert report["telemetry"] is None
    assert "missing_trusted_observer_output" in report["failures"]


def test_observer_output_is_rejected_when_candidate_blend_changes(tmp_path: Path) -> None:
    from videoact.observer_contract import read_trusted_observer_output

    _write_trusted_fixture(tmp_path)
    (tmp_path / "candidate.blend").write_bytes(b"candidate-blend-mutated")

    report = read_trusted_observer_output(tmp_path, observer_source_path=tmp_path / "trusted_observer.py")

    assert report["status"] == "fail"
    assert report["telemetry"] is None
    assert "candidate_blend_hash_mismatch" in report["failures"]


def test_observer_source_hash_is_allowlisted_by_exact_content(tmp_path: Path) -> None:
    from videoact.observer_contract import read_trusted_observer_output

    _write_trusted_fixture(tmp_path)
    (tmp_path / "trusted_observer.py").write_text("# modified observer\n", encoding="utf-8")

    report = read_trusted_observer_output(tmp_path, observer_source_path=tmp_path / "trusted_observer.py")

    assert report["status"] == "fail"
    assert report["telemetry"] is None
    assert "observer_source_hash_mismatch" in report["failures"]


def test_valid_observer_output_returns_raw_observations_only(tmp_path: Path) -> None:
    from videoact.observer_contract import read_trusted_observer_output

    _write_trusted_fixture(tmp_path)

    report = read_trusted_observer_output(tmp_path, observer_source_path=tmp_path / "trusted_observer.py")

    assert report["status"] == "pass"
    assert report["trusted"] is True
    assert report["telemetry"]["entities"]["actor_a"]["kind"] == "character"


def test_observer_semantic_claims_are_rejected_even_with_matching_hashes(tmp_path: Path) -> None:
    from videoact.observer_contract import read_trusted_observer_output

    _write_trusted_fixture(tmp_path)
    telemetry_path = tmp_path / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    telemetry["handoff_success"] = True
    telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")
    manifest = json.loads((tmp_path / "telemetry_manifest.json").read_text(encoding="utf-8"))
    manifest["telemetry_hash"] = _sha256(telemetry_path)
    (tmp_path / "telemetry_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = read_trusted_observer_output(tmp_path, observer_source_path=tmp_path / "trusted_observer.py")

    assert report["status"] == "fail"
    assert report["telemetry"] is None
    assert "observer_emitted_forbidden_semantic_field:handoff_success" in report["failures"]


def test_observer_request_can_enable_bounded_mesh_geometry_for_physics_narrow_phase() -> None:
    from videoact.observer_contract import create_observer_request

    request = create_observer_request(
        candidate_blend_hash="c" * 64,
        observer_source_hash="o" * 64,
        mesh_entity_ids=["prop_01", "actor_a"],
    )

    assert request["mesh_entity_ids"] == ["prop_01", "actor_a"]


def test_executor_quarantines_generated_telemetry_before_observer(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "case-a"
    job_dir.mkdir()
    (job_dir / "blender_job.py").write_text("# generated job\n", encoding="utf-8")
    (job_dir / "proxy.blend").write_bytes(b"candidate")
    (job_dir / "candidate.blend").write_bytes(b"candidate")
    observer_hash = _sha256(renderer.OBSERVER_SOURCE)
    (job_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "code_hash": _sha256(job_dir / "blender_job.py"),
                "trusted_observer_required": True,
                "observer_source_hash": observer_hash,
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "telemetry.json").write_text(json.dumps({"handoff_success": True}), encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if str(renderer.OBSERVER_SOURCE) in command:
            request = json.loads((job_dir / "observer_request.json").read_text(encoding="utf-8"))
            telemetry = {
                "schema_version": "trusted-observer-v1",
                "frame_start": 1,
                "frame_end": 2,
                "fps": 2,
                "entities": {},
                "observations": [],
                "camera_observations": [],
            }
            telemetry_path = job_dir / "telemetry.json"
            telemetry_path.write_text(json.dumps(telemetry, sort_keys=True), encoding="utf-8")
            (job_dir / "telemetry_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "trusted-observer-v1",
                        "trusted": True,
                        "request_nonce": request["nonce"],
                        "candidate_blend_hash": _sha256(job_dir / "candidate.blend"),
                        "observer_source_hash": observer_hash,
                        "telemetry_hash": _sha256(telemetry_path),
                    }
                ),
                encoding="utf-8",
            )
            frames = job_dir / "frames" / "animation"
            frames.mkdir(parents=True)
            from PIL import Image

            for frame in (1, 2, 3):
                Image.new("RGB", (8, 8), (frame, 0, 0)).save(frames / f"frame_{frame:06d}.png")
            (job_dir / "frames" / "index.json").write_text(json.dumps({"frames": []}), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    result = renderer._run_one(job_dir, "blender", timeout_s=10, max_retries=0)

    assert result["status"] == "success"
    assert len(calls) == 2
    assert (job_dir / "untrusted_candidate_telemetry.json").is_file()
    trusted = json.loads((job_dir / "telemetry.json").read_text(encoding="utf-8"))
    assert "handoff_success" not in trusted


def test_executor_rejects_trusted_observer_run_without_candidate_blend(tmp_path: Path, monkeypatch) -> None:
    import scripts.render_proxy_jobs_parallel as renderer

    job_dir = tmp_path / "case-missing-candidate"
    job_dir.mkdir()
    job = job_dir / "blender_job.py"
    job.write_text("# generated job\n", encoding="utf-8")
    (job_dir / "proxy.blend").write_bytes(b"proxy-only")
    observer_hash = _sha256(renderer.OBSERVER_SOURCE)
    (job_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "code_hash": _sha256(job),
                "trusted_observer_required": True,
                "observer_source_hash": observer_hash,
            }
        ),
        encoding="utf-8",
    )

    result = renderer._run_trusted_observer(
        job_dir,
        blender_bin="blender",
        timeout_s=1,
        manifest_payload=json.loads((job_dir / "run_manifest.json").read_text(encoding="utf-8")),
    )

    assert result["status"] == "failed"
    assert result["error"] == "candidate_blend_missing"
    assert not (job_dir / "candidate.blend").exists()
