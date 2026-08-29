from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from videoact.real_video import assemble_mp4_from_pngs


def _write_real_example_run(root: Path, case_id: str) -> Path:
    run = root / case_id
    frames = run / "frames"
    frames.mkdir(parents=True)
    frame_paths = []
    for number in (1, 2, 3):
        frame = frames / f"frame_{number:06d}.png"
        Image.new("RGB", (16, 16), (number * 30, 20, 10)).save(frame)
        frame_paths.append(frame)
    (frames / "index.json").write_text(
        json.dumps({"frames": [{"path": path.name} for path in frame_paths]}), encoding="utf-8"
    )
    assemble_mp4_from_pngs(frame_paths, run / "proxy.mp4", fps=3)
    (run / "proxy.blend").write_bytes(b"real-blend-artifact")
    (run / "scene_contract.json").write_text("{}", encoding="utf-8")
    (run / "trajectory.json").write_text("{}", encoding="utf-8")
    (run / "camera_plan.json").write_text("{}", encoding="utf-8")
    source = run / "blender_job.py"
    source.write_text("import bpy\n", encoding="utf-8")
    code_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    (run / "telemetry.json").write_text(json.dumps({"case_id": case_id}), encoding="utf-8")
    (run / "director_plan.json").write_text(json.dumps({"case_id": case_id}), encoding="utf-8")
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": f"run-{case_id}",
                "case_id": case_id,
                "split": "train",
                "prompt_hash": "p" * 64,
                "plan_hash": "h" * 64,
                "harness_version": "h1",
                "evaluator_version": "v6",
                "blender_version": "5.1.2",
                "fps": 3,
                "frame_start": 1,
                "frame_end": 3,
                "render_settings": {},
                "fingerprint": "f" * 64,
                "state": "artifact_valid",
                "code_hash": code_hash,
            }
        ),
        encoding="utf-8",
    )
    return run


def _write_example_manifest(
    root: Path,
    run: Path,
    *,
    review_source: str = "human_review",
    plan_hash: str | None = None,
) -> None:
    root.mkdir(exist_ok=True)
    from videoact.run_manifest import hash_payload

    plan_path = run / "director_plan.json"
    actual_plan_hash = (
        hash_payload(json.loads(plan_path.read_text(encoding="utf-8")))
        if plan_path.is_file()
        else "h" * 64
    )
    record = {
        "case_id": run.name,
        "generation_mode": "agent",
        "artifact_root": str(run),
        "plan_hash": plan_hash or actual_plan_hash,
        "code_hash": hashlib.sha256((run / "blender_job.py").read_bytes()).hexdigest(),
        "artifact_status": "complete",
        "deterministic_score": 85,
        "review_source": review_source,
        "review_confidence": 0.9,
        "new_primitives": ["spiral_surface"],
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_codegen_example_validator_rejects_missing_real_artifact(tmp_path: Path) -> None:
    from scripts.validate_codegen_examples import validate_codegen_examples

    run = _write_real_example_run(tmp_path / "runs", "case-missing-video")
    (run / "proxy.mp4").unlink()
    examples = tmp_path / "examples"
    _write_example_manifest(examples, run)

    report = validate_codegen_examples(examples)

    assert report["status"] == "fail"
    assert any("proxy.mp4" in error for error in report["errors"])


def test_codegen_example_validator_rejects_artifact_only_review(tmp_path: Path) -> None:
    from scripts.validate_codegen_examples import validate_codegen_examples

    run = _write_real_example_run(tmp_path / "runs", "case-artifact-only")
    examples = tmp_path / "examples"
    _write_example_manifest(examples, run, review_source="frame_statistics")

    report = validate_codegen_examples(examples)

    assert report["status"] == "fail"
    assert any("review_source" in error for error in report["errors"])


def test_codegen_example_validator_requires_the_reviewed_director_plan(tmp_path: Path) -> None:
    from scripts.validate_codegen_examples import validate_codegen_examples

    run = _write_real_example_run(tmp_path / "runs", "case-missing-plan")
    (run / "director_plan.json").unlink()
    examples = tmp_path / "examples"
    _write_example_manifest(examples, run, plan_hash="h" * 64)

    report = validate_codegen_examples(examples)

    assert report["status"] == "fail"
    assert any("director_plan.json" in error for error in report["errors"])


def test_codegen_example_validator_rejects_mismatched_plan_hash(tmp_path: Path) -> None:
    from scripts.validate_codegen_examples import validate_codegen_examples

    run = _write_real_example_run(tmp_path / "runs", "case-mismatched-plan")
    examples = tmp_path / "examples"
    _write_example_manifest(examples, run, plan_hash="a" * 64)

    report = validate_codegen_examples(examples)

    assert report["status"] == "fail"
    assert any("plan_hash mismatch" in error for error in report["errors"])


def test_valid_codegen_context_loads_only_reviewed_plan_source_pairs(tmp_path: Path) -> None:
    from videoact.codegen_context import load_validated_context_examples

    run = _write_real_example_run(tmp_path / "runs", "case-reviewed")
    examples = tmp_path / "examples"
    _write_example_manifest(examples, run)

    loaded, status = load_validated_context_examples(examples)

    assert status == "pass"
    assert [item.case_id for item in loaded] == ["case-reviewed"]
    assert loaded[0].artifact_path == str(run.resolve())
    from videoact.run_manifest import hash_payload

    expected_plan_hash = hash_payload(json.loads((run / "director_plan.json").read_text(encoding="utf-8")))
    assert loaded[0].plan_hash == expected_plan_hash
    assert loaded[0].code_hash == hashlib.sha256((run / "blender_job.py").read_bytes()).hexdigest()


def test_promotion_report_requires_three_distinct_real_cases_and_never_edits_library(tmp_path: Path) -> None:
    from scripts.promote_fallback_primitives import build_promotion_report

    manifest = tmp_path / "code_manifest.jsonl"
    rows = [
        {
            "case_id": f"case-{index}",
            "status": "success",
            "artifact_status": "complete",
            "review_source": "human_review",
            "new_primitives": ["spiral_surface"],
        }
        for index in range(3)
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    library_file = tmp_path / "library.py"
    library_file.write_text("original", encoding="utf-8")

    report = build_promotion_report([manifest], library_roots=[library_file])

    assert report["status"] == "promotion_candidates"
    assert report["candidates"][0]["primitive"] == "spiral_surface"
    assert report["candidates"][0]["case_count"] == 3
    assert report["candidates"][0]["requires_human_approval"] is True
    assert library_file.read_text(encoding="utf-8") == "original"
