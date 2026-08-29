from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from videoact.real_video import assemble_mp4_from_pngs


DIMENSIONS = (
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


def _write_valid_bundle(root: Path, case_count: int = 30):
    root.mkdir(exist_ok=True)
    manifest = []
    scores = []
    sample_labels = ("sample_a", "sample_b", "sample_c")
    fixture_frames = root / "fixture_frames"
    fixture_frames.mkdir(exist_ok=True)
    frame_paths = []
    for number in (1, 2, 3):
        frame = fixture_frames / f"frame_{number:06d}.png"
        Image.new("RGB", (16, 16), (number * 30, 20, 10)).save(frame)
        frame_paths.append(frame)
    fixture_video = root / "fixture.mp4"
    assemble_mp4_from_pngs(frame_paths, fixture_video, fps=3)
    for index in range(case_count):
        case_id = f"gold-{index:03d}"
        for sample_id in sample_labels:
            destination = root / "videos" / case_id / f"{sample_id}.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fixture_video, destination)
        manifest.append(
            {
                "case_id": case_id,
                "prompt": f"A named actor performs action {index}.",
                "blind_group": f"sample-{index:03d}",
                "sampled_frames": {
                    label: [f"frames/{case_id}/{label}/frame_01.png"]
                    for label in sample_labels
                },
                "sampled_videos": {
                    label: str(root / "videos" / case_id / f"{label}.mp4")
                    for label in sample_labels
                },
                "source_fingerprint": f"sha-{index:03d}",
            }
        )
        for annotator in ("annotator-1", "annotator-2"):
            for sample_id in sample_labels:
                scores.append(
                    {
                        "case_id": case_id,
                        "sample_id": sample_id,
                        "annotator_id": annotator,
                        "scores": {dimension: 50 for dimension in DIMENSIONS},
                        "weaknesses": ["limited detail"],
                    }
                )
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
    )
    (root / "human_scores.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in scores), encoding="utf-8"
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "golden-review-v1",
                "patch_selection_allowed": False,
                "arms_hidden": True,
                "annotators": ["annotator-1", "annotator-2"],
                "inter_rater_agreement": {dimension: {"metric": "icc", "value": 0.72} for dimension in DIMENSIONS},
                "reproducibility_seed": 20260827,
            }
        ),
        encoding="utf-8",
    )


def test_golden_review_validator_accepts_blinded_two_annotator_bundle(tmp_path: Path):
    from scripts.validate_golden_review_set import validate_golden_review_set

    _write_valid_bundle(tmp_path)

    report = validate_golden_review_set(tmp_path)

    assert report["status"] == "pass"
    assert report["case_count"] == 30
    assert report["annotators_per_case"] == 2
    assert report["patch_selection_allowed"] is False


def test_golden_review_validator_rejects_arm_leak_and_missing_dimension(tmp_path: Path):
    from scripts.validate_golden_review_set import validate_golden_review_set

    _write_valid_bundle(tmp_path)
    manifest_path = tmp_path / "manifest.jsonl"
    first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
    first["arm"] = "trained"
    manifest_path.write_text(
        "\n".join([json.dumps(first), *manifest_path.read_text(encoding="utf-8").splitlines()[1:]]) + "\n",
        encoding="utf-8",
    )
    score_path = tmp_path / "human_scores.jsonl"
    rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["scores"].pop("event_timing")
    score_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = validate_golden_review_set(tmp_path)

    assert report["status"] == "fail"
    assert any("arm" in error for error in report["errors"])
    assert any("event_timing" in error for error in report["errors"])


def test_golden_review_validator_requires_two_annotators_for_each_blind_video(tmp_path: Path):
    _write_valid_bundle(tmp_path)
    score_path = tmp_path / "human_scores.jsonl"
    rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
    rows = [
        row
        for row in rows
        if not (row["case_id"] == "gold-000" and row["sample_id"] == "sample_b")
    ]
    score_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    from scripts.validate_golden_review_set import validate_golden_review_set

    report = validate_golden_review_set(tmp_path)

    assert report["status"] == "fail"
    assert any("sample_b" in error for error in report["errors"])


def test_golden_review_validator_rejects_boolean_scores(tmp_path: Path):
    _write_valid_bundle(tmp_path)
    score_path = tmp_path / "human_scores.jsonl"
    rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["scores"]["camera_coverage"] = True
    score_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    from scripts.validate_golden_review_set import validate_golden_review_set

    report = validate_golden_review_set(tmp_path)

    assert report["status"] == "fail"
    assert any("camera_coverage" in error and "numeric" in error for error in report["errors"])


def test_finalize_golden_review_calculates_agreement_and_updates_metadata(tmp_path: Path):
    _write_valid_bundle(tmp_path)

    from scripts.finalize_golden_review import finalize_golden_review

    report = finalize_golden_review(tmp_path)
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert report["status"] == "pass"
    assert report["finalized"] is True
    assert metadata["status"] == "annotations_finalized"
    assert metadata["annotations_complete"] is True
    assert metadata["agreement_method"] == "icc_2_1_first_two_sorted_annotators"
    assert set(metadata["inter_rater_agreement"]) == set(DIMENSIONS)
    assert all(item["metric"] == "icc_2_1" for item in metadata["inter_rater_agreement"].values())


def test_finalize_cli_is_fail_closed_with_structured_error(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "scripts/finalize_golden_review.py", "--root", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "fail"
    assert "metadata" in payload["error"]


def test_golden_review_validator_cli_accepts_named_root(tmp_path: Path, monkeypatch):
    import sys

    import scripts.validate_golden_review_set as validator

    observed = {}
    monkeypatch.setattr(sys, "argv", ["validate_golden_review_set.py", "--root", str(tmp_path)])
    monkeypatch.setattr(
        validator,
        "validate_golden_review_set",
        lambda root: observed.update(root=str(root)) or {"status": "pass"},
    )

    assert validator.main() == 0
    assert observed["root"] == str(tmp_path)


def _write_real_source_tree(root: Path, case_count: int = 30):
    dataset_root = root / "dataset"
    dataset_root.mkdir()
    records = []
    for index in range(case_count):
        records.append(
            {
                "case_id": f"case-{index:03d}",
                "category": "camera_motion",
                "prompt": f"A named actor performs action {index}.",
                "split": "train",
            }
        )
    (dataset_root / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    source_frames = root / "source_frames"
    source_frames.mkdir()
    frame_paths = []
    for number in (1, 2, 3):
        frame = source_frames / f"frame_{number:06d}.png"
        Image.new("RGB", (16, 16), (number * 30, 20, 10)).save(frame)
        frame_paths.append(frame)
    source_mp4 = root / "source.mp4"
    assemble_mp4_from_pngs(frame_paths, source_mp4, fps=3)

    arm_roots = {}
    for arm in ("current", "pretrain", "direct"):
        arm_root = root / arm
        arm_roots[arm] = arm_root
        for record in records:
            run = arm_root / "real" / "train" / record["case_id"]
            animation = run / "frames" / "animation"
            animation.mkdir(parents=True)
            for frame in frame_paths:
                shutil.copy2(frame, animation / frame.name)
            (run / "scene_contract.json").write_text(
                json.dumps({"fps": 3, "events": [], "must_show": []}),
                encoding="utf-8",
            )
            shutil.copy2(source_mp4, run / "proxy.mp4")
    return dataset_root, arm_roots


def test_golden_builder_copies_blind_videos_and_publishes_video_paths(tmp_path: Path):
    dataset_root, arm_roots = _write_real_source_tree(tmp_path)
    output_root = tmp_path / "golden"

    from scripts.build_golden_review_set import build_golden_review_set
    from videoact.real_artifacts import probe_mp4

    report = build_golden_review_set(
        dataset_root=dataset_root,
        arm_roots=arm_roots,
        output_root=output_root,
        sample_count=30,
        seed=20260827,
    )

    assert report["status"] == "awaiting_human_annotations"
    manifest_rows = [
        json.loads(line)
        for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert set(manifest_rows[0]["sampled_videos"]) == {"sample_a", "sample_b", "sample_c"}
    video = output_root / "videos" / manifest_rows[0]["case_id"] / "sample_a.mp4"
    assert video.is_file()
    assert probe_mp4(video)["playable"] is True
    assert len(list((output_root / "videos").rglob("*.mp4"))) == 90


def test_golden_builder_uses_verbatim_source_prompt_for_review(tmp_path: Path):
    dataset_root, arm_roots = _write_real_source_tree(tmp_path)
    manifest_path = dataset_root / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    records[0]["source_prompt"] = "A raw benchmark prompt with a camera orbit."
    records[0]["source_dataset"] = "VBench-2.0"
    records[0]["source_dimension"] = "Camera_Motion"
    records[0]["source_index"] = 42
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    from scripts.build_golden_review_set import build_golden_review_set

    output_root = tmp_path / "golden"
    report = build_golden_review_set(
        dataset_root=dataset_root,
        arm_roots=arm_roots,
        output_root=output_root,
        sample_count=30,
        seed=20260827,
    )

    rows = [json.loads(line) for line in (output_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    first = next(row for row in rows if row["case_id"] == "case-000")
    assert first["prompt"] == "A raw benchmark prompt with a camera orbit."
    assert first["source_prompt"] == first["prompt"]
    assert first["prompt_origin"] == "benchmark_verbatim"
    assert first["source_dataset"] == "VBench-2.0"
    assert report["video_count"] == 90
    assert report["comparison_only"] is True
    assert report["render_prompt_mismatch_count"] == 1
    metadata = json.loads((output_root / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["render_prompt_mismatch_count"] == 1
    assert metadata["comparison_only"] is True


def test_golden_review_validator_rejects_render_prompt_mismatch(tmp_path: Path):
    _write_valid_bundle(tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["render_prompt_mismatch_count"] = 1
    metadata["comparison_only"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    from scripts.validate_golden_review_set import validate_golden_review_set

    report = validate_golden_review_set(tmp_path)

    assert report["status"] == "fail"
    assert any("render prompt" in error for error in report["errors"])


def test_golden_builder_requires_three_valid_video_sources(tmp_path: Path):
    from scripts.build_golden_review_set import build_golden_review_set

    with pytest.raises(ValueError, match="three"):
        build_golden_review_set(
            dataset_root=tmp_path / "dataset",
            arm_roots={"pretrain": tmp_path / "pretrain", "trained": tmp_path / "trained"},
            output_root=tmp_path / "golden",
        )


def test_golden_builder_can_limit_calibration_to_non_frozen_splits(tmp_path: Path):
    dataset_root, arm_roots = _write_real_source_tree(tmp_path, case_count=31)
    manifest_path = dataset_root / "manifest.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    for index, record in enumerate(records):
        record["split"] = "test" if index == 0 else "train"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    from scripts.build_golden_review_set import build_golden_review_set

    report = build_golden_review_set(
        dataset_root=dataset_root,
        arm_roots=arm_roots,
        output_root=tmp_path / "golden",
        sample_count=30,
        include_splits={"train"},
    )

    assert report["case_count"] == 30
    rows = [json.loads(line) for line in (tmp_path / "golden" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["split"] == "train" for row in rows)
