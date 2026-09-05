from __future__ import annotations

from pathlib import Path

from PIL import Image


SEMANTIC_FIELDS = {
    "prompt_compliance",
    "event_timing",
    "physical_plausibility",
    "physical_realism",
    "object_trajectory",
    "character_trajectory",
}


def _frames(tmp_path: Path) -> list[Path]:
    paths = []
    for index, color in enumerate(((15, 15, 15), (80, 30, 20), (20, 70, 35))):
        path = tmp_path / f"frame_{index:03d}.png"
        Image.new("RGB", (96, 96), color).save(path)
        paths.append(path)
    return paths


def test_frame_statistics_cannot_claim_semantic_dimensions(tmp_path: Path):
    from scripts.author_three_arm_local_reviews import score_sample_frames

    review = score_sample_frames(_frames(tmp_path))

    assert review["review_source"] == "frame_statistics"
    assert review["method"] == "frame_statistics_only-v1"
    assert all(review["scores"][field] is None for field in SEMANTIC_FIELDS)
    assert review["artifact_health"]["readable"] is True
    assert review["score"] is None


def test_frame_statistics_can_only_report_low_level_observations(tmp_path: Path):
    from scripts.author_three_arm_local_reviews import score_sample_frames

    review = score_sample_frames(_frames(tmp_path))

    measurable = {
        "visual_clarity",
        "temporal_smoothness",
        "appearance_detail",
        "spatial_consistency",
        "visual_presentation",
    }
    assert measurable.issubset(review["scores"])
    assert all(value is None or 0 <= value <= 100 for value in review["scores"].values())


def test_visual_evidence_report_is_artifact_health_not_quality_score(tmp_path: Path):
    from evaluator.realism import score_realism
    from evaluator.visual_evidence import inspect_render_frames

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    frames = _frames(source_dir)
    run_dir = tmp_path / "run"
    (run_dir / "frames").mkdir(parents=True)
    for frame in frames:
        (run_dir / "frames" / frame.name).write_bytes(frame.read_bytes())

    report = inspect_render_frames(run_dir)

    assert report["score"] is None
    assert report["score_kind"] == "artifact_health_only"
    assert report["artifact_health"]["readable"] is True
    realism = score_realism(
        {
            "coverage_score": 100,
            "topology_score": 100,
            "primitive_score": 100,
            "semantic_score": 100,
            "structural_score": 100,
            "hard_gate_failed": False,
        },
        {"status": "complete", "source": "frame_statistics", "score": 100},
    )
    assert realism["score"] == 48.0
    assert realism["evaluator_version"] == "realism-v5-independent-review-boundary"


def test_render_frame_health_rejects_uniform_frames_but_accepts_spatial_content(tmp_path: Path):
    from evaluator.visual_evidence import assess_render_frame_health

    blank = []
    content = []
    for index in range(3):
        blank_path = tmp_path / f"blank_{index:03d}.png"
        Image.new("RGB", (32, 32), (51, 51, 51)).save(blank_path)
        blank.append(blank_path)
        content_path = tmp_path / f"content_{index:03d}.png"
        image = Image.new("RGB", (32, 32), (51, 51, 51))
        for x in range(8, 24):
            for y in range(8, 24):
                image.putpixel((x, y), (210, 80 + index * 10, 30))
        image.save(content_path)
        content.append(content_path)

    blank_health = assess_render_frame_health(blank)
    content_health = assess_render_frame_health(content)

    assert blank_health["status"] == "blank"
    assert blank_health["readable_count"] == 3
    assert content_health["status"] == "visible"
