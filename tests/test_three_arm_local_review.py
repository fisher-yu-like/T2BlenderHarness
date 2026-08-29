from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_frame_statistics_is_derived_from_real_frames_without_semantic_scores(tmp_path):
    module = _load("author_three_arm_local_reviews", ROOT / "scripts" / "author_three_arm_local_reviews.py")
    paths = []
    for index, color in enumerate(((0, 0, 0), (255, 255, 255), (80, 80, 80))):
        path = tmp_path / f"frame_{index}.png"
        Image.new("RGB", (32, 32), color).save(path)
        paths.append(str(path))

    review = module.score_sample_frames(paths)
    assert "prompt_compliance" in review["scores"]
    assert review["scores"]["prompt_compliance"] is None
    assert review["review_source"] == "frame_statistics"
    assert review["method"] == "frame_statistics_only-v1"
    assert review["frame_metrics"]["frame_count"] == 3
    assert review["artifact_health"]["readable"] is True
    assert review["score"] is None
    assert review["visible_evidence"]


def test_three_arm_skill_records_use_train_video_evidence_and_one_owner(tmp_path):
    module = _load("build_three_arm_skill_records", ROOT / "scripts" / "build_three_arm_skill_records.py")
    row = {
        "arm": "trained",
        "case_id": "case-1",
        "split": "train",
        "proxy_video": "C:/videos/case-1.mp4",
        "review_scores": {
            "camera_coverage": 40,
            "camera_innovation": 40,
            "visual_clarity": 40,
            "object_trajectory": 35,
            "character_trajectory": 35,
            "event_timing": 35,
            "temporal_smoothness": 35,
            "appearance_detail": 20,
        },
        "run_dir": "C:/runs/case-1",
    }
    records = module.build_records([row, {**row, "case_id": "case-2", "proxy_video": "C:/videos/case-2.mp4"}])
    assert records
    assert all(item["split"] == "train" for item in records)
    assert all(item["source"] == "actual_blender_video_local_review" for item in records)
    assert all(len({finding["owner"] for finding in item["findings"]}) == 1 for item in records)
    assert all(any(str(evidence).startswith("proxy_video:") for evidence in finding["evidence"]) for item in records for finding in item["findings"])
