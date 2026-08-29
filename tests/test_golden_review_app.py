from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen

import pytest


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


def _write_bundle(root: Path) -> Path:
    bundle = root / "golden"
    video_dir = bundle / "videos" / "case-001"
    video_dir.mkdir(parents=True)
    sampled_videos = {}
    sampled_frames = {}
    for label, payload in (("sample_a", b"A-video"), ("sample_b", b"B-video"), ("sample_c", b"C-video")):
        video = video_dir / f"{label}.mp4"
        video.write_bytes(payload)
        sampled_videos[label] = str(video.resolve())
        sampled_frames[label] = [str((bundle / "frames" / "case-001" / label / "frame_0001.png").resolve())]
    (bundle / "manifest.jsonl").write_text(
        json.dumps(
            {
                "case_id": "case-001",
                "blind_group": "case-001-blind",
                "prompt": "A person moves a red cup to the marked support.",
                "sampled_frames": sampled_frames,
                "sampled_videos": sampled_videos,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle / "human_scores.jsonl").write_text("", encoding="utf-8")
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "golden-review-fixture",
                "arms_hidden": True,
                "patch_selection_allowed": False,
                "comparison_only": True,
                "render_prompt_mismatch_count": 1,
            }
        ),
        encoding="utf-8",
    )
    return bundle


def _payload(sample_id: str = "sample_a") -> dict:
    return {
        "case_id": "case-001",
        "sample_id": sample_id,
        "annotator_id": "annotator-sy",
        "scores": {dimension: 60 for dimension in DIMENSIONS},
        "pass_fail": "borderline",
        "primary_failure_owner": "director_camera",
        "visible_evidence": ["the active object is visible in the sampled sequence"],
        "weaknesses": ["the handoff is hard to verify"],
        "confidence": 0.8,
    }


def test_store_publishes_only_blind_media_urls_and_sample_progress(tmp_path: Path):
    from scripts.golden_review_app import GoldenReviewStore

    store = GoldenReviewStore(_write_bundle(tmp_path))

    public = store.public_manifest()
    assert public[0]["samples"]["sample_a"]["video_url"] == "/media/case-001/sample_a.mp4"
    assert public[0]["prompt_zh"].startswith("中文提示词翻译：")
    assert public[0]["prompt_zh"] != public[0]["prompt"]
    assert "sampled_videos" not in public[0]
    assert store.public_progress("annotator-sy") == {}

    saved = store.save_score(_payload())
    assert saved["sample_id"] == "sample_a"
    assert store.public_progress("annotator-sy") == {"case-001/sample_a": {"saved": True}}

    updated = _payload()
    updated["scores"] = {dimension: 75 for dimension in DIMENSIONS}
    store.save_score(updated)
    rows = [
        json.loads(line)
        for line in (store.scores_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["scores"]["camera_coverage"] == 75


def test_store_publishes_safe_metadata_and_original_prompt_separately(tmp_path: Path):
    from scripts.golden_review_app import GoldenReviewStore

    bundle = _write_bundle(tmp_path)
    store = GoldenReviewStore(bundle)

    public = store.public_manifest()[0]
    metadata = store.public_metadata()

    assert public["prompt_en"] == public["prompt"]
    assert public["prompt_zh"].startswith("中文提示词翻译：")
    assert metadata["dataset_id"] == "golden-review-fixture"
    assert metadata["case_count"] == 1
    assert metadata["sample_count"] == 3
    assert metadata["required_annotators"] == 2
    assert metadata["comparison_only"] is True
    assert metadata["render_prompt_mismatch_count"] == 1
    assert "blind_manifest" not in metadata
    assert "mapping" not in metadata


def test_store_rejects_unknown_sample_and_out_of_range_score(tmp_path: Path):
    from scripts.golden_review_app import GoldenReviewStore

    store = GoldenReviewStore(_write_bundle(tmp_path))

    unknown = _payload("sample_z")
    with pytest.raises(ValueError, match="sample_id"):
        store.save_score(unknown)

    invalid = _payload()
    invalid["scores"]["visual_clarity"] = 101
    with pytest.raises(ValueError, match="0-100"):
        store.save_score(invalid)


def test_http_server_serves_blind_media_range_and_score_api(tmp_path: Path):
    from scripts.golden_review_app import create_server

    bundle = _write_bundle(tmp_path)
    server = create_server(bundle, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urlopen(base + "/api/manifest") as response:
            manifest = json.loads(response.read().decode("utf-8"))
        assert manifest[0]["samples"]["sample_b"]["video_url"] == "/media/case-001/sample_b.mp4"

        with urlopen(base + "/api/metadata") as response:
            metadata = json.loads(response.read().decode("utf-8"))
        assert metadata["required_annotators"] == 2

        request = Request(base + "/media/case-001/sample_a.mp4", headers={"Range": "bytes=2-4"})
        with urlopen(request) as response:
            assert response.status == 206
            assert response.read() == b"vid"

        body = json.dumps(_payload()).encode("utf-8")
        request = Request(base + "/api/score", data=body, method="POST", headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            saved = json.loads(response.read().decode("utf-8"))
        assert saved["sample_id"] == "sample_a"

        with urlopen(base + "/api/progress?annotator_id=annotator-sy") as response:
            progress = json.loads(response.read().decode("utf-8"))
        assert progress == {"case-001/sample_a": {"saved": True}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_review_ui_contains_video_labels_rubric_and_save_route():
    html = Path("scripts/golden_review_ui/index.html").read_text(encoding="utf-8")

    assert '<html lang="zh-CN">' in html
    assert "盲评视频评分" in html
    assert "评分标准" in html
    assert "提示词遵循" in html
    assert "镜头调度创新" in html
    assert "人物轨迹" in html
    assert "物体轨迹" in html
    assert "保存当前视频评分" in html
    assert "Prompt compliance" not in html
    assert "Physical plausibility" not in html
    assert "sample_a" in html and "sample_b" in html and "sample_c" in html
    assert "prompt_compliance" in html
    assert "appearance_detail" in html
    assert "/api/score" in html
    assert "prompt_zh" in html
    assert "中文提示词" in html
    assert "promptEnglish" in html
    assert "保存并下一个" in html
    assert "未保存" in html
    assert "/api/metadata" in html
    assert "comparison-only" in html
    assert "仅比较用途" in html
