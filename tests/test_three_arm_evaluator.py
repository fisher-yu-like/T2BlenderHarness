from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_runtime_report_is_artifact_only_and_has_no_director_score():
    module = _load("evaluate_three_arm_ablation", ROOT / "scripts" / "evaluate_three_arm_ablation.py")

    report = module.build_direct_runtime_report(
        {
            "artifact_status": "complete",
            "readable_frame_count": 8,
            "video_frame_count": 3,
            "video_fps": 12.0,
            "video_duration_s": 6.0,
            "hard_failures": [],
        },
        {"planning_mode": "direct_prompt_code", "objects": {"actor_a": {}, "prop_a": {}}},
    )

    assert report["terminal_status"] == "pass"
    assert report["score_kind"] == "runtime_artifact_only"
    assert report["director_plan_score"] is None
    assert report["director_findings"] == []
    assert report["interaction_findings"] == []


def test_blind_review_requires_all_fourteen_dimensions_and_does_not_impute_unavailable():
    module = _load("aggregate_three_arm_ablation", ROOT / "scripts" / "aggregate_three_arm_ablation.py")
    dimensions = list(module.VISUAL_DIMENSIONS)
    complete = {
        "review_version": module.BLIND_REVIEW_VERSION,
        "status": "complete",
        "review_source": "assistant_local_review",
        "reviewer": "codex-assistant",
        "sample_label": "sample_a",
        "sampled_frames": ["frame_1.png"],
        "scores": {name: 60 for name in dimensions},
        "visible_evidence": ["chronological sampled frames were inspected"],
        "weaknesses": ["proxy geometry remains coarse"],
        "confidence": 0.8,
    }

    validated = module.validate_blind_review(complete, expected_sample_label="sample_a", expected_frames=["frame_1.png"])
    assert set(validated["scores"]) == set(dimensions)
    assert module.score_blind_review(validated, deterministic_score=None, geometry_report={}, visual_report={})["task_vlm"] == 60.0

    unavailable = {**complete, "status": "unavailable", "scores": {name: None for name in dimensions}, "confidence": 0.0}
    unavailable["visible_evidence"] = ["external visual reviewer unavailable"]
    result = module.score_blind_review(
        module.validate_blind_review(unavailable, expected_sample_label="sample_a", expected_frames=["frame_1.png"]),
        deterministic_score=None,
        geometry_report={},
        visual_report={},
    )
    assert result["task_vlm"] is None
    assert result["realism_final"] is None


def test_visual_score_has_no_action_or_arm_bonus():
    module = _load("aggregate_three_arm_ablation", ROOT / "scripts" / "aggregate_three_arm_ablation.py")
    review = {
        "review_version": module.BLIND_REVIEW_VERSION,
        "status": "complete",
        "review_source": "assistant_local_review",
        "reviewer": "codex-assistant",
        "sample_label": "sample_a",
        "sampled_frames": ["frame_1.png"],
        "scores": {name: 55 for name in module.VISUAL_DIMENSIONS},
        "visible_evidence": ["same visible evidence"],
        "weaknesses": [],
        "confidence": 0.8,
    }
    validated = module.validate_blind_review(review, expected_sample_label="sample_a", expected_frames=["frame_1.png"])
    first = module.score_blind_review(validated, deterministic_score=80, geometry_report={}, visual_report={}, action_variant="direct_transfer", arm="pretrain")
    second = module.score_blind_review(validated, deterministic_score=80, geometry_report={}, visual_report={}, action_variant="reveal_elliptical_return", arm="trained")
    assert first["task_vlm"] == second["task_vlm"]
    assert first["realism_final"] == second["realism_final"]

