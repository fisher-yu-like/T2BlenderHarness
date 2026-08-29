from __future__ import annotations


def _record(index: int) -> dict:
    value = 20 + index * 10
    dimensions = {
        "prompt_compliance": value,
        "physical_plausibility": value,
        "camera_coverage": value,
        "camera_innovation": value,
        "character_trajectory": value,
        "object_trajectory": value,
        "event_timing": value,
        "temporal_smoothness": value,
        "visual_clarity": value,
        "appearance_detail": value,
        "physical_realism": value,
        "spatial_consistency": value,
        "motion_naturalness": value,
        "visual_presentation": value,
    }
    return {
        "case_id": f"case-{index}",
        "human": {"dimensions": dimensions, "task_vlm": value, "realism_final": value},
        "vlm": {"dimensions": dimensions, "task_vlm": value, "realism_final": value},
        "frame_statistics": {
            "dimensions": {
                **{name: 50 for name in dimensions if name not in {"visual_clarity", "temporal_smoothness", "appearance_detail", "spatial_consistency", "visual_presentation"}},
                "visual_clarity": value,
                "temporal_smoothness": value,
                "appearance_detail": value,
                "spatial_consistency": value,
                "visual_presentation": value,
            },
            "task_vlm": 50,
            "realism_final": 50,
        },
        "deterministic": {"dimensions": {name: value for name in dimensions}, "task_vlm": value, "realism_final": value},
    }


def test_calibration_reports_dimension_correlations_and_gates():
    from scripts.calibrate_evaluator import calibrate_records

    report = calibrate_records([_record(index) for index in range(6)], seed=20260827, bootstrap_iterations=100)

    assert report["evaluator_version"] == "evaluator-v5-calibration"
    assert report["dimensions"]["prompt_compliance"]["vlm"]["spearman"] == 1.0
    assert report["dimensions"]["prompt_compliance"]["frame_statistics"]["status"] == "uninformative"
    assert report["overall_gates"]["vlm_task_vlm_spearman"] is True
    assert report["overall_gates"]["vlm_realism_spearman"] is True
    assert report["overall_gates"]["frame_statistics_semantics_uninformative"] is True
