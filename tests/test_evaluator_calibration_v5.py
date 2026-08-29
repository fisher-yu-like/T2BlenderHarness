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


def test_calibration_reports_event_f1_mae_and_confidence_reliability() -> None:
    from scripts.calibrate_evaluator import calibrate_records

    records = []
    for index in range(6):
        record = _record(index)
        record["human"]["required_event_labels"] = {"event_01": True}
        record["vlm"]["required_event_labels"] = {"event_01": True}
        record["vlm"]["confidence"] = 0.9
        records.append(record)

    report = calibrate_records(records, seed=20260827, bootstrap_iterations=100)

    assert report["dimensions"]["prompt_compliance"]["vlm"]["mae"] == 0.0
    assert report["overall_gates"]["required_event_f1_value"] == 1.0
    assert report["overall_gates"]["required_event_f1"] is True
    assert report["confidence_reliability"]["status"] == "scored"
    assert report["confidence_reliability"]["buckets"]["0.8-1.0"]["mean_absolute_task_error"] == 0.0


def test_calibration_does_not_invent_required_event_labels() -> None:
    from scripts.calibrate_evaluator import calibrate_records

    report = calibrate_records([_record(index) for index in range(6)], seed=1, bootstrap_iterations=20)

    assert report["overall_gates"]["required_event_f1_status"] == "unavailable"
    assert report["overall_gates"]["required_event_f1"] is False
