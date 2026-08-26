def test_calibration_reports_not_ready_without_independent_human_labels():
    from scripts.build_evaluator_calibration import build_calibration_report

    report = build_calibration_report(
        [
            {
                "case_id": "case-01",
                "pass_fail": "unreviewed",
                "primary_failure_owner": "unreviewed",
            }
        ],
        minimum_labeled_cases=3,
    )

    assert report["status"] == "not_ready"
    assert report["labeled_cases"] == 0


def test_calibration_computes_owner_accuracy_for_labeled_cases():
    from scripts.build_evaluator_calibration import build_calibration_report

    report = build_calibration_report(
        [
            {"case_id": "case-01", "pass_fail": "pass", "primary_failure_owner": "camera_planner", "predicted_owner": "camera_planner"},
            {"case_id": "case-02", "pass_fail": "fail", "primary_failure_owner": "trajectory_planner", "predicted_owner": "camera_planner"},
            {"case_id": "case-03", "pass_fail": "pass", "primary_failure_owner": "physics_validator", "predicted_owner": "physics_validator"},
        ],
        minimum_labeled_cases=3,
    )

    assert report["status"] == "ready"
    assert report["labeled_cases"] == 3
    assert report["owner_accuracy"] == 2 / 3
