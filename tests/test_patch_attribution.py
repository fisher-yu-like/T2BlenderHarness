from __future__ import annotations


def test_attribute_confirms_edit_when_predicted_fixes_hold(tmp_path):
    from videoact.patch_attribution import attribute

    verdict = attribute(
        {
            "edit_id": "edit-camera-01",
            "affected_files": ["src/videoact/director_camera.py"],
            "predicted_fixes": ["case-a", "case-b"],
            "predicted_regressions": [],
        },
        {"case-a": 4.0, "case-b": 1.0, "case-c": 0.0},
    )

    assert verdict.verdict == "confirmed"
    assert verdict.rollback_required is False
    assert verdict.fixed_case_ids == ["case-a", "case-b"]
    assert verdict.unpredicted_break_case_ids == []


def test_attribute_marks_unpredicted_regression_for_file_rollback():
    from videoact.patch_attribution import attribute

    verdict = attribute(
        {
            "edit_id": "edit-camera-02",
            "affected_files": ["src/videoact/director_camera.py"],
            "predicted_fixes": ["case-a", "case-b"],
            "predicted_regressions": ["case-c"],
        },
        {"case-a": 2.0, "case-b": 0.0, "case-c": 0.0, "case-d": -3.0},
    )

    assert verdict.verdict == "refuted"
    assert verdict.rollback_required is True
    assert verdict.unpredicted_break_case_ids == ["case-d"]
    assert verdict.rollback_files == ["src/videoact/director_camera.py"]

