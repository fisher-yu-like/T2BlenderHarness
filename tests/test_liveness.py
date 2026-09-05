from __future__ import annotations

import pytest


def test_default_liveness_suite_discovers_and_attributes_all_required_faults() -> None:
    from videoact.liveness import default_fault_injections, run_liveness_suite

    report = run_liveness_suite(default_fault_injections())

    assert report.status == "pass"
    assert report.fault_count == 7
    assert report.detected_count == 7
    assert report.attributed_count == 7
    assert report.owner_accuracy == 1.0
    assert report.training_allowed is True


def test_liveness_blocks_training_when_two_faults_are_not_detected() -> None:
    from videoact.liveness import default_fault_injections, run_liveness_suite

    faults = default_fault_injections()
    faults[0] = faults[0].model_copy(update={"detected": False})
    faults[1] = faults[1].model_copy(update={"detected": False})
    report = run_liveness_suite(faults)

    assert report.status == "failed"
    assert report.training_allowed is False
    assert report.detected_count == 5
    assert any("detection_recall" in failure for failure in report.failures)


def test_wrong_owner_proposal_is_rejected() -> None:
    from videoact.liveness import validate_liveness_proposal

    with pytest.raises(ValueError, match="owner mismatch"):
        validate_liveness_proposal(
            expected_owner="director_camera",
            proposal={"owner": "director_trajectory", "source_split": "train"},
        )


def test_liveness_rejects_non_train_faults_and_failed_repairs() -> None:
    from videoact.liveness import default_fault_injections, run_liveness_suite

    faults = default_fault_injections()
    with pytest.raises(ValueError, match="train-only"):
        run_liveness_suite([faults[0].model_copy(update={"split": "dev"})])

    faults[0] = faults[0].model_copy(update={"repair_succeeds": False})
    report = run_liveness_suite(faults)
    assert report.training_allowed is False
    assert any("repair" in failure for failure in report.failures)
