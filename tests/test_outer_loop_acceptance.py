def test_candidate_is_accepted_only_with_train_improvement_and_no_dev_regression():
    from videoact.outer_loop import evaluate_candidate

    decision = evaluate_candidate(
        {"train_score": 70.0, "dev_score": 68.0},
        {"train_score": 72.0, "dev_score": 68.0},
        {"hard_regression": False},
        {"hard_regression": False},
    )

    assert decision.accepted is True
    assert decision.rollback_required is False
    assert "train" in decision.reason


def test_candidate_is_rejected_when_dev_regresses_or_train_does_not_improve():
    from videoact.outer_loop import evaluate_candidate

    no_train_gain = evaluate_candidate(
        {"train_score": 70.0, "dev_score": 68.0},
        {"train_score": 70.0, "dev_score": 69.0},
        {"hard_regression": False},
        {"hard_regression": False},
    )
    dev_regression = evaluate_candidate(
        {"train_score": 70.0, "dev_score": 68.0},
        {"train_score": 72.0, "dev_score": 67.0},
        {"hard_regression": False},
        {"hard_regression": False},
    )

    assert no_train_gain.accepted is False
    assert dev_regression.accepted is False
    assert no_train_gain.rollback_required is True
    assert dev_regression.rollback_required is True


def test_candidate_is_rejected_on_hard_dev_regression_even_with_score_gain():
    from videoact.outer_loop import evaluate_candidate

    decision = evaluate_candidate(
        {"train_score": 70.0, "dev_score": 68.0},
        {"train_score": 75.0, "dev_score": 69.0},
        {"hard_regression": False},
        {"hard_regression": True},
    )

    assert decision.accepted is False
    assert decision.rollback_required is True
    assert "hard" in decision.reason


def test_candidate_uses_paired_ci_for_small_dev_noise_and_keeps_safety_gates():
    from videoact.outer_loop import evaluate_candidate
    from videoact.experiment_fingerprint import ExperimentFingerprint, REQUIRED_HASH_FIELDS

    safety = {
        "artifact_completion": 1.0,
        "execution_validity": 1.0,
        "required_event_failure_count": 0,
        "hard_failure_count": 0,
    }
    fingerprint_payload = {field: "a" * 64 for field in REQUIRED_HASH_FIELDS}
    fingerprint_payload.update({"harness_version": "harness-v1", "blender_version": "4.3", "rollout_seed": "none"})
    fingerprint = ExperimentFingerprint.model_validate(fingerprint_payload).with_digest().model_dump(mode="json")
    train = {
        "paired_case_deltas": [3.8, 4.0, 4.2, 3.9],
        "paired_statistics_required": True,
        "safety_before": safety,
        "safety_after": safety,
    }
    dev = {
        "paired_case_deltas": [-0.3, -0.2, -0.4, -0.1],
        "secondary_deltas": {"camera_effectiveness": [-0.3, -0.2, -0.4, -0.1]},
        "safety_before": safety,
        "safety_after": safety,
    }

    decision = evaluate_candidate(
        {"train_score": 70.0, "dev_score": 80.0, "experiment_fingerprint": fingerprint},
        {"train_score": 74.0, "dev_score": 79.7, "experiment_fingerprint": fingerprint},
        train,
        dev,
    )

    assert decision.accepted is True
    assert decision.checks["overall_dev_non_regression"] is True
    assert decision.paired_statistics["checks"]["dev_noninferiority"] is True


def test_formal_candidate_rejects_incompatible_experiment_fingerprints():
    from videoact.experiment_fingerprint import ExperimentFingerprint, REQUIRED_HASH_FIELDS
    from videoact.outer_loop import evaluate_candidate

    payload = {field: "a" * 64 for field in REQUIRED_HASH_FIELDS}
    payload.update({"harness_version": "harness-v1", "blender_version": "4.3", "rollout_seed": "none"})
    before_fingerprint = ExperimentFingerprint.model_validate(payload).with_digest().model_dump(mode="json")
    after_payload = {**payload, "dataset_fingerprint": "b" * 64}
    after_fingerprint = ExperimentFingerprint.model_validate(after_payload).with_digest().model_dump(mode="json")
    safety = {
        "artifact_completion": 1.0,
        "execution_validity": 1.0,
        "required_event_failure_count": 0,
        "hard_failure_count": 0,
    }
    decision = evaluate_candidate(
        {"train_score": 70.0, "dev_score": 80.0, "experiment_fingerprint": before_fingerprint},
        {"train_score": 74.0, "dev_score": 80.0, "experiment_fingerprint": after_fingerprint},
        {"paired_case_deltas": [4.0, 4.0], "paired_statistics_required": True, "safety_before": safety, "safety_after": safety},
        {"paired_case_deltas": [0.0, 0.0], "safety_before": safety, "safety_after": safety},
    )

    assert decision.accepted is False
    assert "experiment_fingerprint_compatible" in decision.failed_checks
