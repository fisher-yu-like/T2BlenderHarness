from __future__ import annotations

import random


def _safety(**overrides):
    values = {
        "artifact_completion": 1.0,
        "execution_validity": 1.0,
        "required_event_failure_count": 0,
        "hard_failure_count": 0,
    }
    values.update(overrides)
    return values


def test_bootstrap_ci_is_reproducible() -> None:
    from videoact.paired_statistics import bootstrap_mean_ci

    first = bootstrap_mean_ci([1.0, 2.0, 4.0, 5.0], seed=17, iterations=400)
    second = bootstrap_mean_ci([1.0, 2.0, 4.0, 5.0], seed=17, iterations=400)

    assert first == second
    assert first["ci_lower"] <= first["mean"] <= first["ci_upper"]


def test_noisy_task_gain_and_small_camera_noise_can_pass() -> None:
    from videoact.paired_statistics import evaluate_paired_acceptance

    report = evaluate_paired_acceptance(
        [3.7, 4.0, 4.3, 4.1, 3.9],
        [0.2, -0.3, 0.1, -0.2, 0.0],
        secondary_deltas={"camera_effectiveness": [-0.3, -0.2, -0.4, -0.1, -0.3]},
        safety_before=_safety(),
        safety_after=_safety(),
        seed=20260829,
        iterations=800,
    )

    assert report["accepted"] is True
    assert report["checks"]["train_min_gain"] is True
    assert report["checks"]["dev_noninferiority"] is True
    assert report["effect_size"] is not None
    assert report["secondary"]["camera_effectiveness"]["allowed_noise"] is True


def test_task_gain_is_rejected_when_required_event_failures_increase() -> None:
    from videoact.paired_statistics import evaluate_paired_acceptance

    report = evaluate_paired_acceptance(
        [4.0, 4.0, 4.0],
        [0.0, 0.1, -0.1],
        safety_before=_safety(required_event_failure_count=1),
        safety_after=_safety(required_event_failure_count=2),
        seed=1,
        iterations=300,
    )

    assert report["accepted"] is False
    assert report["checks"]["required_event_failure_count_non_regression"] is False
    assert "required_event_failure_count_non_regression" in report["failed_checks"]


def test_missing_safety_evidence_is_not_silently_accepted() -> None:
    from videoact.paired_statistics import evaluate_paired_acceptance

    report = evaluate_paired_acceptance([2.0, 2.0], [0.0, 0.0], seed=1, iterations=100)

    assert report["accepted"] is False
    assert "safety_metrics_missing" in report["failed_checks"]


def test_zero_effect_false_acceptance_stays_below_five_percent() -> None:
    from videoact.paired_statistics import evaluate_paired_acceptance

    accepted = 0
    rng = random.Random(20260829)
    for _ in range(100):
        train = [rng.gauss(0.0, 0.4) for _ in range(10)]
        dev = [rng.gauss(0.0, 0.4) for _ in range(10)]
        report = evaluate_paired_acceptance(
            train,
            dev,
            safety_before=_safety(),
            safety_after=_safety(),
            seed=20260829,
            iterations=200,
        )
        accepted += int(report["accepted"])

    assert accepted / 100 <= 0.05
