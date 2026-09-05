from __future__ import annotations

import pytest


def _attempt(root: str = "camera_visibility", reason: str = "patch_no_effect", effect=None):
    row = {
        "action": "blocked",
        "status": "blocked",
        "reason": reason,
        "root_cause_id": root,
        "attribution_confidence": 0.9,
        "proposal": {"root_cause_id": root},
    }
    if effect is not None:
        row["target_effect"] = effect
    return row


def test_two_identical_no_effect_attempts_stagnate_without_test_context():
    from videoact.stagnation import detect_stagnation

    report = detect_stagnation([_attempt(), _attempt()])

    assert report.status == "stagnated"
    assert report.reason == "patch_no_effect"
    assert report.formal_training_continues is False
    assert len(report.evidence) == 2


def test_new_high_confidence_root_cause_keeps_outer_loop_active():
    from videoact.stagnation import detect_stagnation

    report = detect_stagnation([_attempt("camera_visibility"), _attempt("trajectory_execution")])

    assert report.status == "active"
    assert report.new_high_confidence_root_cause is True
    assert report.formal_training_continues is True


@pytest.mark.parametrize(
    "reason",
    [
        "attribution_uncertain",
        "acceptance_noise_limited",
        "evaluator_insensitive",
        "data_coverage_insufficient",
    ],
)
def test_stagnation_reason_categories_are_deterministic(reason):
    from videoact.stagnation import detect_stagnation

    report = detect_stagnation([_attempt(reason=reason), _attempt(reason=reason)])
    assert report.status == "stagnated"
    assert report.reason in {
        "attribution_uncertain",
        "acceptance_noise_limited",
        "evaluator_insensitive",
        "data_coverage_insufficient",
        "patch_no_effect",
    }


def test_low_effect_stops_after_one_observation_and_test_keys_are_forbidden():
    from videoact.stagnation import detect_stagnation

    report = detect_stagnation([_attempt(effect=0.1)], minimum_effect=1.0)
    assert report.status == "stagnated"
    assert report.reason == "patch_no_effect"
    with pytest.raises(ValueError, match="test evidence"):
        detect_stagnation([{"status": "blocked", "test_score": 1.0}])


def test_bounded_outer_runner_stops_before_a_third_no_effect_attempt():
    from scripts.train_real_harness import run_bounded_outer_attempts

    seen: list[int] = []

    def transition(attempt, _reports):
        return {
            "action": "patch",
            "status": "patch_no_effect",
            "reason": "patch_no_effect",
            "proposal": {
                "owner": "director_camera",
                "root_cause_id": "camera_visibility",
                "source_split": "train",
                "affected_files": ["src/videoact/director_camera.py"],
            },
        }

    result = run_bounded_outer_attempts(
        run_attempt=lambda attempt: seen.append(attempt) or {"attempt": attempt},
        transition=transition,
        max_attempts=5,
    )

    assert seen == [1, 2]
    assert result["status"] == "stagnated"
    assert result["stagnation"]["reason"] == "patch_no_effect"
