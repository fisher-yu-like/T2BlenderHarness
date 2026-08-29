from __future__ import annotations


def test_bounded_outer_attempts_stops_when_no_patch_is_available():
    from scripts.train_real_harness import run_bounded_outer_attempts

    seen: list[int] = []

    def run_attempt(attempt_number: int) -> dict:
        seen.append(attempt_number)
        return {"attempt": attempt_number, "train": {"findings": []}}

    def transition(attempt_number: int, reports: list[dict]) -> dict:
        assert attempt_number == len(reports)
        return {"action": "stop", "status": "awaiting_harness_patch", "reason": "no patch callback"}

    result = run_bounded_outer_attempts(
        run_attempt=run_attempt,
        transition=transition,
        max_attempts=5,
    )

    assert seen == [1]
    assert result["status"] == "awaiting_harness_patch"
    assert result["attempt_count"] == 1
    assert result["reports"][0]["attempt"] == 1


def test_bounded_outer_attempts_never_exceeds_five_and_records_each_patch():
    from scripts.train_real_harness import run_bounded_outer_attempts

    seen: list[int] = []

    def run_attempt(attempt_number: int) -> dict:
        seen.append(attempt_number)
        return {"attempt": attempt_number}

    def transition(attempt_number: int, reports: list[dict]) -> dict:
        assert attempt_number == len(reports)
        if attempt_number < 5:
            return {
                "action": "patch",
                "status": "patch_applied",
                "proposal": {"owner": "camera_planner", "attempt": attempt_number},
            }
        return {"action": "stop", "status": "max_attempts_exhausted", "reason": "five attempts reached"}

    result = run_bounded_outer_attempts(
        run_attempt=run_attempt,
        transition=transition,
        max_attempts=5,
    )

    assert seen == [1, 2, 3, 4, 5]
    assert result["status"] == "max_attempts_exhausted"
    assert result["attempt_count"] == 5
    assert len(result["transitions"]) == 5
    assert [item["proposal"]["attempt"] for item in result["transitions"][:4]] == [1, 2, 3, 4]


def test_bounded_outer_attempts_rejects_invalid_transition_actions():
    import pytest

    from scripts.train_real_harness import run_bounded_outer_attempts

    with pytest.raises(ValueError, match="action"):
        run_bounded_outer_attempts(
            run_attempt=lambda _attempt: {},
            transition=lambda _attempt, _reports: {"action": "render_again"},
            max_attempts=5,
        )

