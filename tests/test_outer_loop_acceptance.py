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
