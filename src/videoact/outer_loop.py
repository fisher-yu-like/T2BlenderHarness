"""Acceptance gate for versioned Harness candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict


class AcceptanceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    rollback_required: bool
    reason: str
    train_score_before: float
    train_score_after: float
    dev_score_before: float
    dev_score_after: float


def evaluate_candidate(
    before: dict[str, float],
    after: dict[str, float],
    train: dict[str, Any],
    dev: dict[str, Any],
) -> AcceptanceDecision:
    train_before = float(before["train_score"])
    train_after = float(after["train_score"])
    dev_before = float(before["dev_score"])
    dev_after = float(after["dev_score"])
    if dev.get("hard_regression", False):
        reason = "rejected: hard regression detected on dev"
        accepted = False
    elif train_after <= train_before:
        reason = "rejected: train score did not strictly improve"
        accepted = False
    elif dev_after < dev_before:
        reason = "rejected: dev score regressed"
        accepted = False
    else:
        reason = "accepted: train improved and dev did not regress"
        accepted = True
    return AcceptanceDecision(
        accepted=accepted,
        rollback_required=not accepted,
        reason=reason,
        train_score_before=train_before,
        train_score_after=train_after,
        dev_score_before=dev_before,
        dev_score_after=dev_after,
    )


def write_optimization_record(path: str | Path, record: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return destination
