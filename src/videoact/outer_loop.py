"""Acceptance gate for versioned Harness candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .patch_attribution import PatchVerdict, attribute


class AcceptanceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    rollback_required: bool
    reason: str
    train_score_before: float
    train_score_after: float
    dev_score_before: float
    dev_score_after: float
    checks: dict[str, bool] = {}
    failed_checks: list[str] = []


def _hard_regression(payload: dict[str, Any]) -> bool:
    if payload.get("hard_regression", False):
        return True
    before = payload.get("hard_failure_count_before")
    after = payload.get("hard_failure_count_after")
    return before is not None and after is not None and float(after) > float(before)


def _paired_train_ok(train: dict[str, Any], before: float, after: float) -> bool:
    if "paired_train_improvement" in train:
        return bool(train["paired_train_improvement"])
    deltas = train.get("paired_case_deltas")
    if deltas is not None:
        return bool(deltas) and all(float(delta) > 0 for delta in deltas)
    return after > before


def _paired_dev_ok(dev: dict[str, Any], before: float, after: float) -> bool:
    if "paired_dev_non_regression" in dev:
        return bool(dev["paired_dev_non_regression"])
    deltas = dev.get("paired_case_deltas")
    if deltas is not None:
        return all(float(delta) >= 0 for delta in deltas)
    return after >= before


def _contains_frame_statistics(payload: dict[str, Any]) -> bool:
    values: list[Any] = [payload.get("review_source"), payload.get("review_sources")]
    values.extend(payload.get("case_review_sources", []) or [])
    return any(
        value == "frame_statistics"
        or isinstance(value, (list, tuple, set)) and "frame_statistics" in value
        for value in values
    )


def evaluate_candidate(
    before: dict[str, float],
    after: dict[str, float],
    train: dict[str, Any],
    dev: dict[str, Any],
    *,
    owner: str | None = None,
) -> AcceptanceDecision:
    train_before = float(before["train_score"])
    train_after = float(after["train_score"])
    dev_before = float(before["dev_score"])
    dev_after = float(after["dev_score"])
    checks = {
        "overall_train_improved": train_after > train_before,
        "paired_train": _paired_train_ok(train, train_before, train_after),
        "overall_dev_non_regression": dev_after >= dev_before,
        "paired_dev_non_regression": _paired_dev_ok(dev, dev_before, dev_after),
        "zero_hard_regression": not (_hard_regression(train) or _hard_regression(dev)),
    }
    artifact_before = before.get("artifact_completion", before.get("artifact_rate"))
    artifact_after = after.get("artifact_completion", after.get("artifact_rate"))
    checks["artifact_completion"] = (
        artifact_before is None
        or artifact_after is not None and float(artifact_after) >= float(artifact_before)
    )
    checks["independent_visual_review"] = not (
        _contains_frame_statistics(train) or _contains_frame_statistics(dev)
    )
    realism_patch = owner in {"proxy_renderer", "blender_code_agent"} or train.get("patch_category") == "realism" or dev.get("patch_category") == "realism"
    if realism_patch:
        checks["realism_improved"] = float(after.get("realism_score", -1)) > float(before.get("realism_score", -1))
        checks["task_non_regression"] = float(after.get("task_score", -1)) >= float(before.get("task_score", -1))
    failed_checks = [name for name, passed in checks.items() if not passed]
    accepted = not failed_checks
    reason = (
        "accepted: strict paired train/dev gates passed"
        if accepted
        else "rejected: " + ", ".join(failed_checks)
    )
    return AcceptanceDecision(
        accepted=accepted,
        rollback_required=not accepted,
        reason=reason,
        train_score_before=train_before,
        train_score_after=train_after,
        dev_score_before=dev_before,
        dev_score_after=dev_after,
        checks=checks,
        failed_checks=failed_checks,
    )


def write_optimization_record(path: str | Path, record: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return destination


def record_patch_attribution(
    path: str | Path,
    manifest_entry: dict[str, Any],
    observed_deltas: dict[str, float],
) -> PatchVerdict:
    """Attribute a patch before distillation and append the immutable verdict."""
    verdict = attribute(manifest_entry, observed_deltas)
    write_optimization_record(
        path,
        {
            "event": "patch_attribution",
            "ordering": "before_root_cause_distillation",
            "manifest_entry": manifest_entry,
            "observed_deltas": observed_deltas,
            "verdict": verdict.model_dump(mode="json"),
        },
    )
    return verdict
