"""Acceptance gate for versioned Harness candidates."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from .patch_attribution import PatchVerdict, attribute
from .paired_statistics import evaluate_paired_acceptance
from .experiment_fingerprint import compare_experiment_fingerprints
from .patch_impact import PatchImpactProof, validate_patch_impact


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
    paired_statistics: dict[str, Any] = {}
    experiment_fingerprint: dict[str, Any] | None = None
    patch_impact: dict[str, Any] | None = None


def _hard_regression(payload: dict[str, Any]) -> bool:
    if payload.get("hard_regression", False):
        return True
    before = payload.get("hard_failure_count_before")
    after = payload.get("hard_failure_count_after")
    return before is not None and after is not None and float(after) > float(before)


def _paired_train_ok(train: dict[str, Any], before: float, after: float) -> bool:
    if "paired_train_improvement" in train:
        return bool(train["paired_train_improvement"])
    if "paired_case_deltas" in train:
        # Case-level noise is handled by the bootstrap gate.  Retaining the
        # old all-positive rule here would defeat T10's purpose.
        return True
    return after > before


def _paired_dev_ok(dev: dict[str, Any], before: float, after: float) -> bool:
    if "paired_dev_non_regression" in dev:
        return bool(dev["paired_dev_non_regression"])
    if "paired_case_deltas" in dev:
        return True
    return after >= before


def _paired_deltas(payload: dict[str, Any], before: float, after: float) -> list[float]:
    values = payload.get("paired_case_deltas")
    if values is None:
        return [float(after) - float(before)]
    if not isinstance(values, (list, tuple)):
        raise ValueError("paired_case_deltas must be a list or tuple")
    return [float(value) for value in values]


def _safety_snapshots(
    before: dict[str, Any],
    after: dict[str, Any],
    train: dict[str, Any],
    dev: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Collect frozen safety metrics without inventing missing evidence."""

    before_values: dict[str, Any] = {}
    after_values: dict[str, Any] = {}
    for payload in (before, train, dev):
        nested = payload.get("safety_before")
        if isinstance(nested, dict):
            before_values.update(nested)
        nested = payload.get("safety_after")
        if isinstance(nested, dict):
            after_values.update(nested)
    for metric in (
        "artifact_completion",
        "execution_validity",
        "required_event_failure_count",
        "hard_failure_count",
    ):
        if metric in before and metric in after:
            before_values.setdefault(metric, before[metric])
            after_values.setdefault(metric, after[metric])
        for payload in (train, dev):
            if f"{metric}_before" in payload and f"{metric}_after" in payload:
                before_values.setdefault(metric, payload[f"{metric}_before"])
                after_values.setdefault(metric, payload[f"{metric}_after"])
    return before_values, after_values


def _contains_frame_statistics(payload: dict[str, Any]) -> bool:
    values: list[Any] = [payload.get("review_source"), payload.get("review_sources")]
    values.extend(payload.get("case_review_sources", []) or [])
    return any(
        value == "frame_statistics"
        or isinstance(value, (list, tuple, set)) and "frame_statistics" in value
        for value in values
    )


def _fingerprint_check(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    required: bool,
) -> tuple[bool, dict[str, Any] | None]:
    first = before.get("experiment_fingerprint")
    second = after.get("experiment_fingerprint")
    if first is None and second is None:
        return (not required), None
    if not isinstance(first, dict) or not isinstance(second, dict):
        return False, {"compatible": False, "mismatches": ["missing_experiment_fingerprint"]}
    try:
        report = compare_experiment_fingerprints(first, second)
    except (TypeError, ValueError) as exc:
        return False, {"compatible": False, "mismatches": [f"invalid_experiment_fingerprint:{type(exc).__name__}"]}
    return bool(report["compatible"]), report


def evaluate_candidate(
    before: dict[str, float],
    after: dict[str, float],
    train: dict[str, Any],
    dev: dict[str, Any],
    *,
    owner: str | None = None,
    impact_proof: Mapping[str, Any] | PatchImpactProof | None = None,
    require_impact_proof: bool = False,
) -> AcceptanceDecision:
    train_before = float(before["train_score"])
    train_after = float(after["train_score"])
    dev_before = float(before["dev_score"])
    dev_after = float(after["dev_score"])
    statistics_mode = any(
        "paired_case_deltas" in payload or payload.get("paired_statistics_required")
        for payload in (train, dev)
    )
    safety_before, safety_after = _safety_snapshots(before, after, train, dev)
    paired_statistics = evaluate_paired_acceptance(
        _paired_deltas(train, train_before, train_after),
        _paired_deltas(dev, dev_before, dev_after),
        secondary_deltas=train.get("secondary_deltas") or dev.get("secondary_deltas"),
        safety_before=safety_before,
        safety_after=safety_after,
        require_safety_metrics=bool(train.get("paired_statistics_required") or dev.get("paired_statistics_required")),
    )
    fingerprint_compatible, fingerprint_report = _fingerprint_check(
        before,
        after,
        required=bool(train.get("paired_statistics_required") or dev.get("paired_statistics_required")),
    )
    impact_required = bool(
        require_impact_proof
        or train.get("patch_impact_proof_required")
        or dev.get("patch_impact_proof_required")
    )
    impact_ok = True
    impact_report: dict[str, Any] | None = None
    if impact_proof is not None:
        try:
            impact_model = validate_patch_impact(impact_proof)
            impact_report = impact_model.model_dump(mode="json")
        except (TypeError, ValueError) as exc:
            impact_ok = False
            impact_report = {"status": "rejected", "reason": str(exc)}
    elif impact_required:
        impact_ok = False
        impact_report = {"status": "blocked", "reason": "complete Patch Impact Proof is required"}
    checks = {
        "overall_train_improved": train_after > train_before,
        "paired_train": _paired_train_ok(train, train_before, train_after)
        and paired_statistics["checks"]["train_min_gain"],
        "overall_dev_non_regression": (
            paired_statistics["checks"]["dev_noninferiority"] if statistics_mode else dev_after >= dev_before
        ),
        "paired_dev_non_regression": _paired_dev_ok(dev, dev_before, dev_after)
        and paired_statistics["checks"]["dev_noninferiority"],
        "paired_statistics": paired_statistics["accepted"],
        "experiment_fingerprint_compatible": fingerprint_compatible,
        "patch_impact_proof": impact_ok,
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
        paired_statistics=paired_statistics,
        experiment_fingerprint=fingerprint_report,
        patch_impact=impact_report,
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
