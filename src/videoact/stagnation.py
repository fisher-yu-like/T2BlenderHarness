"""Deterministic early-stop detection for outer-loop stagnation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field


STAGNATION_SCHEMA_VERSION = "stagnation-report-v1"
STAGNATION_REASONS = {
    "no_actionable_failure",
    "attribution_uncertain",
    "patch_no_effect",
    "acceptance_noise_limited",
    "evaluator_insensitive",
    "architecture_ceiling",
    "data_coverage_insufficient",
}


class StagnationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = STAGNATION_SCHEMA_VERSION
    status: Literal["active", "stagnated"]
    attempts_without_accepted_patch: int = Field(ge=0)
    reason: str = Field(min_length=1)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    recommended_action: str = Field(min_length=1)
    formal_training_continues: bool
    new_high_confidence_root_cause: bool = False
    minimum_effect: float = 0.0
    observed_effect: float | None = None


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(child) for child in value]
    return value


def _forbidden_test_context(attempts: Sequence[Mapping[str, Any]]) -> None:
    forbidden_keys = {
        "test",
        "test_metrics",
        "test_score",
        "test_records",
        "test_case_ids",
        "frozen_test",
    }
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise ValueError("stagnation attempts must be objects")
        if forbidden_keys.intersection(str(key).casefold() for key in attempt):
            raise ValueError("test evidence cannot enter stagnation detection")


def _is_accepted(attempt: Mapping[str, Any]) -> bool:
    return str(attempt.get("action") or attempt.get("status") or "").casefold() in {
        "accepted",
        "accept",
        "patch_accepted",
    }


def _root(attempt: Mapping[str, Any]) -> str | None:
    proposal = attempt.get("proposal")
    if isinstance(proposal, Mapping):
        value = proposal.get("root_cause_id")
        if value:
            return str(value)
    value = attempt.get("root_cause_id") or attempt.get("failure_root_cause")
    return str(value) if value else None


def _confidence(attempt: Mapping[str, Any]) -> float:
    values = [attempt.get("owner_confidence"), attempt.get("attribution_confidence")]
    attribution = attempt.get("attribution")
    if isinstance(attribution, Mapping):
        values.append(attribution.get("owner_confidence", attribution.get("confidence")))
    proposal = attempt.get("proposal")
    if isinstance(proposal, Mapping):
        values.append(proposal.get("attribution_confidence"))
    for value in values:
        try:
            if value is not None:
                return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    return 0.0


def _effect(attempt: Mapping[str, Any]) -> float | None:
    for key in ("target_effect", "effect", "target_metric_delta", "minimum_effect_observed"):
        value = attempt.get(key)
        if value is None and isinstance(attempt.get("impact_proof"), Mapping):
            value = attempt["impact_proof"].get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def _reason(attempts: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    text = " ".join(
        str(attempt.get("reason") or attempt.get("status") or "").casefold()
        for attempt in attempts[-2:]
    )
    if not any(attempt.get("proposal") or attempt.get("findings") for attempt in attempts[-2:]):
        return "no_actionable_failure", "collect more independent train evidence before another patch attempt"
    if "attribution" in text or "owner" in text or "uncertain" in text:
        return "attribution_uncertain", "run a bounded counterfactual and abstain until one owner is isolated"
    if "no_effect" in text or "no effect" in text or "impact" in text:
        return "patch_no_effect", "inspect production call path, cache invalidation, and Patch Impact Proof"
    if "noise" in text or "noninferior" in text or "confidence interval" in text:
        return "acceptance_noise_limited", "increase paired evidence under the frozen statistical policy"
    if "evaluator" in text or "insensitive" in text:
        return "evaluator_insensitive", "audit evaluator sensitivity without changing it in this experiment"
    if "coverage" in text or "data" in text:
        return "data_coverage_insufficient", "add pre-registered train coverage; do not use dev/test for selection"
    return "architecture_ceiling", "inspect the owner contract and collect a new independent failure family"


def detect_stagnation(
    attempts: Sequence[Mapping[str, Any]],
    *,
    minimum_effect: float = 1.0,
    confidence_threshold: float = 0.6,
    consecutive_limit: int = 2,
) -> StagnationReport:
    """Return a deterministic active/stagnated decision from outer history."""

    if not isinstance(consecutive_limit, int) or consecutive_limit < 2:
        raise ValueError("consecutive_limit must be at least two")
    if minimum_effect < 0:
        raise ValueError("minimum_effect must be non-negative")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    rows = [_dump(item) for item in attempts]
    _forbidden_test_context(rows)
    accepted_index = max((index for index, item in enumerate(rows) if _is_accepted(item)), default=-1)
    tail = rows[accepted_index + 1 :]
    without_acceptance = len(tail)
    roots_before = {_root(item) for item in rows[:accepted_index + 1] if _root(item)}
    # Only the latest failed attempt can reset the consecutive-stagnation
    # counter.  Treating the first occurrence in the tail as “new progress”
    # would let two identical no-effect proposals run forever.
    latest = tail[-1] if tail else {}
    prior_roots = roots_before | {_root(item) for item in tail[:-1] if _root(item)}
    new_high = bool(_root(latest) and _root(latest) not in prior_roots and _confidence(latest) >= confidence_threshold)
    observed_effects = [value for value in (_effect(item) for item in tail) if value is not None]
    observed_effect = min(observed_effects) if observed_effects else None
    effect_low = observed_effect is not None and observed_effect < minimum_effect
    should_stop = without_acceptance >= consecutive_limit and not new_high or effect_low
    if not should_stop:
        return StagnationReport(
            status="active",
            attempts_without_accepted_patch=without_acceptance,
            reason="outer progress remains eligible for another explicitly budgeted attempt",
            evidence=[dict(item) for item in tail[-consecutive_limit:]],
            recommended_action="continue within the explicit outer-attempt budget",
            formal_training_continues=True,
            new_high_confidence_root_cause=new_high,
            minimum_effect=float(minimum_effect),
            observed_effect=observed_effect,
        )
    reason, recommendation = _reason(tail)
    if effect_low:
        reason = "patch_no_effect"
        recommendation = "stop and inspect the missing downstream effect before another attempt"
    return StagnationReport(
        status="stagnated",
        attempts_without_accepted_patch=without_acceptance,
        reason=reason,
        evidence=[dict(item) for item in tail[-consecutive_limit:]],
        recommended_action=recommendation,
        formal_training_continues=False,
        new_high_confidence_root_cause=new_high,
        minimum_effect=float(minimum_effect),
        observed_effect=observed_effect,
    )


def write_stagnation_report(path: str | Path, report: StagnationReport) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "STAGNATION_REASONS",
    "STAGNATION_SCHEMA_VERSION",
    "StagnationReport",
    "detect_stagnation",
    "write_stagnation_report",
]
