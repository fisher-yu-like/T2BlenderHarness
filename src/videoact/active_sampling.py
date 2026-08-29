"""Budgeted failure sampling that is blind to frozen/test outcomes."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from .paired_statistics import bootstrap_mean_ci


ACTIVE_SAMPLING_VERSION = "active-sampling-v2-replay"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _features(record: Mapping[str, Any], root_counts: Counter[str], known_owners: set[str]) -> tuple[dict[str, float], list[str], list[str]]:
    findings = [item for item in record.get("findings", []) or [] if isinstance(item, Mapping)]
    roots = sorted({str(item.get("root_cause_id") or "unknown") for item in findings})
    owners = sorted({str(item.get("owner") or "unknown") for item in findings})
    confidence = max(0.0, min(1.0, _number(record.get("review_confidence", record.get("confidence")), 0.0)))
    disagreement = max(0.0, min(1.0, _number(record.get("judge_disagreement"), 0.0)))
    repeated = max((min(1.0, (root_counts[root] - 1) / 4.0) for root in roots), default=0.0)
    novel_owner = 1.0 if any(owner not in known_owners for owner in owners) else 0.0
    severity = max(
        ({"hard": 1.0, "error": 0.8, "warning": 0.5, "info": 0.2}.get(str(item.get("severity")), 0.0) for item in findings),
        default=0.0,
    )
    values = {
        "uncertainty": 1.0 - confidence,
        "judge_disagreement": disagreement,
        "repeated_root_cause": repeated,
        "novel_owner": novel_owner,
        "severity": severity,
    }
    return values, roots, owners


def sample_failure_cases(
    records: Iterable[Mapping[str, Any]],
    *,
    budget: int,
    known_owners: set[str] | None = None,
    seed: int = 20260829,
) -> dict[str, Any]:
    """Select high-information train/dev cases without reading prompt text."""

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    values = [record for record in records if isinstance(record, Mapping)]
    if any(str(record.get("split") or "") not in {"train", "dev"} for record in values):
        raise ValueError("active sampler cannot consume the test split or any non-train/dev record")
    root_counts = Counter(
        str(finding.get("root_cause_id") or "unknown")
        for record in values
        for finding in (record.get("findings", []) or [])
        if isinstance(finding, Mapping)
    )
    known = {str(value) for value in (known_owners or set())}
    candidates: list[dict[str, Any]] = []
    for record in values:
        case_id = str(record.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("active sampler requires a non-empty case_id")
        features, roots, owners = _features(record, root_counts, known)
        priority = (
            0.35 * features["uncertainty"]
            + 0.30 * features["judge_disagreement"]
            + 0.20 * features["repeated_root_cause"]
            + 0.10 * features["novel_owner"]
            + 0.05 * features["severity"]
        )
        reasons = []
        if features["uncertainty"] >= 0.4:
            reasons.append("high uncertainty")
        if features["judge_disagreement"] >= 0.4:
            reasons.append("judge disagreement")
        if features["repeated_root_cause"] > 0:
            reasons.append("repeated root cause")
        if features["novel_owner"]:
            reasons.append("new failure owner")
        if not reasons:
            reasons.append("coverage and severity baseline")
        tie = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()
        candidates.append(
            {
                "case_id": case_id,
                "split": str(record.get("split")),
                "priority": round(priority, 8),
                "features": features,
                "root_cause_ids": roots,
                "owners": owners,
                "selection_reason": "; ".join(reasons),
                "_tie": tie,
            }
        )
    if not candidates:
        return {
            "version": ACTIVE_SAMPLING_VERSION,
            "status": "empty",
            "budget": budget,
            "selected": [],
            "candidates": [],
        }
    total_priority = sum(item["priority"] for item in candidates)
    uniform_probability = 1.0 / len(candidates)
    for item in candidates:
        item["sampling_probability"] = round(
            item["priority"] / total_priority if total_priority > 0 else uniform_probability,
            8,
        )
        item.pop("_tie", None)
    candidates.sort(
        key=lambda item: (
            -float(item["priority"]),
            hashlib.sha256(f"{seed}:{item['case_id']}".encode("utf-8")).hexdigest(),
        )
    )
    selected = candidates[: min(budget, len(candidates))]
    return {
        "version": ACTIVE_SAMPLING_VERSION,
        "status": "pass",
        "budget": budget,
        "candidate_count": len(candidates),
        "selected_case_ids": [item["case_id"] for item in selected],
        "selected": selected,
        "candidates": candidates,
        "selection_policy": "uncertainty+disagreement+repeated-root-cause+new-owner+severity; prompt/test-blind",
        "seed": int(seed),
    }


def sequential_stopping_decision(
    deltas: Iterable[float],
    *,
    target_lower_bound: float,
    min_cases: int = 10,
    seed: int = 20260829,
    iterations: int = 1000,
) -> dict[str, Any]:
    values = [float(value) for value in deltas]
    if min_cases < 1:
        raise ValueError("min_cases must be positive")
    if len(values) < min_cases:
        return {"version": ACTIVE_SAMPLING_VERSION, "decision": "continue", "n": len(values), "reason": "minimum_cases_not_reached"}
    interval = bootstrap_mean_ci(values, seed=seed, iterations=iterations)
    if interval["ci_lower"] >= float(target_lower_bound):
        decision = "stop_success"
    elif interval["ci_upper"] < float(target_lower_bound):
        decision = "stop_failure"
    else:
        decision = "continue"
    return {
        "version": ACTIVE_SAMPLING_VERSION,
        "decision": decision,
        "n": len(values),
        "target_lower_bound": float(target_lower_bound),
        "interval": interval,
    }


def _finite_replay_number(value: Any, field: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"replay record requires numeric {field}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"replay record requires {field} to be {qualifier}")
    return number


def _replay_decision(
    deltas: list[float],
    *,
    target_lower_bound: float,
    min_cases: int,
    seed: int,
    iterations: int,
) -> str:
    return str(
        sequential_stopping_decision(
            deltas,
            target_lower_bound=target_lower_bound,
            min_cases=min_cases,
            seed=seed,
            iterations=iterations,
        )["decision"]
    )


def audit_sampling_replay(
    batches: Iterable[Mapping[str, Any]],
    *,
    budget: int,
    target_lower_bound: float,
    min_cases: int = 10,
    min_reduction: float = 0.30,
    min_agreement: float = 0.95,
    known_owners: set[str] | None = None,
    seed: int = 20260829,
    iterations: int = 1000,
) -> dict[str, Any]:
    """Audit active sampling against a sealed historical full-render replay.

    Each batch must contain train/dev-only records with a numeric
    ``paired_delta`` and optional positive ``render_cost``.  The full decision
    is recomputed from every delta; a recorded ``full_decision`` is accepted
    only when it agrees with that recomputation.  The selected decision is
    recomputed from exactly the cases returned by :func:`sample_failure_cases`.

    This function deliberately copies no prompt, score payload, or raw record
    into its report.  Test records are rejected before sampling, so a replay
    cannot accidentally tune the sampler from frozen outcomes.
    """

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if isinstance(min_cases, bool) or not isinstance(min_cases, int) or min_cases < 1:
        raise ValueError("min_cases must be a positive integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    target = _finite_replay_number(target_lower_bound, "target_lower_bound")
    reduction_target = _finite_replay_number(min_reduction, "min_reduction")
    agreement_target = _finite_replay_number(min_agreement, "min_agreement")
    if not 0.0 <= reduction_target <= 1.0:
        raise ValueError("min_reduction must be between 0 and 1")
    if not 0.0 <= agreement_target <= 1.0:
        raise ValueError("min_agreement must be between 0 and 1")

    raw_batches = list(batches)
    if not raw_batches:
        raise ValueError("historical replay requires at least one batch")

    batch_reports: list[dict[str, Any]] = []
    total_full_render_count = 0.0
    total_sampled_render_count = 0.0
    matched_decisions = 0
    replay_failures: list[str] = []
    for batch_index, raw_batch in enumerate(raw_batches):
        if not isinstance(raw_batch, Mapping):
            raise ValueError("historical replay batches must be mappings")
        batch_id = str(raw_batch.get("batch_id") or "").strip()
        if not batch_id:
            raise ValueError("historical replay requires a non-empty batch_id")
        records = raw_batch.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"historical replay batch {batch_id} requires records")
        if any(not isinstance(record, Mapping) for record in records):
            raise ValueError(f"historical replay batch {batch_id} contains a non-mapping record")
        # Validate the split before invoking the sampler.  This keeps frozen
        # test IDs and outcomes outside the active-sampling decision path.
        if any(str(record.get("split") or "") not in {"train", "dev"} for record in records):
            raise ValueError("active sampling replay cannot consume the test split or any non-train/dev record")
        case_ids = [str(record.get("case_id") or "").strip() for record in records]
        if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError(f"historical replay batch {batch_id} requires unique non-empty case_id values")
        deltas: list[float] = []
        render_costs: dict[str, float] = {}
        for record, case_id in zip(records, case_ids):
            if "paired_delta" not in record:
                raise ValueError(f"historical replay record {case_id} requires paired_delta")
            deltas.append(_finite_replay_number(record.get("paired_delta"), "paired_delta"))
            render_costs[case_id] = _finite_replay_number(record.get("render_cost", 1.0), "render_cost", positive=True)

        full_decision = _replay_decision(
            deltas,
            target_lower_bound=target,
            min_cases=min_cases,
            seed=seed + batch_index,
            iterations=iterations,
        )
        recorded_full_decision = raw_batch.get("full_decision")
        if recorded_full_decision is not None:
            recorded_full_decision = str(recorded_full_decision)
            if recorded_full_decision not in {"stop_success", "stop_failure", "continue"}:
                raise ValueError(f"historical replay batch {batch_id} has an invalid full_decision")
            if recorded_full_decision != full_decision:
                replay_failures.append(f"{batch_id}:recorded_full_decision_mismatch")

        sample_report = sample_failure_cases(
            records,
            budget=budget,
            known_owners=known_owners,
            seed=seed + batch_index,
        )
        selected_case_ids = list(sample_report.get("selected_case_ids", []))
        selected_deltas = [deltas[case_ids.index(case_id)] for case_id in selected_case_ids]
        sampled_decision = _replay_decision(
            selected_deltas,
            target_lower_bound=target,
            min_cases=min_cases,
            seed=seed + batch_index,
            iterations=iterations,
        )
        recorded_sampled_decision = raw_batch.get("sampled_decision")
        if recorded_sampled_decision is not None:
            recorded_sampled_decision = str(recorded_sampled_decision)
            if recorded_sampled_decision not in {"stop_success", "stop_failure", "continue"}:
                raise ValueError(f"historical replay batch {batch_id} has an invalid sampled_decision")
            if recorded_sampled_decision != sampled_decision:
                replay_failures.append(f"{batch_id}:recorded_sampled_decision_mismatch")

        full_render_count = sum(render_costs.values())
        sampled_render_count = sum(render_costs[case_id] for case_id in selected_case_ids)
        decision_match = sampled_decision == full_decision
        if decision_match:
            matched_decisions += 1
        else:
            replay_failures.append(f"{batch_id}:decision_disagreement")
        total_full_render_count += full_render_count
        total_sampled_render_count += sampled_render_count
        batch_reports.append(
            {
                "batch_id": batch_id,
                "record_count": len(records),
                "selected_case_ids": selected_case_ids,
                "selected_count": len(selected_case_ids),
                "full_render_count": full_render_count,
                "sampled_render_count": sampled_render_count,
                "full_decision": full_decision,
                "sampled_decision": sampled_decision,
                "decision_match": decision_match,
                "selection_reason_by_case": {
                    item["case_id"]: item["selection_reason"]
                    for item in sample_report.get("selected", [])
                },
            }
        )

    batch_count = len(batch_reports)
    render_reduction = 1.0 - (total_sampled_render_count / total_full_render_count)
    decision_agreement = matched_decisions / batch_count
    if render_reduction < reduction_target:
        replay_failures.append("render_reduction_below_target")
    if decision_agreement < agreement_target:
        replay_failures.append("decision_agreement_below_target")
    return {
        "version": ACTIVE_SAMPLING_VERSION,
        "status": "pass" if not replay_failures else "fail",
        "batch_count": batch_count,
        "budget": budget,
        "target_lower_bound": target,
        "min_cases": min_cases,
        "render_reduction": round(render_reduction, 8),
        "decision_agreement": round(decision_agreement, 8),
        "full_render_count": total_full_render_count,
        "sampled_render_count": total_sampled_render_count,
        "min_reduction": reduction_target,
        "min_agreement": agreement_target,
        "failures": sorted(set(replay_failures)),
        "batches": batch_reports,
        "selection_policy": "uncertainty+disagreement+repeated-root-cause+new-owner+severity; train/dev-only; content/test-outcome-blind",
        "seed": int(seed),
        "iterations": iterations,
    }


__all__ = [
    "ACTIVE_SAMPLING_VERSION",
    "audit_sampling_replay",
    "sample_failure_cases",
    "sequential_stopping_decision",
]
