"""Resumable, no-patch pilot/shadow execution for formal release gates.

The runner is intentionally callback-driven: the repository can supply the
real Blender/evaluator execution, while this module owns the safety boundary,
exact split counts, append-only checkpoints, and release-shaped evidence.  A
shadow callback receives the case needed to render it, but no controller,
proposal, test score, or patch hook is provided.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .release_gates import (
    seal_report,
    validate_paired_pilot,
    validate_shadow_report,
)


SHADOW_ROUND_SCHEMA_VERSION = "shadow-round-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _call(callback: Callable[..., Any], context: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(context)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if positional or any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return callback(context)
    return callback()


def _case_id(record: Mapping[str, Any]) -> str:
    value = str(record.get("case_id") or "").strip()
    if not value:
        raise ValueError("shadow case requires case_id")
    return value


def _split(record: Mapping[str, Any]) -> str:
    value = str(record.get("split") or "").strip().casefold()
    if value not in {"train", "dev"}:
        raise ValueError(f"shadow case split must be train or dev: {value}")
    return value


def _normalise_fingerprints(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}


def _progress_paths(output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / "shadow_round_progress.jsonl", root / "shadow_round_report.json"


def _load_progress(path: Path, *, experiment_fingerprint: str | None) -> tuple[dict[str, dict[str, Any]], bool, dict[str, int]]:
    latest: dict[str, dict[str, Any]] = {}
    attempts: dict[str, int] = {}
    valid = True
    if not path.is_file():
        return latest, valid, attempts
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"shadow progress is unreadable: {path}") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"shadow progress contains invalid JSON: {path}") from exc
        if not isinstance(row, Mapping) or row.get("event") != "shadow_case_completed":
            raise ValueError(f"shadow progress contains an invalid event: {path}")
        if experiment_fingerprint is not None and str(row.get("experiment_fingerprint") or "") != experiment_fingerprint:
            raise ValueError("shadow resume fingerprint does not match the current experiment")
        case_id = str(row.get("case_id") or "").strip()
        case_result = row.get("case_result")
        if not case_id or not isinstance(case_result, Mapping):
            raise ValueError("shadow progress event requires case_id and case_result")
        attempt = int(row.get("attempt") or 1)
        attempts[case_id] = max(attempts.get(case_id, 0), attempt)
        # A passed case is immutable and is the only case eligible for resume
        # skipping.  Failed cases remain retryable, but their evidence stays in
        # the append-only log.
        if str(case_result.get("status") or "") == "pass":
            if case_id in latest:
                valid = False
            latest[case_id] = dict(case_result)
    return latest, valid, attempts


def _public_case_result(
    case_id: str,
    split: str,
    raw: Any,
    *,
    expected_fingerprints: Mapping[str, str],
    error: str | None = None,
) -> dict[str, Any]:
    row = dict(raw) if isinstance(raw, Mapping) else ({"status": "pass"} if raw is True else {})
    observed = _normalise_fingerprints(row.get("fingerprints"))
    artifact_complete = row.get("artifact_complete") is True or row.get("artifact_status") == "complete"
    artifact_slo = row.get("artifact_completion_slo_pass") is True
    memory_recorded = row.get("memory_recorded") is True
    cost_slo = row.get("cost_slo_pass") is True
    hard_failure_slo = row.get("hard_failure_slo_pass") is True
    judge_unavailable_slo = row.get("judge_unavailable_slo_pass") is True
    controller_blocked = row.get("controller_test_access_blocked") is True
    fingerprint_stable = bool(expected_fingerprints) and observed == dict(expected_fingerprints)
    failures: list[str] = []
    if str(row.get("status") or "").casefold() not in {"pass", "passed", "success"}:
        failures.append("runner_status_not_pass")
    if error:
        failures.append("runner_exception")
    if row.get("patch_applied") is True:
        failures.append("patch_applied_in_shadow")
    if not artifact_complete:
        failures.append("artifact_incomplete")
    if not artifact_slo:
        failures.append("artifact_completion_slo_failed_or_missing")
    if not memory_recorded:
        failures.append("memory_evidence_missing")
    if not cost_slo:
        failures.append("cost_slo_failed_or_missing")
    if not hard_failure_slo:
        failures.append("hard_failure_slo_failed_or_missing")
    if not judge_unavailable_slo:
        failures.append("judge_unavailable_slo_failed_or_missing")
    if not controller_blocked:
        failures.append("controller_test_access_not_blocked")
    if not fingerprint_stable:
        failures.append("component_fingerprint_drift_or_missing")
    evidence_refs = row.get("evidence_refs")
    if isinstance(evidence_refs, str):
        evidence_refs = [evidence_refs]
    if not isinstance(evidence_refs, (list, tuple)):
        evidence_refs = []
    return {
        "case_id": case_id,
        "split": split,
        "status": "pass" if not failures else "blocked",
        "artifact_complete": artifact_complete,
        "artifact_completion_slo_pass": artifact_slo,
        "memory_recorded": memory_recorded,
        "cost_slo_pass": cost_slo,
        "hard_failure_slo_pass": hard_failure_slo,
        "judge_unavailable_slo_pass": judge_unavailable_slo,
        "controller_test_access_blocked": controller_blocked,
        "fingerprint_stable": fingerprint_stable,
        "evidence_refs": [str(item) for item in evidence_refs if str(item).strip()],
        "failures": list(dict.fromkeys(failures)),
        "error": error,
    }


def _write_report(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def run_shadow_round(
    records: Sequence[Mapping[str, Any]],
    *,
    runner: Callable[..., Any],
    output_dir: str | Path | None = None,
    experiment_fingerprint: str | None = None,
    component_fingerprints: Mapping[str, Any] | None = None,
    expected_train: int = 60,
    expected_dev: int = 60,
) -> dict[str, Any]:
    """Run exactly one no-patch 60-train + 60-dev shadow round.

    A failed case is retried on a later resumed invocation; a passed case is
    never executed again.  The progress log contains only case identity and
    evidence summaries, not prompt text or test metrics.
    """

    if not callable(runner):
        raise ValueError("shadow round requires a runner callback")
    if expected_train < 0 or expected_dev < 0:
        raise ValueError("shadow expected counts must be non-negative")
    cases = [dict(record) for record in records]
    ids = [_case_id(record) for record in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("shadow cases must have unique case IDs")
    splits = [_split(record) for record in cases]
    split_counts = {"train": splits.count("train"), "dev": splits.count("dev")}
    expected_counts = {"train": expected_train, "dev": expected_dev}
    count_failures = [
        f"{split}_case_count_must_equal_{expected}"
        for split, expected in expected_counts.items()
        if split_counts[split] != expected
    ]
    expected_fingerprints = _normalise_fingerprints(component_fingerprints)
    progress_path: Path | None = None
    report_path: Path | None = None
    completed: dict[str, dict[str, Any]] = {}
    resume_verified = True
    attempts: dict[str, int] = {}
    if output_dir is not None:
        progress_path, report_path = _progress_paths(output_dir)
        completed, resume_verified, attempts = _load_progress(
            progress_path, experiment_fingerprint=experiment_fingerprint
        )
        unknown = sorted(set(completed) - set(ids))
        if unknown:
            raise ValueError(f"shadow resume contains cases absent from current batch: {unknown}")

    results: dict[str, dict[str, Any]] = dict(completed)
    run_count = 0
    for record, split in zip(cases, splits):
        case_id = _case_id(record)
        if case_id in completed:
            continue
        run_count += 1
        context = {
            "case_id": case_id,
            "split": split,
            "record": dict(record),
            "shadow_round": True,
            "patch_allowed": False,
            "controller_access": "none",
            "test_metrics_available": False,
            "component_fingerprints": dict(expected_fingerprints),
        }
        error: str | None = None
        raw: Any = None
        try:
            raw = _call(runner, context)
        except Exception as exc:  # Keep the round auditable and continue the fixed batch.
            error = f"{type(exc).__name__}: {exc}"
        result = _public_case_result(
            case_id,
            split,
            raw,
            expected_fingerprints=expected_fingerprints,
            error=error,
        )
        results[case_id] = result
        if progress_path is not None:
            attempt = attempts.get(case_id, 0) + 1
            attempts[case_id] = attempt
            event = {
                "schema_version": SHADOW_ROUND_SCHEMA_VERSION,
                "event": "shadow_case_completed",
                "append_only": True,
                "experiment_fingerprint": experiment_fingerprint,
                "attempt": attempt,
                "case_id": case_id,
                "split": split,
                "case_result": result,
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            if result["status"] == "pass":
                completed[case_id] = result

    ordered_results = [results[case_id] for case_id in ids if case_id in results]
    failures = list(count_failures)
    failures.extend(
        f"{item['case_id']}:{failure}"
        for item in ordered_results
        for failure in item.get("failures", [])
    )
    all_cases_present = len(ordered_results) == len(cases)
    aggregate = {
        "artifact_complete": all(item.get("artifact_complete") is True for item in ordered_results) and all_cases_present,
        "artifact_completion_slo_pass": all(item.get("artifact_completion_slo_pass") is True for item in ordered_results) and all_cases_present,
        "memory_complete": all(item.get("memory_recorded") is True for item in ordered_results) and all_cases_present,
        "cost_slo_pass": all(item.get("cost_slo_pass") is True for item in ordered_results) and all_cases_present,
        "hard_failure_slo_pass": all(item.get("hard_failure_slo_pass") is True for item in ordered_results) and all_cases_present,
        "judge_unavailable_slo_pass": all(item.get("judge_unavailable_slo_pass") is True for item in ordered_results) and all_cases_present,
        "controller_test_access_blocked": all(item.get("controller_test_access_blocked") is True for item in ordered_results) and all_cases_present,
        "fingerprints_stable": bool(expected_fingerprints) and all(item.get("fingerprint_stable") is True for item in ordered_results) and all_cases_present,
    }
    if not aggregate["fingerprints_stable"]:
        failures.append("fingerprints_not_stable")
    if not resume_verified:
        failures.append("duplicate_completed_case_in_resume_log")
    payload: dict[str, Any] = {
        "schema_version": SHADOW_ROUND_SCHEMA_VERSION,
        "gate_id": "G3",
        "status": "pass" if not failures else "blocked",
        "case_count": len(cases),
        "split_case_counts": split_counts,
        "patch_applied": False,
        "resume_verified": resume_verified,
        "resumed": bool(progress_path is not None and progress_path.is_file() and run_count < len(cases)),
        "new_case_count": run_count,
        "experiment_fingerprint": experiment_fingerprint,
        "component_fingerprint_hash": _hash(expected_fingerprints) if expected_fingerprints else None,
        **aggregate,
        "failures": list(dict.fromkeys(failures)),
        "case_results": ordered_results,
        "progress_path": str(progress_path.resolve()) if progress_path is not None else None,
    }
    report = seal_report(payload)
    validation = validate_shadow_report(report)
    report.update(
        {
            "status": validation["status"] if not failures else "blocked",
            "failures": list(dict.fromkeys([*payload["failures"], *validation["failures"]])),
            "reason": "shadow_round_passed" if not failures and validation["status"] == "pass" else "shadow_round_blocked",
        }
    )
    report = seal_report(report)
    if report_path is not None:
        _write_report(report_path, report)
    return report


def run_paired_pilot(
    records: Sequence[Mapping[str, Any]],
    *,
    runner: Callable[..., Any],
    output_dir: str | Path | None = None,
    primary_outcome: str,
    noninferiority_margin: float,
    hard_failure_rule: str,
) -> dict[str, Any]:
    """Run an exact 10-train + 10-dev paired pilot with no hidden scoring."""

    if not callable(runner):
        raise ValueError("paired pilot requires a runner callback")
    cases = [dict(record) for record in records]
    ids = [_case_id(record) for record in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("paired pilot cases must have unique case IDs")
    splits = [_split(record) for record in cases]
    counts = {"train": splits.count("train"), "dev": splits.count("dev")}
    failures = []
    if counts != {"train": 10, "dev": 10}:
        failures.append("paired_pilot_requires_exact_10_train_10_dev")
    outcomes: list[dict[str, Any]] = []
    for record, split in zip(cases, splits):
        context = {
            "case_id": _case_id(record),
            "split": split,
            "record": dict(record),
            "paired_pilot": True,
            "baseline_arm": "rule_template_baseline",
            "candidate_arm": "model_driven_candidate",
            "patch_allowed": False,
            "controller_access": "none",
        }
        error: str | None = None
        raw: Any = None
        try:
            raw = _call(runner, context)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = dict(raw) if isinstance(raw, Mapping) else ({"status": "pass"} if raw is True else {})
        required = {
            "all_artifacts_complete": row.get("all_artifacts_complete") is True,
            "trusted_observer_complete": row.get("trusted_observer_complete") is True,
            "blind_review_complete": row.get("blind_review_complete") is True,
            "disagreement_audit_complete": row.get("disagreement_audit_complete") is True,
            "paired_outcome_registered": row.get("paired_outcome_registered") is True,
        }
        row_failures = [key for key, value in required.items() if not value]
        if str(row.get("status") or "").casefold() not in {"pass", "passed", "success"}:
            row_failures.append("runner_status_not_pass")
        if error:
            row_failures.append("runner_exception")
        outcomes.append({
            "case_id": _case_id(record),
            "split": split,
            "status": "pass" if not row_failures else "blocked",
            **required,
            "failures": row_failures,
            "error": error,
        })
        failures.extend(f"{_case_id(record)}:{item}" for item in row_failures)
    payload: dict[str, Any] = {
        "schema_version": SHADOW_ROUND_SCHEMA_VERSION,
        "gate_id": "G2",
        "status": "pass" if not failures else "blocked",
        "case_count": len(cases),
        "split_case_counts": counts,
        "all_artifacts_complete": not failures and all(item["all_artifacts_complete"] for item in outcomes),
        "trusted_observer_complete": not failures and all(item["trusted_observer_complete"] for item in outcomes),
        "blind_review_complete": not failures and all(item["blind_review_complete"] for item in outcomes),
        "disagreement_audit_complete": not failures and all(item["disagreement_audit_complete"] for item in outcomes),
        "paired_outcome_registered": not failures and all(item["paired_outcome_registered"] for item in outcomes),
        "baseline_arm": "rule_template_baseline",
        "candidate_arm": "model_driven_candidate",
        "primary_outcome": str(primary_outcome),
        "noninferiority_margin": float(noninferiority_margin),
        "hard_failure_rule": str(hard_failure_rule),
        "case_results": outcomes,
        "failures": list(dict.fromkeys(failures)),
    }
    report = seal_report(payload)
    validation = validate_paired_pilot(report)
    report.update({
        "status": validation["status"] if not failures else "blocked",
        "failures": list(dict.fromkeys([*payload["failures"], *validation["failures"]])),
        "reason": "paired_pilot_passed" if not failures and validation["status"] == "pass" else "paired_pilot_blocked",
    })
    report = seal_report(report)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "paired_pilot_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


__all__ = [
    "SHADOW_ROUND_SCHEMA_VERSION",
    "run_paired_pilot",
    "run_shadow_round",
]
