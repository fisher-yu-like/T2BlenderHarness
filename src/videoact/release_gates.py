"""Machine-readable G0--G4 release gates for formal Harness training.

The existing readiness matrix answers whether individual prerequisites are
present.  This module adds the immutable pilot/shadow release boundary from
the improvement plan: every gate is a sealed report, exact case counts are
checked, and a formal run cannot be unlocked by a hand-written boolean.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


RELEASE_GATES_VERSION = "release-gates-v1"
REQUIRED_RELEASE_GATE_IDS = ("G0", "G1", "G2", "G3")


def _canonical_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if str(key) != "report_hash"}


def report_hash(report: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_payload(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seal_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Add a content hash to an evidence report without mutating the input."""

    if not isinstance(report, Mapping):
        raise ValueError("release gate report must be an object")
    result = _canonical_payload(report)
    result["report_hash"] = report_hash(result)
    return result


def verify_sealed_report(report: Mapping[str, Any]) -> tuple[bool, str]:
    if not isinstance(report, Mapping):
        return False, "report_not_an_object"
    supplied = str(report.get("report_hash") or "").strip()
    if not supplied:
        return False, "report_hash_missing"
    if supplied != report_hash(report):
        return False, "report_hash_mismatch"
    return True, "sealed"


def _failure_report(gate_id: str, source: Mapping[str, Any] | None, failures: list[str], reason: str) -> dict[str, Any]:
    source_hash = str(source.get("report_hash") or "").strip() if isinstance(source, Mapping) else ""
    return {
        "version": RELEASE_GATES_VERSION,
        "gate_id": gate_id,
        "status": "pass" if not failures else "blocked",
        "failures": list(dict.fromkeys(failures)),
        "reason": "release_gate_passed" if not failures else reason,
        "report_hash": source_hash or None,
        # Keep the exact sealed source report so a later verifier can replay
        # the gate predicate.  A bare hash is not enough to audit whether the
        # derived status still matches the source evidence.
        "source_report": dict(source) if isinstance(source, Mapping) else None,
    }


def _base_failures(report: Mapping[str, Any] | None, gate_id: str) -> list[str]:
    if not isinstance(report, Mapping):
        return ["report_not_an_object"]
    failures: list[str] = []
    if str(report.get("gate_id") or "") != gate_id:
        failures.append("gate_id_mismatch")
    if report.get("status") != "pass":
        failures.append("status_not_pass")
    sealed, reason = verify_sealed_report(report)
    if not sealed:
        failures.append(reason)
    return failures


def _require_exact_count(report: Mapping[str, Any], expected_total: int, expected_split_counts: Mapping[str, int]) -> list[str]:
    failures: list[str] = []
    if report.get("case_count") != expected_total:
        failures.append(f"case_count_must_equal_{expected_total}")
    observed = report.get("split_case_counts")
    if not isinstance(observed, Mapping):
        failures.append("split_case_counts_missing")
    else:
        for split, expected in expected_split_counts.items():
            if observed.get(split) != expected:
                failures.append(f"{split}_case_count_must_equal_{expected}")
        if set(observed) != set(expected_split_counts):
            failures.append("unexpected_split_case_count_keys")
    return failures


def _require_true_fields(report: Mapping[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if report.get(field) is not True]


def validate_paired_pilot(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate G2's exact 10-train + 10-dev paired pilot evidence."""

    failures = _base_failures(report, "G2")
    if isinstance(report, Mapping):
        failures.extend(_require_exact_count(report, 20, {"train": 10, "dev": 10}))
        failures.extend(
            _require_true_fields(
                report,
                (
                    "all_artifacts_complete",
                    "trusted_observer_complete",
                    "blind_review_complete",
                    "disagreement_audit_complete",
                    "paired_outcome_registered",
                ),
            )
        )
        if report.get("baseline_arm") != "rule_template_baseline":
            failures.append("baseline_arm_must_be_rule_template_baseline")
        if report.get("candidate_arm") != "model_driven_candidate":
            failures.append("candidate_arm_must_be_model_driven_candidate")
        if not str(report.get("primary_outcome") or "").strip():
            failures.append("primary_outcome_missing")
        margin = report.get("noninferiority_margin")
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not math.isfinite(float(margin)):
            failures.append("noninferiority_margin_missing_or_invalid")
        if not str(report.get("hard_failure_rule") or "").strip():
            failures.append("hard_failure_rule_missing")
    return _failure_report("G2", report, failures, "paired_pilot_evidence_incomplete")


def validate_shadow_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate G3's no-patch 60-train + 60-dev shadow run."""

    failures = _base_failures(report, "G3")
    if isinstance(report, Mapping):
        failures.extend(_require_exact_count(report, 120, {"train": 60, "dev": 60}))
        if report.get("patch_applied") is not False:
            failures.append("patch_applied_must_be_false")
        failures.extend(
            _require_true_fields(
                report,
                (
                    "resume_verified",
                    "fingerprints_stable",
                    "memory_complete",
                    "cost_slo_pass",
                    "artifact_completion_slo_pass",
                    "hard_failure_slo_pass",
                    "judge_unavailable_slo_pass",
                ),
            )
        )
    return _failure_report("G3", report, failures, "shadow_evidence_incomplete")


def _validate_generic_gate(report: Mapping[str, Any] | None, gate_id: str) -> dict[str, Any]:
    failures = _base_failures(report, gate_id)
    return _failure_report(gate_id, report, failures, f"{gate_id}_evidence_incomplete")


def _revalidate_embedded_gate(derived: Mapping[str, Any], gate_id: str) -> list[str]:
    """Replay a gate from the source report embedded in a release report."""

    failures: list[str] = []
    source = derived.get("source_report")
    if not isinstance(source, Mapping):
        return [f"{gate_id}:source_report_missing"]
    expected = (
        validate_paired_pilot(source)
        if gate_id == "G2"
        else validate_shadow_report(source)
        if gate_id == "G3"
        else _validate_generic_gate(source, gate_id)
    )
    for field in ("status", "failures", "reason", "report_hash"):
        if derived.get(field) != expected.get(field):
            failures.append(f"{gate_id}:derived_{field}_mismatch")
    sealed, reason = verify_sealed_report(source)
    if not sealed:
        failures.append(f"{gate_id}:source_{reason}")
    if str(derived.get("report_hash") or "") != str(source.get("report_hash") or ""):
        failures.append(f"{gate_id}:source_hash_reference_mismatch")
    return failures


def build_formal_release_report(
    g0: Mapping[str, Any] | None,
    g1: Mapping[str, Any] | None,
    pilot: Mapping[str, Any] | None,
    shadow: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a sealed G0--G3 decision; only a complete pass unlocks training."""

    checks = {
        "G0": _validate_generic_gate(g0, "G0"),
        "G1": _validate_generic_gate(g1, "G1"),
        "G2": validate_paired_pilot(pilot),
        "G3": validate_shadow_report(shadow),
    }
    blocking = [gate_id for gate_id, result in checks.items() if result["status"] != "pass"]
    payload: dict[str, Any] = {
        "version": RELEASE_GATES_VERSION,
        "status": "pass" if not blocking else "blocked",
        "training_allowed": not blocking,
        "blocking_gates": blocking,
        "gate_reports": checks,
        "protocol": {
            "pilot_train_cases": 10,
            "pilot_dev_cases": 10,
            "shadow_train_cases": 60,
            "shadow_dev_cases": 60,
            "formal_round_count": 6,
            "max_attempts_per_round": 5,
        },
        "reason": "formal_release_gates_passed" if not blocking else "formal_release_blocked_by_gate_evidence",
    }
    return seal_report(payload)


def validate_formal_release_report(report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fail closed when a caller tries to use an altered release decision."""

    if not isinstance(report, Mapping):
        return {"status": "blocked", "training_allowed": False, "reason": "release_report_not_an_object"}
    sealed, reason = verify_sealed_report(report)
    missing = [gate_id for gate_id in REQUIRED_RELEASE_GATE_IDS if gate_id not in (report.get("gate_reports") or {})]
    if not sealed or report.get("status") != "pass" or report.get("training_allowed") is not True or missing:
        failures = []
        if not sealed:
            failures.append(reason)
        if report.get("status") != "pass":
            failures.append("status_not_pass")
        if report.get("training_allowed") is not True:
            failures.append("training_allowed_not_true")
        failures.extend(f"missing_gate_report:{gate_id}" for gate_id in missing)
        return {
            "version": RELEASE_GATES_VERSION,
            "status": "blocked",
            "training_allowed": False,
            "failures": failures,
            "reason": "formal_release_report_invalid",
        }
    nested_failures: list[str] = []
    for gate_id in REQUIRED_RELEASE_GATE_IDS:
        nested = report["gate_reports"].get(gate_id)
        if not isinstance(nested, Mapping):
            nested_failures.append(f"{gate_id}:gate_report_not_an_object")
            continue
        nested_failures.extend(_revalidate_embedded_gate(nested, gate_id))
        if nested.get("status") != "pass":
            nested_failures.append(f"gate_not_pass:{gate_id}")
    expected_blocking = [gate_id for gate_id in REQUIRED_RELEASE_GATE_IDS if (report["gate_reports"].get(gate_id) or {}).get("status") != "pass"]
    if report.get("blocking_gates") != expected_blocking:
        nested_failures.append("blocking_gates_mismatch")
    expected_status = "pass" if not expected_blocking else "blocked"
    if report.get("status") != expected_status:
        nested_failures.append("release_status_mismatch")
    if report.get("training_allowed") != (not expected_blocking):
        nested_failures.append("training_allowed_mismatch")
    if nested_failures:
        return {
            "version": RELEASE_GATES_VERSION,
            "status": "blocked",
            "training_allowed": False,
            "failures": sorted(set(nested_failures)),
            "reason": "formal_release_nested_gate_invalid",
        }
    return {
        "version": RELEASE_GATES_VERSION,
        "status": "pass",
        "training_allowed": True,
        "report_hash": str(report["report_hash"]),
        "gate_report_hashes": {
            gate_id: str(report["gate_reports"][gate_id].get("report_hash") or "")
            for gate_id in REQUIRED_RELEASE_GATE_IDS
        },
        "reason": "formal_release_report_verified",
    }


__all__ = [
    "RELEASE_GATES_VERSION",
    "REQUIRED_RELEASE_GATE_IDS",
    "build_formal_release_report",
    "report_hash",
    "seal_report",
    "validate_formal_release_report",
    "validate_paired_pilot",
    "validate_shadow_report",
    "verify_sealed_report",
]
