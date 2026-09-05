"""Finding severity policy and root-cause deduplication."""

from __future__ import annotations

import re

from videoact.contracts import Finding


SEVERITY_PENALTIES = {"hard": 30.0, "error": 18.0, "warning": 8.0, "info": 2.0}
SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "hard": 3}


def normalize_root_cause_id(
    root_cause_id: str | None = None,
    *,
    failure_id: str | None = None,
    category: str | None = None,
    message: str | None = None,
) -> str:
    """Map equivalent natural-language failures to one stable root-cause ID.

    The normalizer deliberately uses category/keywords only as a fallback;
    it never converts an aggregate score into a root cause.  Explicit IDs are
    retained for categories that have no known semantic family.
    """

    text = " ".join(
        str(value or "")
        for value in (root_cause_id, failure_id, category, message)
    ).casefold()
    category_text = str(category or "").casefold()
    if "disagree" in text or "judge_disagreement" in text:
        return "judge_disagreement"
    if (
        "camera" in category_text
        or "visibility" in category_text
        or any(token in text for token in ("occlud", "not visible", "visibility", "out of frame", "cropped"))
    ):
        if any(token in text for token in ("motion", "orbit", "dolly", "follow", "pan", "tilt")) and not any(
            token in text for token in ("visible", "visibility", "occlud", "coverage")
        ):
            return "camera_motion"
        return "camera_visibility"
    if any(token in text for token in ("ownership", "final owner", "owner mismatch", "transfer window", "handoff")):
        return "ownership_transition"
    if "telemetry" in category_text or "observer" in category_text or "runtime_observation" in category_text:
        return "runtime_telemetry"
    if any(token in text for token in ("artifact", "mp4", "blend", "completeness", "missing_artifact")):
        return "artifact_completeness"
    if any(token in text for token in ("provenance", "fingerprint", "hash mismatch", "hash_mismatch")):
        return "provenance_integrity"
    if any(token in text for token in ("latency", "timeout", "timed out", "budget exceeded")):
        return "runtime_latency"
    if any(token in text for token in ("telemetry", "observer", "runtime observation")):
        return "runtime_telemetry"
    if any(token in text for token in ("trajectory", "motion primitive", "phase alignment")):
        return "trajectory_execution"
    if any(token in text for token in ("semantic", "event order", "required event", "prompt compliance")):
        return "semantic_event_completion"
    candidate = str(root_cause_id or failure_id or "unknown_failure").casefold()
    candidate = re.sub(r"[^a-z0-9]+", "_", candidate).strip("_")
    return candidate or "unknown_failure"


def finding_from_failure_evidence(evidence) -> dict[str, object]:
    """Convert normalized evidence to the legacy Finding-shaped payload.

    Abstentions are intentionally not converted by this helper.  Callers must
    filter on ``evidence.actionable`` before sending findings to proposal
    aggregation.
    """

    if not bool(getattr(evidence, "actionable", False)):
        raise ValueError("non-actionable failure evidence cannot become a patch finding")
    severity = str(getattr(evidence, "severity", "error"))
    if severity == "semantic_hard":
        severity = "hard"
    if severity not in SEVERITY_PENALTIES:
        severity = "error"
    return {
        "failure_id": str(getattr(evidence, "failure_id")),
        "owner": str(getattr(evidence, "owner_candidate") or "unassigned"),
        "category": str(getattr(evidence, "category")),
        "severity": severity,
        "message": str(getattr(evidence, "message")),
        "root_cause_id": str(getattr(evidence, "root_cause_id")),
        "evidence": list(getattr(evidence, "evidence_refs", []) or []),
        "repair_route": str(getattr(evidence, "repair_route", "candidate_recovery")),
    }


def deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    """Keep one penalty per root cause while merging its evidence."""
    grouped: dict[str, Finding] = {}
    order: list[str] = []
    for finding in findings:
        key = finding.root_cause_id or finding.failure_id
        if key not in grouped:
            grouped[key] = finding.model_copy(update={"root_cause_id": key})
            order.append(key)
            continue
        current = grouped[key]
        strongest = finding if SEVERITY_RANK[finding.severity] > SEVERITY_RANK[current.severity] else current
        merged_evidence = list(dict.fromkeys([*current.evidence, *finding.evidence]))
        grouped[key] = strongest.model_copy(update={"root_cause_id": key, "evidence": merged_evidence})
    return [grouped[key] for key in order]


def score_findings(findings: list[Finding]) -> float:
    unique = deduplicate_findings(findings)
    return max(0.0, 100.0 - sum(SEVERITY_PENALTIES[finding.severity] for finding in unique))
