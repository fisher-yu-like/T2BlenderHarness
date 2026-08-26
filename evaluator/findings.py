"""Finding severity policy and root-cause deduplication."""

from __future__ import annotations

from videoact.contracts import Finding


SEVERITY_PENALTIES = {"hard": 30.0, "error": 18.0, "warning": 8.0, "info": 2.0}
SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2, "hard": 3}


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
