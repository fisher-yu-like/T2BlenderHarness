"""Cross-round, case-level attribution for falsifiable Harness edits."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PatchVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edit_id: str
    verdict: str
    rollback_required: bool
    predicted_fix_case_ids: list[str] = Field(default_factory=list)
    fixed_case_ids: list[str] = Field(default_factory=list)
    missing_fix_case_ids: list[str] = Field(default_factory=list)
    predicted_regression_case_ids: list[str] = Field(default_factory=list)
    observed_break_case_ids: list[str] = Field(default_factory=list)
    unpredicted_break_case_ids: list[str] = Field(default_factory=list)
    rollback_files: list[str] = Field(default_factory=list)
    rationale: str


def attribute(manifest_entry: dict[str, Any], observed_deltas: dict[str, float]) -> PatchVerdict:
    """Compare predicted case deltas to measured paired deltas.

    Positive deltas count as fixed, negative deltas as breaks, and zero as
    inconclusive.  Any unpredicted break is a file-granularity rollback signal.
    """
    edit_id = str(manifest_entry.get("edit_id") or manifest_entry.get("root_cause_id") or "unknown-edit")
    predicted_fixes = sorted({str(case_id) for case_id in manifest_entry.get("predicted_fixes", [])})
    predicted_regressions = sorted({str(case_id) for case_id in manifest_entry.get("predicted_regressions", [])})
    fixed = sorted(case_id for case_id in predicted_fixes if float(observed_deltas.get(case_id, 0.0)) > 0)
    missing = sorted(set(predicted_fixes) - set(fixed))
    observed_breaks = sorted(case_id for case_id, delta in observed_deltas.items() if float(delta) < 0)
    predicted_breaks = sorted(set(observed_breaks) & set(predicted_regressions))
    unpredicted = sorted(set(observed_breaks) - set(predicted_regressions))
    if unpredicted:
        verdict = "refuted"
        rollback_required = True
        rationale = f"unpredicted regressions observed in {unpredicted}; rollback affected files"
    elif fixed == predicted_fixes and not observed_breaks:
        verdict = "confirmed"
        rollback_required = False
        rationale = "all predicted fixes improved and no observed regression was recorded"
    else:
        verdict = "partial"
        rollback_required = False
        rationale = "some predicted fixes were inconclusive or predicted regressions were observed"
    return PatchVerdict(
        edit_id=edit_id,
        verdict=verdict,
        rollback_required=rollback_required,
        predicted_fix_case_ids=predicted_fixes,
        fixed_case_ids=fixed,
        missing_fix_case_ids=missing,
        predicted_regression_case_ids=predicted_regressions,
        observed_break_case_ids=observed_breaks,
        unpredicted_break_case_ids=unpredicted,
        rollback_files=sorted({str(path) for path in manifest_entry.get("affected_files", [])}) if rollback_required else [],
        rationale=rationale,
    )

