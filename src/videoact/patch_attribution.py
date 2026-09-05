"""Cross-round, case-level attribution for falsifiable Harness edits."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


PATCH_SCOPE_VERSION = "harness-patch-scope-v1"
_FROZEN_PATH_PARTS = {
    "dataset",
    "datasets",
    "evaluator",
    "observer",
    "observers",
    "test",
    "tests",
}


def normalize_patch_path(path: str) -> str:
    """Normalize one proposal path and reject workspace escapes."""

    value = str(path).replace("\\", "/").strip()
    if not value:
        raise ValueError("patch path cannot be empty")
    if value.startswith("/") or len(value) > 2 and value[1] == ":":
        raise ValueError(f"patch path must be workspace-relative: {path}")
    parts = [part for part in value.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise ValueError(f"patch path cannot escape workspace: {path}")
    return "/".join(parts)


def validate_patch_paths(paths: list[str] | tuple[str, ...], *, allow_generated_contracts: bool = False) -> list[str]:
    """Return a normalized Harness-only path list.

    Dataset, evaluator, observer, and test paths are frozen in the formal
    experiment.  Generated plan/contract JSON is also frozen unless a caller
    explicitly requests the diagnostic-only exception.
    """

    if not isinstance(paths, (list, tuple)) or any(not isinstance(item, str) for item in paths):
        raise ValueError("patch paths must be a list of strings")
    normalized = list(dict.fromkeys(normalize_patch_path(item) for item in paths))
    for path in normalized:
        if not path.startswith("src/videoact/"):
            raise ValueError(
                "Harness-only patch scope violation: allowed files are under src/videoact/; "
                f"rejected {path}"
            )
        parts = path.casefold().split("/")
        basename = parts[-1]
        if any(part in _FROZEN_PATH_PARTS for part in parts[:-1]) or basename in _FROZEN_PATH_PARTS:
            raise ValueError(f"Harness-only patch scope violation: frozen component path {path}")
        if basename.startswith("test_") or basename.startswith("observer") or basename.startswith("evaluator"):
            raise ValueError(f"Harness-only patch scope violation: frozen component path {path}")
        if not allow_generated_contracts and basename in {"trajectory.json", "camera_plan.json", "scene_contract.json"}:
            raise ValueError(
                "Harness-only patch scope violation: generated plan/contract contents are immutable; "
                f"rejected {path}"
            )
    return normalized


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


__all__ = [
    "PATCH_SCOPE_VERSION",
    "PatchVerdict",
    "attribute",
    "normalize_patch_path",
    "validate_patch_paths",
]

