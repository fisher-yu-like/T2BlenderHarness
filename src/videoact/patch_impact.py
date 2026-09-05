"""Causal impact proof for accepted Harness patches.

An edit is not useful merely because a source file changed.  This module
requires a production call-site anchor and an owner-specific downstream
signal.  It also rejects cache/template reuse and aggregate-score-only claims.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field

from .source_fingerprints import compare_source_fingerprints, source_fingerprint


PATCH_IMPACT_SCHEMA_VERSION = "patch-impact-proof-v1"


class PatchImpactProof(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PATCH_IMPACT_SCHEMA_VERSION
    edit_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    root_cause_id: str | None = None
    source_diff_present: bool
    production_call_sites_changed: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    plan_hash_changed: bool = False
    obligation_hash_changed: bool = False
    code_hash_changed: bool = False
    code_ast_changed: bool = False
    code_call_sites_changed: bool = False
    blend_hash_changed: bool = False
    camera_plan_changed: bool = False
    camera_telemetry_changed: bool = False
    key_event_visibility_changed: bool = False
    telemetry_changed: bool = False
    fcurve_changed: bool = False
    contact_changed: bool = False
    completion_changed: bool = False
    retry_changed: bool = False
    provenance_changed: bool = False
    telemetry_delta: dict[str, Any] = Field(default_factory=dict)
    video_delta: dict[str, Any] = Field(default_factory=dict)
    target_metric_delta: float | None = None
    target_obligation_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    metric_evidence_refs: list[str] = Field(default_factory=list)
    cache_reuse_detected: bool = False
    status: Literal["pass", "no_effect_patch", "rejected", "blocked"]
    causal_chain_complete: bool = False
    reason: str = Field(min_length=1)


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(child) for child in value]
    return value


def _hash_value(value: Any) -> str:
    payload = json.dumps(_dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    dumped = _dump(value)
    return dict(dumped) if isinstance(dumped, Mapping) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _nested(snapshot: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in snapshot:
            return snapshot[key]
    return None


def _changed(before: Mapping[str, Any], after: Mapping[str, Any], *keys: str) -> bool:
    left = _nested(before, *keys)
    right = _nested(after, *keys)
    return left is not None and right is not None and _hash_value(left) != _hash_value(right)


def _hash_changed(before: Mapping[str, Any], after: Mapping[str, Any], *keys: str) -> bool:
    left = _nested(before, *keys)
    right = _nested(after, *keys)
    if left is None or right is None:
        return False
    return str(left) != str(right)


def _source_from(snapshot: Mapping[str, Any]) -> str | None:
    value = _nested(snapshot, "source", "generated_source", "generated_code", "source_text")
    return value if isinstance(value, str) and value.strip() else None


def _metric_evidence(
    after: Mapping[str, Any],
    explicit: Sequence[str] | None,
) -> list[str]:
    values = list(explicit or [])
    values.extend(_strings(after.get("metric_evidence_refs")))
    values.extend(_strings(after.get("evidence_refs")))
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _owner_family(owner: str) -> str:
    value = owner.casefold()
    if "camera" in value:
        return "camera"
    if "code" in value or "source" in value:
        return "code"
    if any(token in value for token in ("primitive", "library", "interaction")):
        return "interaction"
    if any(token in value for token in ("executor", "renderer", "proxy")):
        return "executor"
    if "director" in value or "planner" in value or "parser" in value:
        return "director"
    return "unknown"


def build_patch_impact_proof(
    proposal: Mapping[str, Any] | Any,
    before: Mapping[str, Any] | Any,
    after: Mapping[str, Any] | Any,
    *,
    changed_files: Sequence[str] = (),
    production_call_sites_changed: Sequence[str] = (),
    target_obligation_ids: Sequence[str] = (),
    evidence_refs: Sequence[str] = (),
    metric_evidence_refs: Sequence[str] = (),
    target_metric_delta: float | None = None,
    source_diff_present: bool = True,
) -> PatchImpactProof:
    """Build and validate a proof from explicit before/after observations."""

    proposal_row = _as_dict(proposal)
    left = _as_dict(before)
    right = _as_dict(after)
    owner = str(proposal_row.get("owner") or "").strip()
    if not owner:
        raise ValueError("impact proof requires one owner")
    edit_id = str(proposal_row.get("proposal_id") or proposal_row.get("edit_id") or proposal_row.get("root_cause_id") or "unknown-edit")
    root_cause_id = str(proposal_row.get("root_cause_id") or "") or None
    files = list(dict.fromkeys(str(item) for item in changed_files if str(item).strip()))
    call_sites = list(dict.fromkeys(str(item) for item in production_call_sites_changed if str(item).strip()))
    if not call_sites:
        call_sites = _strings(right.get("production_call_sites_changed")) or _strings(right.get("production_call_sites"))
    obligations = list(dict.fromkeys(str(item) for item in (target_obligation_ids or proposal_row.get("target_obligations") or proposal_row.get("affected_obligation_ids") or []) if str(item).strip()))
    refs = list(dict.fromkeys([*(_strings(evidence_refs)), *_strings(right.get("evidence_refs"))]))
    plan_changed = _hash_changed(left, right, "plan_hash", "director_plan_hash")
    obligation_changed = _hash_changed(left, right, "obligation_hash", "obligations_hash", "compilation_fingerprint")
    code_changed = _hash_changed(left, right, "code_hash", "source_hash")
    blend_changed = _hash_changed(left, right, "blend_hash", "candidate_blend_hash")
    source_before = _source_from(left)
    source_after = _source_from(right)
    code_ast_changed = False
    code_call_sites_changed = False
    cache_reuse = False
    if source_before is not None and source_after is not None:
        comparison = compare_source_fingerprints(source_before, source_after)
        code_ast_changed = not bool(comparison.get("normalized_ast_equal"))
        code_call_sites_changed = not bool(comparison.get("library_call_sequence_equal"))
        raw_changed = _hash_value(source_before) != _hash_value(source_after)
        cache_reuse = raw_changed and bool(comparison.get("probable_template_reuse"))
        code_changed = code_changed or raw_changed
    camera_plan_changed = _changed(left, right, "camera_plan", "camera_plan_hash", "camera")
    camera_telemetry_changed = _changed(left, right, "camera_telemetry", "camera_telemetry_delta")
    camera_telemetry_changed = camera_telemetry_changed or bool(_nested(right, "camera_telemetry_changed"))
    key_event_visibility_changed = _changed(left, right, "key_event_visibility", "visibility", "event_visibility")
    telemetry_changed = _changed(left, right, "telemetry", "runtime_telemetry", "telemetry_hash")
    fcurve_changed = _changed(left, right, "fcurve", "fcurves", "animation_fcurves")
    contact_changed = _changed(left, right, "contact", "contacts", "contact_events")
    completion_changed = _changed(left, right, "completion", "completion_status", "status")
    retry_changed = _changed(left, right, "retry_count", "render_retry_count", "retries")
    provenance_changed = _changed(left, right, "provenance", "provenance_hash", "fingerprint")
    telemetry_delta = _as_dict(_nested(right, "telemetry_delta", "telemetry_changes"))
    video_delta = _as_dict(_nested(right, "video_delta", "video_changes"))
    metric_delta = target_metric_delta
    if metric_delta is None and right.get("target_metric_delta") is not None:
        try:
            metric_delta = float(right["target_metric_delta"])
        except (TypeError, ValueError):
            metric_delta = None
    metric_refs = _metric_evidence(right, metric_evidence_refs)
    family = _owner_family(owner)
    downstream = {
        "director": plan_changed or obligation_changed,
        "code": code_ast_changed or code_call_sites_changed,
        "interaction": telemetry_changed or fcurve_changed or contact_changed,
        "camera": camera_plan_changed or camera_telemetry_changed or key_event_visibility_changed,
        "executor": completion_changed or retry_changed or provenance_changed,
    }.get(family, plan_changed or code_changed or blend_changed or telemetry_changed)
    status: Literal["pass", "no_effect_patch", "rejected", "blocked"] = "pass"
    reason = "owner-specific downstream impact and production call-site evidence verified"
    if not source_diff_present:
        status = "rejected"
        reason = "source diff is missing"
    elif cache_reuse:
        status = "rejected"
        reason = "code cache/template reuse detected: changed source has the same semantic fingerprint"
    elif not call_sites:
        status = "rejected"
        reason = "patch has no production call-site change"
    elif family == "camera" and not downstream:
        status = "no_effect_patch"
        reason = "camera patch changed neither camera plan nor camera telemetry/key-event visibility"
    elif not downstream:
        status = "no_effect_patch"
        reason = f"source diff has no observed downstream impact for owner family {family}"
    elif metric_delta is not None:
        linked = bool(obligations) and any(
            any(identifier.casefold() in ref.casefold() for identifier in obligations)
            or "obligation" in ref.casefold()
            for ref in metric_refs
        )
        if not linked:
            status = "rejected"
            reason = "target metric delta is not traced to obligation evidence"
    complete = status == "pass" and bool(call_sites) and bool(obligations or family in {"code", "executor"})
    if status == "pass" and not complete:
        status = "blocked"
        reason = "impact proof is incomplete: target obligation/evidence anchor is missing"
    return PatchImpactProof(
        edit_id=edit_id,
        owner=owner,
        root_cause_id=root_cause_id,
        source_diff_present=bool(source_diff_present),
        production_call_sites_changed=call_sites,
        changed_files=files,
        plan_hash_changed=plan_changed,
        obligation_hash_changed=obligation_changed,
        code_hash_changed=code_changed,
        code_ast_changed=code_ast_changed,
        code_call_sites_changed=code_call_sites_changed,
        blend_hash_changed=blend_changed,
        camera_plan_changed=camera_plan_changed,
        camera_telemetry_changed=camera_telemetry_changed,
        key_event_visibility_changed=key_event_visibility_changed,
        telemetry_changed=telemetry_changed,
        fcurve_changed=fcurve_changed,
        contact_changed=contact_changed,
        completion_changed=completion_changed,
        retry_changed=retry_changed,
        provenance_changed=provenance_changed,
        telemetry_delta=telemetry_delta,
        video_delta=video_delta,
        target_metric_delta=metric_delta,
        target_obligation_ids=obligations,
        evidence_refs=refs,
        metric_evidence_refs=metric_refs,
        cache_reuse_detected=cache_reuse,
        status=status,
        causal_chain_complete=complete,
        reason=reason,
    )


def validate_patch_impact(proof: Mapping[str, Any] | PatchImpactProof) -> PatchImpactProof:
    """Fail closed unless a complete accepted proof is supplied."""

    model = proof if isinstance(proof, PatchImpactProof) else PatchImpactProof.model_validate(proof)
    if model.status != "pass" or not model.causal_chain_complete:
        raise ValueError(f"Patch Impact Proof is not acceptable: {model.status}: {model.reason}")
    if not model.source_diff_present:
        raise ValueError("Patch Impact Proof requires a source diff")
    if not model.production_call_sites_changed:
        raise ValueError("Patch Impact Proof requires a production call-site change")
    return model


def prove_patch_impact(*args: Any, **kwargs: Any) -> PatchImpactProof:
    """Short functional alias for callers building one proof."""

    return build_patch_impact_proof(*args, **kwargs)


__all__ = [
    "PATCH_IMPACT_SCHEMA_VERSION",
    "PatchImpactProof",
    "build_patch_impact_proof",
    "prove_patch_impact",
    "validate_patch_impact",
]
