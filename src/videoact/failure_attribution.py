"""Bounded counterfactual attribution for normalized train failures."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field


ATTRIBUTION_SCHEMA_VERSION = "failure-attribution-v1"
DEFAULT_MAX_COUNTERFACTUAL_RUNS = 5
UPSTREAM_OWNERS = {
    "director_prompt_interpreter",
    "director_event_scheduler",
    "director_trajectory",
    "director_camera",
    "blender_code_agent",
}


class CounterfactualRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    split: Literal["train"]
    owner: str | None = None
    status: str = Field(min_length=1)
    input_hash: str = Field(min_length=64, max_length=64)
    output_hash: str = Field(min_length=64, max_length=64)
    evidence_refs: list[str] = Field(default_factory=list)


class CounterfactualBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ATTRIBUTION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    split: Literal["train"]
    parent_hash: str = Field(min_length=64, max_length=64)
    child_hashes: list[str] = Field(default_factory=list)
    runs: list[CounterfactualRun] = Field(default_factory=list)
    budget: int = Field(gt=0)


class FailureAttribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = ATTRIBUTION_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    split: Literal["train"]
    failure_id: str = Field(min_length=1)
    root_cause_id: str = Field(min_length=1)
    first_divergence_stage: str = Field(min_length=1)
    owner_candidate: str | None = None
    owner_confidence: float = Field(default=0.0, ge=0, le=1)
    excluded_owners: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    parent_hash: str = Field(min_length=64, max_length=64)
    child_hashes: list[str] = Field(default_factory=list)
    counterfactual_count: int = Field(ge=0)
    budget: int = Field(gt=0)
    abstain: bool
    reason: str = Field(min_length=1)
    causal_chain: list[str] = Field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.owner_confidence


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(child) for child in value]
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(_dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    dumped = _dump(value)
    return dumped if isinstance(dumped, dict) else {}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _family_defaults(family: str) -> tuple[str, str]:
    normalized = family.casefold()
    if "prompt" in normalized or "director" in normalized:
        return "planned", "director_prompt_interpreter"
    if "plan" in normalized and "code" in normalized:
        return "implemented", "blender_code_agent"
    if "source" in normalized and "blender" in normalized:
        return "runtime_execution", "blender_executor"
    if "blend" in normalized and "observer" in normalized:
        return "runtime_execution", "trusted_observer"
    if "video" in normalized and "judge" in normalized:
        return "judged", "evaluator"
    if "primitive" in normalized or "library" in normalized:
        return "runtime_execution", "interaction_library"
    return "runtime_execution", "unassigned"


class CounterfactualRunner:
    """Materialize bounded parent/child identities for counterfactual specs."""

    def __init__(self, *, max_runs: int = DEFAULT_MAX_COUNTERFACTUAL_RUNS) -> None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive")
        self.max_runs = int(max_runs)

    def run(
        self,
        *,
        case_id: str,
        prompt: str,
        split: str,
        variants: Sequence[Mapping[str, Any]],
    ) -> CounterfactualBatch:
        if str(split).casefold() != "train":
            raise ValueError("counterfactual runner is train-only")
        if len(variants) > self.max_runs:
            raise ValueError(f"counterfactual budget exceeded: {len(variants)} > {self.max_runs}")
        parent_hash = _hash({"case_id": case_id, "prompt": prompt, "split": "train"})
        runs: list[CounterfactualRun] = []
        child_hashes: list[str] = []
        for index, variant in enumerate(variants, start=1):
            item = _as_dict(variant)
            family = str(item.get("family") or "counterfactual")
            stage, default_owner = _family_defaults(family)
            output = item.get("output", item)
            input_hash = _hash({"parent_hash": parent_hash, "family": family, "variant": item.get("input", {})})
            output_hash = _hash(output)
            child_hashes.append(output_hash)
            runs.append(
                CounterfactualRun(
                    run_id=f"{case_id}:cf:{index:02d}",
                    family=family,
                    stage=str(item.get("stage") or stage),
                    split="train",
                    owner=str(item.get("owner") or default_owner) or None,
                    status=str(item.get("status") or "completed"),
                    input_hash=input_hash,
                    output_hash=output_hash,
                    evidence_refs=_strings(item.get("evidence_refs")),
                )
            )
        return CounterfactualBatch(
            case_id=str(case_id),
            split="train",
            parent_hash=parent_hash,
            child_hashes=child_hashes,
            runs=runs,
            budget=self.max_runs,
        )


class CounterfactualAttributor:
    """Attribute one normalized failure using a fixed counterfactual budget."""

    def __init__(self, *, max_runs: int = DEFAULT_MAX_COUNTERFACTUAL_RUNS, confidence_threshold: float = 0.6) -> None:
        if max_runs <= 0:
            raise ValueError("max_runs must be positive")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.max_runs = int(max_runs)
        self.confidence_threshold = float(confidence_threshold)

    def attribute(
        self,
        failure: Mapping[str, Any] | Any,
        *,
        counterfactuals: Sequence[Mapping[str, Any] | CounterfactualRun] = (),
        forbidden_case_ids: set[str] | None = None,
    ) -> FailureAttribution:
        raw = _as_dict(failure)
        case_id = str(raw.get("case_id") or "").strip()
        split = str(raw.get("split") or "train").casefold()
        if split != "train":
            raise ValueError("failure attribution is train-only")
        forbidden = {str(item) for item in (forbidden_case_ids or set())}
        if case_id in forbidden:
            raise ValueError(f"forbidden case entered attribution: {case_id}")
        serialized = json.dumps(raw, ensure_ascii=False, sort_keys=True, default=str)
        leaked = sorted(item for item in forbidden if item and item in serialized)
        if leaked:
            raise ValueError(f"forbidden case entered attribution: {leaked}")
        if len(counterfactuals) > self.max_runs:
            raise ValueError(f"counterfactual budget exceeded: {len(counterfactuals)} > {self.max_runs}")
        failure_id = str(raw.get("failure_id") or "failure")
        root_cause_id = str(raw.get("root_cause_id") or failure_id)
        first_stage = str(raw.get("first_divergence_stage") or "runtime_execution")
        base_owner = str(raw.get("owner_candidate") or "") or None
        base_confidence = float(raw.get("owner_confidence", raw.get("confidence", 0.0)) or 0.0)
        parent_hash = _hash({
            "case_id": case_id,
            "failure_id": failure_id,
            "root_cause_id": root_cause_id,
            "first_divergence_stage": first_stage,
            "evidence_refs": _strings(raw.get("evidence_refs") or raw.get("evidence")),
        })
        child_hashes: list[str] = []
        candidate_owners: set[str] = set()
        excluded: set[str] = set()
        refs = _strings(raw.get("evidence_refs") or raw.get("evidence"))
        causal_chain = ["failure evidence", "bounded counterfactuals"]
        timeout_seen = False
        for index, variant_value in enumerate(counterfactuals, start=1):
            variant = _as_dict(variant_value)
            family = str(variant.get("family") or "counterfactual")
            family_stage, family_owner = _family_defaults(family)
            owner = str(variant.get("owner") or family_owner) or None
            status = str(variant.get("status") or "completed").casefold()
            child_hashes.append(_hash({"parent_hash": parent_hash, "index": index, "variant": variant}))
            refs.extend(_strings(variant.get("evidence_refs")))
            if status in {"timeout", "timed_out"}:
                timeout_seen = True
                if owner:
                    candidate_owners.add(owner)
                excluded.update(UPSTREAM_OWNERS)
                first_stage = first_stage if first_stage != "runtime_execution" else family_stage
            elif status in {"fail", "failed", "error", "mismatch"}:
                if owner:
                    candidate_owners.add(owner)
                if family_stage and first_stage == "runtime_execution":
                    first_stage = family_stage
            elif owner:
                excluded.add(owner)
        # Passing counterfactuals explicitly eliminate their owner.  An
        # executor timeout may not be blamed on a Director or code generator.
        candidate_owners -= excluded
        if base_owner and base_confidence >= self.confidence_threshold and bool(raw.get("evidence_complete", True)) and not bool(raw.get("abstain")):
            candidate_owners.add(base_owner)
        if len(candidate_owners) == 1:
            selected_owner = next(iter(candidate_owners))
            abstain = False
            confidence = 0.95 if len(counterfactuals) else min(1.0, base_confidence)
            reason = "counterfactual_owner_isolated" if counterfactuals else "evidence_owner_supported"
        elif len(candidate_owners) > 1:
            selected_owner = None
            abstain = True
            confidence = 0.0
            reason = "multiple_indistinguishable_owner_candidates"
        elif base_owner and base_confidence >= self.confidence_threshold and not bool(raw.get("abstain")):
            selected_owner = base_owner
            abstain = False
            confidence = base_confidence
            reason = "evidence_owner_supported"
        else:
            selected_owner = None
            abstain = True
            confidence = 0.0
            reason = "attribution_uncertain" if not timeout_seen else "counterfactual_timeout"
        if selected_owner:
            excluded.discard(selected_owner)
        return FailureAttribution(
            case_id=case_id,
            split="train",
            failure_id=failure_id,
            root_cause_id=root_cause_id,
            first_divergence_stage=first_stage,
            owner_candidate=selected_owner,
            owner_confidence=confidence,
            excluded_owners=sorted(excluded),
            evidence_refs=list(dict.fromkeys(refs)),
            parent_hash=parent_hash,
            child_hashes=child_hashes,
            counterfactual_count=len(counterfactuals),
            budget=self.max_runs,
            abstain=abstain,
            reason=reason,
            causal_chain=causal_chain + (["timeout isolated to execution"] if timeout_seen else ["owner decision"]),
        )


def attribute_failure(failure: Mapping[str, Any] | Any, **kwargs: Any) -> FailureAttribution:
    return CounterfactualAttributor().attribute(failure, **kwargs)


def run_counterfactuals(**kwargs: Any) -> CounterfactualBatch:
    return CounterfactualRunner().run(**kwargs)


__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "CounterfactualAttributor",
    "CounterfactualBatch",
    "CounterfactualRun",
    "CounterfactualRunner",
    "FailureAttribution",
    "attribute_failure",
    "run_counterfactuals",
]
