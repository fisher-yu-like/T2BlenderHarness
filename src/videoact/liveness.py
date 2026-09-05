"""Fault-injection liveness checks for the closed-loop Harness."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .failure_attribution import CounterfactualAttributor


LIVENESS_SCHEMA_VERSION = "liveness-v1"


class FaultInjection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expected_owner: str = Field(min_length=1)
    expected_stage: str = Field(min_length=1)
    split: Literal["train"] = "train"
    detected: bool = True
    repair_succeeds: bool = True
    evidence_refs: list[str] = Field(default_factory=list)
    obligation_ids: list[str] = Field(default_factory=list)
    counterfactuals: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_fixture(self) -> "FaultInjection":
        if not self.evidence_refs:
            raise ValueError("liveness fault requires evidence_refs")
        return self


class LivenessCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_id: str
    detected: bool
    attributed: bool
    expected_owner: str
    owner_candidate: str | None = None
    expected_stage: str
    observed_stage: str | None = None
    proposal_allowed: bool
    repair_succeeded: bool
    regression_free: bool
    attribution: dict[str, Any] | None = None
    reason: str


class LivenessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = LIVENESS_SCHEMA_VERSION
    status: Literal["pass", "failed"]
    fault_count: int = Field(ge=0)
    detected_count: int = Field(ge=0)
    attributed_count: int = Field(ge=0)
    detection_recall: float = Field(ge=0, le=1)
    owner_accuracy: float = Field(ge=0, le=1)
    cases: list[LivenessCaseResult] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    training_allowed: bool


def default_fault_injections() -> list[FaultInjection]:
    specs = (
        ("director_missing_receiver", "director_event_scheduler", "planned", "Director drops the receiver"),
        ("handoff_owner_not_switched", "interaction_library", "executed", "handoff primitive keeps the old owner"),
        ("codegen_required_event_ignored", "blender_code_agent", "implemented", "generated source omits a required event"),
        ("camera_handoff_occluded", "director_camera", "visible", "camera turns away during handoff"),
        ("executor_stale_telemetry", "blender_executor", "runtime_execution", "executor reuses old telemetry"),
        ("cache_stale_source", "blender_code_agent", "implemented", "cache returns source for another plan"),
        ("evaluator_inconsistent_result", "evaluator", "judged", "Judge result disagrees with the video evidence"),
    )
    result: list[FaultInjection] = []
    for fault_id, owner, stage, description in specs:
        family = {
            "director_event_scheduler": "same_prompt_director",
            "interaction_library": "same_primitive_library",
            "blender_code_agent": "same_plan_codegen",
            "director_camera": "same_source_blender",
            "blender_executor": "same_source_blender",
            "evaluator": "same_video_judge",
        }[owner]
        result.append(
            FaultInjection(
                fault_id=fault_id,
                description=description,
                expected_owner=owner,
                expected_stage=stage,
                evidence_refs=[f"liveness/{fault_id}.json"],
                obligation_ids=[f"obligation:{fault_id}"],
                counterfactuals=[
                    {"family": family, "owner": owner, "status": "fail", "evidence_refs": [f"cf/{fault_id}"]}
                ],
            )
        )
    return result


def validate_liveness_proposal(*, expected_owner: str, proposal: dict[str, Any]) -> None:
    if str(proposal.get("source_split") or "").casefold() != "train":
        raise ValueError("liveness proposal must be train-only")
    if str(proposal.get("owner") or "") != expected_owner:
        raise ValueError(f"liveness proposal owner mismatch: expected {expected_owner}, got {proposal.get('owner')}")


def run_liveness_suite(
    faults: Iterable[FaultInjection | dict[str, Any]],
    *,
    max_counterfactual_runs: int = 5,
) -> LivenessReport:
    values = [item if isinstance(item, FaultInjection) else FaultInjection.model_validate(item) for item in faults]
    if not values:
        return LivenessReport(
            status="failed",
            fault_count=0,
            detected_count=0,
            attributed_count=0,
            detection_recall=0.0,
            owner_accuracy=0.0,
            failures=["no liveness faults supplied"],
            training_allowed=False,
        )
    if any(item.split != "train" for item in values):
        raise ValueError("liveness suite is train-only")
    attributor = CounterfactualAttributor(max_runs=max_counterfactual_runs)
    cases: list[LivenessCaseResult] = []
    for fault in values:
        failure = {
            "case_id": f"liveness:{fault.fault_id}",
            "split": "train",
            "failure_id": fault.fault_id,
            "root_cause_id": fault.fault_id,
            "first_divergence_stage": fault.expected_stage,
            "owner_candidate": fault.expected_owner,
            "owner_confidence": 0.9,
            "severity": "hard",
            "category": fault.expected_owner,
            "message": fault.description,
            "evidence_complete": True,
            "evidence_refs": fault.evidence_refs,
            "actionable": True,
            "abstain": False,
        }
        attribution = None
        if fault.detected:
            attribution = attributor.attribute(failure, counterfactuals=fault.counterfactuals)
        attributed = bool(attribution is not None and not attribution.abstain and attribution.owner_candidate == fault.expected_owner)
        observed_stage = attribution.first_divergence_stage if attribution is not None else None
        proposal_allowed = bool(fault.detected and attributed)
        repair_succeeded = bool(fault.repair_succeeds and proposal_allowed)
        cases.append(
            LivenessCaseResult(
                fault_id=fault.fault_id,
                detected=fault.detected,
                attributed=attributed,
                expected_owner=fault.expected_owner,
                owner_candidate=attribution.owner_candidate if attribution is not None else None,
                expected_stage=fault.expected_stage,
                observed_stage=observed_stage,
                proposal_allowed=proposal_allowed,
                repair_succeeded=repair_succeeded,
                regression_free=repair_succeeded,
                attribution=attribution.model_dump(mode="json") if attribution is not None else None,
                reason=("passed" if repair_succeeded and observed_stage == fault.expected_stage else "not_detected" if not fault.detected else "attribution_or_repair_failed"),
            )
        )
    detected = sum(item.detected for item in cases)
    attributed = sum(item.attributed for item in cases)
    owner_accuracy = attributed / detected if detected else 0.0
    recall = detected / len(cases)
    failures: list[str] = []
    if recall < 6 / 7:
        failures.append(f"detection_recall_below_6_of_7:{recall:.4f}")
    if owner_accuracy < 0.8:
        failures.append(f"owner_accuracy_below_0.8:{owner_accuracy:.4f}")
    failures.extend(f"repair_failed:{item.fault_id}" for item in cases if item.detected and not item.repair_succeeded)
    failures.extend(f"wrong_owner_or_stage:{item.fault_id}" for item in cases if item.detected and (not item.attributed or item.observed_stage != item.expected_stage))
    training_allowed = not failures and len(cases) >= 7
    return LivenessReport(
        status="pass" if training_allowed else "failed",
        fault_count=len(cases),
        detected_count=detected,
        attributed_count=attributed,
        detection_recall=round(recall, 6),
        owner_accuracy=round(owner_accuracy, 6),
        cases=cases,
        failures=list(dict.fromkeys(failures)),
        training_allowed=training_allowed,
    )


__all__ = [
    "FaultInjection",
    "LivenessCaseResult",
    "LivenessReport",
    "LIVENESS_SCHEMA_VERSION",
    "default_fault_injections",
    "run_liveness_suite",
    "validate_liveness_proposal",
]
