"""Failure aggregation and single-owner patch briefs for Harness evolution."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts import Finding


class FailureGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_id: str
    root_cause_id: str
    owner: str
    category: str
    severity: str
    count: int = Field(ge=1)
    affected_case_ids: list[str]
    evidence_paths: list[str] = []
    representative_message: str
    attribution_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_obligation_impact: int = Field(default=0, ge=0)
    patch_risk: float = Field(default=0.5, ge=0.0, le=1.0)


class FailureSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int = Field(ge=0)
    groups: list[FailureGroup] = []


class PatchBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str
    root_cause_id: str
    affected_files: list[str]
    observed_failure_pattern: str
    desired_behavior: str
    rerun_command: str
    target_obligations: list[str] = []
    expected_artifact_changes: list[str] = []
    minimum_effect: float = 0.0
    verification_plan: list[str] = []
    attribution_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_risk: float = Field(default=0.5, ge=0.0, le=1.0)


OWNER_FILES = {
    "scene_parser": ["src/videoact/scene_contract.py"],
    "trajectory_planner": ["src/videoact/trajectory.py"],
    "camera_planner": ["src/videoact/camera.py"],
    "director_prompt_interpreter": ["src/videoact/director_prompt.py"],
    "director_event_scheduler": ["src/videoact/director_schedule.py"],
    "director_trajectory": ["src/videoact/director_trajectory.py"],
    "director_camera": ["src/videoact/director_camera.py"],
    # Generated Blender source, evaluator code, and the dataset are frozen
    # during Harness evolution.  A proposal may point only at the Harness
    # owner that produced the behavior; the generated job is evidence, not a
    # patch target.
    "blender_code_agent": [
        "src/videoact/blender_code_agent.py",
        "src/videoact/codex_self_provider.py",
    ],
    "blender_executor": ["src/videoact/orchestrator.py"],
    "proxy_renderer": ["src/videoact/real_artifacts.py"],
}

FROZEN_OWNERS = {"evaluator"}


def _owner_patch_risk(owner: str) -> float:
    return {
        "director_prompt_interpreter": 0.25,
        "director_event_scheduler": 0.25,
        "director_trajectory": 0.30,
        "director_camera": 0.30,
        "trajectory_planner": 0.30,
        "camera_planner": 0.30,
        "blender_code_agent": 0.60,
        "blender_executor": 0.45,
        "proxy_renderer": 0.50,
    }.get(owner, 0.75)


def aggregate_failures(
    records: list[dict[str, Any]],
    *,
    attributions: dict[str, dict[str, Any]] | None = None,
) -> FailureSummary:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    case_ids = set()
    for record in records:
        case_id = str(record.get("case_id", "unknown"))
        case_ids.add(case_id)
        for raw in record.get("findings", []):
            finding = raw if isinstance(raw, Finding) else Finding.model_validate(raw)
            evidence = [str(path).strip() for path in finding.evidence if str(path).strip()]
            if not evidence:
                # A repeated label without a case-local evidence reference is
                # not sufficient to identify a patchable Harness failure.
                continue
            root_cause_id = finding.root_cause_id or finding.failure_id
            key = (finding.failure_id, root_cause_id, finding.owner, finding.category, finding.severity)
            group = grouped.setdefault(
                key,
                {
                    "failure_id": finding.failure_id,
                    "root_cause_id": root_cause_id,
                    "owner": finding.owner,
                    "category": finding.category,
                    "severity": finding.severity,
                    "count": 0,
                    "affected_case_ids": [],
                    "evidence_paths": [],
                    "representative_message": finding.message,
                },
            )
            group["count"] += 1
            if case_id not in group["affected_case_ids"]:
                group["affected_case_ids"].append(case_id)
            for path in evidence:
                if path not in group["evidence_paths"]:
                    group["evidence_paths"].append(path)

            root = str(root_cause_id)
            attribution = (attributions or {}).get(root, {})
            confidence = attribution.get("owner_confidence", attribution.get("confidence", 0.0))
            try:
                group["attribution_confidence"] = max(
                    float(group.get("attribution_confidence", 0.0)),
                    min(1.0, max(0.0, float(confidence))),
                )
            except (TypeError, ValueError):
                pass
            obligation_ids = finding.model_dump(mode="json").get("affected_obligation_ids", []) if isinstance(finding, Finding) else []
            if isinstance(raw, dict):
                obligation_ids = raw.get("affected_obligation_ids", raw.get("target_obligations", obligation_ids))
            if isinstance(obligation_ids, (list, tuple, set)):
                group["expected_obligation_impact"] = max(
                    int(group.get("expected_obligation_impact", 0)), len(set(str(item) for item in obligation_ids))
                )
            group["patch_risk"] = _owner_patch_risk(finding.owner)

    groups = [FailureGroup.model_validate(group) for group in grouped.values()]
    groups.sort(key=lambda group: (-group.count, group.owner, group.failure_id))
    return FailureSummary(total_cases=len(case_ids), groups=groups)


def build_patch_brief(summary: FailureSummary) -> PatchBrief:
    if not summary.groups:
        raise ValueError("cannot build a patch brief without failure groups")
    group = summary.groups[0]
    if group.owner in FROZEN_OWNERS:
        raise ValueError(f"owner is frozen and cannot produce a Harness patch: {group.owner}")
    if group.owner not in OWNER_FILES:
        raise ValueError(f"unknown Harness patch owner: {group.owner}")
    return PatchBrief(
        owner=group.owner,
        root_cause_id=group.root_cause_id,
        affected_files=OWNER_FILES.get(group.owner, [f"owner/{group.owner}.py"]),
        observed_failure_pattern=(
            f"{group.failure_id} repeated in {group.count} case(s): {group.representative_message}"
        ),
        desired_behavior=f"Resolve {group.failure_id} while preserving the frozen contracts and evaluator.",
        rerun_command="python scripts/run_harness_eval.py --split train",
        attribution_confidence=group.attribution_confidence,
        patch_risk=group.patch_risk,
        minimum_effect=1.0,
    )
