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


OWNER_FILES = {
    "scene_parser": ["src/videoact/scene_contract.py"],
    "trajectory_planner": ["src/videoact/trajectory.py"],
    "camera_planner": ["src/videoact/camera.py"],
    "director_prompt_interpreter": ["src/videoact/director_prompt.py"],
    "director_event_scheduler": ["src/videoact/director_schedule.py"],
    "director_trajectory": ["src/videoact/director_trajectory.py"],
    "director_camera": ["src/videoact/director_camera.py"],
    "blender_code_agent": ["blender/real_proxy_job.py"],
    "blender_executor": ["src/videoact/blender_adapter.py"],
    "proxy_renderer": ["src/videoact/real_artifacts.py"],
    "evaluator": ["evaluator/deterministic.py"],
}


def aggregate_failures(records: list[dict[str, Any]]) -> FailureSummary:
    grouped: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    case_ids = set()
    for record in records:
        case_id = str(record.get("case_id", "unknown"))
        case_ids.add(case_id)
        for raw in record.get("findings", []):
            finding = raw if isinstance(raw, Finding) else Finding.model_validate(raw)
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
            for path in finding.evidence:
                if path not in group["evidence_paths"]:
                    group["evidence_paths"].append(path)

    groups = [FailureGroup.model_validate(group) for group in grouped.values()]
    groups.sort(key=lambda group: (-group.count, group.owner, group.failure_id))
    return FailureSummary(total_cases=len(case_ids), groups=groups)


def build_patch_brief(summary: FailureSummary) -> PatchBrief:
    if not summary.groups:
        raise ValueError("cannot build a patch brief without failure groups")
    group = summary.groups[0]
    return PatchBrief(
        owner=group.owner,
        root_cause_id=group.root_cause_id,
        affected_files=OWNER_FILES.get(group.owner, [f"owner/{group.owner}.py"]),
        observed_failure_pattern=(
            f"{group.failure_id} repeated in {group.count} case(s): {group.representative_message}"
        ),
        desired_behavior=f"Resolve {group.failure_id} while preserving the frozen contracts and evaluator.",
        rerun_command="python scripts/run_harness_eval.py --split train",
    )
