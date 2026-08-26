"""MetaHarness optimizer for real train records and one-owner acceptance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from .evolution import PatchBrief, aggregate_failures, build_patch_brief
from .outer_loop import evaluate_candidate, write_optimization_record


class PatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str
    affected_files: list[str]
    observed_failure_pattern: str
    desired_behavior: str
    rerun_command: str
    patch_scope: str = "one-harness-owner"


class MetaHarnessOptimizer:
    def __init__(self, *, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def propose(
        self,
        train_records: list[dict[str, Any]],
        *,
        forbidden_case_ids: set[str] | None = None,
    ) -> PatchProposal:
        forbidden_case_ids = forbidden_case_ids or set()
        leaked = {str(record.get("case_id")) for record in train_records} & forbidden_case_ids
        if leaked:
            raise ValueError(f"test split case IDs leaked into train records: {sorted(leaked)}")
        summary = aggregate_failures(train_records)
        if not summary.groups:
            raise ValueError("no actionable repeated failure in real train records")
        brief: PatchBrief = build_patch_brief(summary)
        return PatchProposal(**brief.model_dump())

    def record_acceptance(
        self,
        proposal: PatchProposal,
        *,
        before: dict[str, float],
        after: dict[str, float],
        train: dict[str, Any],
        dev: dict[str, Any],
        patch_diff: str,
    ) -> dict[str, Any]:
        decision = evaluate_candidate(before, after, train, dev)
        record = {
            "owner": proposal.owner,
            "affected_files": proposal.affected_files,
            "patch_scope": proposal.patch_scope,
            "patch_brief": proposal.model_dump(mode="json"),
            "patch_diff": patch_diff,
            "acceptance": decision.model_dump(mode="json"),
            "train_before": before,
            "train_after": after,
            "dev_gate": dev,
        }
        write_optimization_record(self.output_dir / "optimization_record.jsonl", record)
        return record
