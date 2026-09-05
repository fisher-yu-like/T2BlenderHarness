"""MetaHarness optimizer for real train records and one-owner acceptance."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .evolution import FailureSummary, PatchBrief, aggregate_failures, build_patch_brief
from .outer_loop import evaluate_candidate, write_optimization_record
from .cross_owner import validate_cross_owner_proposal
from .split_access import SplitAccessPolicy


class PatchProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str
    root_cause_id: str
    severity: Literal["info", "warning", "error", "hard"] = "error"
    affected_files: list[str]
    observed_failure_pattern: str
    desired_behavior: str
    rerun_command: str
    patch_scope: str = "one-harness-owner"
    owners: list[str] = Field(default_factory=list)
    cross_owner_exception: bool = False
    dependency_manifest: dict[str, Any] | None = None
    ablation_report: dict[str, Any] | None = None
    predicted_fixes: list[str] = Field(default_factory=list)
    predicted_regressions: list[str] = Field(default_factory=list)
    prediction_rationale: str = ""
    source_split: Literal["train"] = "train"
    source_case_ids: list[str] = Field(default_factory=list)
    attribution: dict[str, Any] | None = None
    target_obligations: list[str] = Field(default_factory=list)
    expected_artifact_changes: list[str] = Field(default_factory=list)
    minimum_effect: float = 0.0
    verification_plan: list[str] = Field(default_factory=list)
    attribution_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    patch_risk: float = Field(default=0.5, ge=0.0, le=1.0)


class MetaHarnessOptimizer:
    def __init__(self, *, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def summarize_train_records(
        train_records: list[dict[str, Any]],
        *,
        forbidden_case_ids: set[str] | None = None,
        split_policy: SplitAccessPolicy | Mapping[str, Any] | None = None,
    ) -> FailureSummary:
        """Validate the controller input and return the train-only summary.

        Keeping this check beside proposal construction prevents callers from
        accidentally aggregating a raw mixed-split list in a separate helper.
        The summary contains findings only; dev/test metrics are not accepted
        as a root-cause signal.
        """

        forbidden_case_ids = {str(item) for item in (forbidden_case_ids or set())}
        leaked = {str(record.get("case_id")) for record in train_records} & forbidden_case_ids
        if leaked:
            raise ValueError(f"test split case IDs leaked into train records: {sorted(leaked)}")
        non_train = sorted(
            {str(record.get("split")) for record in train_records if record.get("split") != "train"}
        )
        if non_train:
            raise ValueError(f"proposal records must be train-only: {non_train}")
        if split_policy is not None:
            policy = split_policy if isinstance(split_policy, SplitAccessPolicy) else SplitAccessPolicy.model_validate(split_policy)
            policy.validate_records(train_records)
        return aggregate_failures(train_records)

    def propose(
        self,
        train_records: list[dict[str, Any]],
        *,
        forbidden_case_ids: set[str] | None = None,
        attributions: dict[str, dict[str, Any]] | None = None,
        split_policy: SplitAccessPolicy | Mapping[str, Any] | None = None,
    ) -> PatchProposal:
        forbidden_case_ids = forbidden_case_ids or set()
        records = list(train_records)
        summary = self.summarize_train_records(
            records, forbidden_case_ids=forbidden_case_ids, split_policy=split_policy
        )
        if not summary.groups:
            raise ValueError("no actionable repeated failure with sufficient evidence in real train records")
        repeated = [
            group for group in summary.groups
            if len(set(group.affected_case_ids)) >= 2
        ]
        if not repeated:
            raise ValueError("a repeated failure must affect two distinct train cases")
        owners = {group.owner for group in repeated}
        if len(owners) != 1:
            raise ValueError(f"mixed-owner failure groups cannot form one proposal: {sorted(owners)}")
        brief: PatchBrief = build_patch_brief(summary)
        if brief.owner not in owners:
            raise ValueError("top failure group is not the sole repeated owner")
        predicted_fixes = sorted({case_id for group in repeated for case_id in group.affected_case_ids})
        attribution = None
        if attributions:
            selected = attributions.get(brief.root_cause_id)
            if selected is not None:
                if selected.get("abstain") is True or not selected.get("owner_candidate"):
                    raise ValueError("attribution abstained; proposal cannot be created")
                if str(selected.get("owner_candidate")) != brief.owner:
                    raise ValueError("attribution owner does not match repeated failure owner")
                attribution = dict(selected)
        return PatchProposal(
            **brief.model_dump(),
            predicted_fixes=predicted_fixes,
            predicted_regressions=[],
            source_split="train",
            source_case_ids=predicted_fixes,
            prediction_rationale=(
                "The proposal targets a repeated train-only failure affecting the predicted cases; "
                "no known regression case was identified, so predicted_regressions is empty."
            ),
            attribution=attribution,
        )

    @staticmethod
    def _group_obligations(
        train_records: list[dict[str, Any]],
        group_root: str,
        group_owner: str,
    ) -> list[str]:
        values: list[str] = []
        for record in train_records:
            for raw in record.get("findings", []) or []:
                row = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
                if not isinstance(row, dict):
                    continue
                root = str(row.get("root_cause_id") or row.get("failure_id") or "")
                owner = str(row.get("owner") or "")
                if root != group_root or owner != group_owner:
                    continue
                for key in ("affected_obligation_ids", "target_obligations", "obligation_ids"):
                    candidate = row.get(key)
                    if isinstance(candidate, (list, tuple, set)):
                        values.extend(str(item) for item in candidate if str(item).strip())
        return list(dict.fromkeys(values))

    @staticmethod
    def _expected_artifacts(owner: str) -> list[str]:
        return {
            "director_camera": ["camera_plan", "camera_telemetry", "video_visibility"],
            "camera_planner": ["camera_plan", "camera_telemetry", "video_visibility"],
            "director_trajectory": ["trajectory_plan", "fcurves", "contact_telemetry"],
            "trajectory_planner": ["trajectory_plan", "fcurves", "contact_telemetry"],
            "blender_code_agent": ["generated_source", "source_fingerprint", "candidate_blend"],
            "blender_executor": ["completion", "retry_report", "provenance"],
            "proxy_renderer": ["candidate_blend", "proxy_video", "artifact_manifest"],
        }.get(owner, ["owner_contract_evidence"])

    def rank_proposals(
        self,
        train_records: list[dict[str, Any]],
        *,
        forbidden_case_ids: set[str] | None = None,
        attributions: dict[str, dict[str, Any]] | None = None,
        split_policy: SplitAccessPolicy | Mapping[str, Any] | None = None,
    ) -> list[PatchProposal]:
        """Rank repeated owner groups and retain non-selected groups as backlog.

        This is the T12 policy surface.  The legacy :meth:`propose` method
        remains strict and blocks mixed owners for compatibility; formal
        callers that opt into ranking can select exactly one owner while
        preserving every other group in the returned backlog.
        """

        forbidden_case_ids = forbidden_case_ids or set()
        records = list(train_records)
        self.summarize_train_records(
            records, forbidden_case_ids=forbidden_case_ids, split_policy=split_policy
        )
        summary = aggregate_failures(records, attributions=attributions)
        repeated = [group for group in summary.groups if len(set(group.affected_case_ids)) >= 2]
        proposals: list[PatchProposal] = []
        severity_rank = {"info": 0, "warning": 1, "error": 2, "hard": 3}
        for group in repeated:
            if group.owner in {"evaluator", "observer"}:
                continue
            try:
                brief = build_patch_brief(FailureSummary(total_cases=summary.total_cases, groups=[group]))
            except ValueError:
                continue
            obligations = self._group_obligations(records, group.root_cause_id, group.owner)
            attribution = (attributions or {}).get(group.root_cause_id)
            proposal_payload = brief.model_dump(mode="json")
            proposal_payload.update(
                {
                    "severity": group.severity,
                    "target_obligations": obligations,
                    "expected_artifact_changes": self._expected_artifacts(group.owner),
                    "verification_plan": [
                        "owner_challenge_set",
                        "unit_and_contract_tests",
                        "paired_train_dev_rerun",
                        "patch_impact_proof",
                    ],
                    "predicted_fixes": sorted(group.affected_case_ids),
                    "source_case_ids": sorted(group.affected_case_ids),
                    "source_split": "train",
                    "attribution": dict(attribution) if isinstance(attribution, dict) else None,
                }
            )
            proposal = PatchProposal(**proposal_payload)
            proposals.append(proposal)
        proposals.sort(
            key=lambda item: (
                -severity_rank.get(item.severity if hasattr(item, "severity") else "error", 2),
                -len(item.source_case_ids),
                -float(item.attribution_confidence),
                -len(item.target_obligations),
                float(item.patch_risk),
                item.owner,
                item.root_cause_id,
            )
        )
        return proposals

    def propose_ranked(self, train_records: list[dict[str, Any]], **kwargs: Any) -> PatchProposal:
        ranked = self.rank_proposals(train_records, **kwargs)
        if not ranked:
            raise ValueError("no repeated train failure can produce a ranked proposal")
        return ranked[0]

    def build_ranked_proposals(self, train_records: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        ranked = self.rank_proposals(train_records, **kwargs)
        return {
            "selected": ranked[0].model_dump(mode="json") if ranked else None,
            "backlog": [item.model_dump(mode="json") for item in ranked[1:]],
            "proposals": [item.model_dump(mode="json") for item in ranked],
        }

    def record_acceptance(
        self,
        proposal: PatchProposal,
        *,
        before: dict[str, float],
        after: dict[str, float],
        train: dict[str, Any],
        dev: dict[str, Any],
        patch_diff: str,
        impact_proof: dict[str, Any] | None = None,
        require_impact_proof: bool = False,
    ) -> dict[str, Any]:
        cross_owner = validate_cross_owner_proposal(proposal.model_dump(mode="json"))
        decision = evaluate_candidate(
            before,
            after,
            train,
            dev,
            owner=proposal.owner,
            impact_proof=impact_proof,
            require_impact_proof=require_impact_proof,
        )
        record = {
            "owner": proposal.owner,
            "affected_files": proposal.affected_files,
            "patch_scope": proposal.patch_scope,
            "patch_brief": proposal.model_dump(mode="json"),
            "patch_diff": patch_diff,
            "patch_impact": impact_proof,
            "acceptance": decision.model_dump(mode="json"),
            "acceptance_checks": decision.checks,
            "failed_checks": decision.failed_checks,
            "train_before": before,
            "train_after": after,
            "dev_gate": dev,
            "cross_owner_audit": cross_owner,
        }
        write_optimization_record(self.output_dir / "optimization_record.jsonl", record)
        return record
