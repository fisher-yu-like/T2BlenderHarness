"""Deterministic evaluator with hard gates and stable JSON reports."""

from __future__ import annotations

from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field

from videoact.contracts import ExecutionResult, Finding, SceneContract, TrajectoryPlan
from videoact.director_contracts import DirectorPlan

from .camera_metrics import check_camera_coverage, check_camera_motion_intent
from .physics_metrics import (
    check_attachment_contact,
    check_support_before_grasp,
    check_velocity_continuity,
)
from .prompt_predicates import check_event_order, check_required_event_coverage
from .trajectory_metrics import check_trajectory_phase_alignment
from .findings import deduplicate_findings, score_findings
from .director_metrics import evaluate_director_plan
from .interaction_metrics import evaluate_interactions


def _runtime_telemetry_findings(telemetry: dict[str, Any]) -> list[Finding]:
    """Convert executable runtime observations into report findings.

    Blender runtime checks are independent evidence. They must be surfaced in
    the real-run report instead of remaining inert JSON fields.
    """
    findings: list[Finding] = []
    routes = {
        "camera_findings": "camera_repair",
        "attachment_penetration": "trajectory_repair",
    }
    for field, route in routes.items():
        for item in telemetry.get(field, []) or []:
            if not isinstance(item, dict):
                continue
            failure_id = str(item.get("failure_id") or f"malformed_{field}")
            severity = str(item.get("severity") or "error")
            if severity not in {"info", "warning", "error", "hard"}:
                severity = "hard"
            findings.append(
                Finding(
                    failure_id=failure_id,
                    owner=str(item.get("owner") or "blender_executor"),
                    category=str(item.get("category") or "runtime_observability"),
                    severity=severity,
                    message=str(item.get("message") or f"runtime finding from {field}"),
                    root_cause_id=str(item.get("root_cause_id") or f"runtime:{field}:{failure_id}"),
                    evidence=[str(value) for value in item.get("evidence", []) or []],
                    repair_route=route,
                )
            )
    return findings


class DeterministicReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_version: str = "deterministic-v3-declarative-policy"
    terminal_status: Literal["pass", "fail"]
    hard_gate_failed: bool
    score: float = Field(ge=0, le=100)
    findings: list[Finding] = Field(default_factory=list)
    director_plan_score: float | None = Field(default=None, ge=0, le=100)
    director_findings: list[Finding] = Field(default_factory=list)
    interaction_findings: list[Finding] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)


class DeterministicEvaluator:
    def evaluate(
        self,
        contract: SceneContract,
        plan: TrajectoryPlan,
        *,
        execution: ExecutionResult | None = None,
        director_plan: DirectorPlan | None = None,
        telemetry: dict[str, Any] | None = None,
    ) -> DeterministicReport:
        findings: list[Finding] = []
        if execution is not None and execution.status != "success":
            findings.append(
                Finding(
                    failure_id="incomplete_proxy",
                    owner="blender_executor",
                    category="runtime",
                    severity="hard",
                    root_cause_id="runtime_execution",
                    message="proxy execution did not complete successfully",
                    repair_route="runtime_repair",
                )
            )

        findings.extend(check_event_order(contract, plan))
        findings.extend(check_required_event_coverage(contract, plan))
        findings.extend(check_camera_coverage(contract, plan))
        findings.extend(check_camera_motion_intent(contract, plan))
        findings.extend(check_trajectory_phase_alignment(contract, plan))
        findings.extend(check_support_before_grasp(contract, plan))
        findings.extend(check_attachment_contact(plan))
        findings.extend(check_velocity_continuity(plan))

        findings = deduplicate_findings(findings)
        hard_gate_failed = any(finding.severity == "hard" for finding in findings)
        score = self._score_findings(findings)
        director_score = None
        director_findings: list[Finding] = []
        if director_plan is not None:
            director_report = evaluate_director_plan(
                director_plan,
                plan,
                telemetry=telemetry,
            )
            director_findings = director_report.findings
            interaction_findings = deduplicate_findings(
                evaluate_interactions(director_plan, plan, telemetry=telemetry)
            )
            director_score = director_report.director_plan_score
        else:
            interaction_findings = []
        return DeterministicReport(
            terminal_status="fail" if hard_gate_failed else "pass",
            hard_gate_failed=hard_gate_failed,
            score=score,
            findings=findings,
            director_plan_score=director_score,
            director_findings=director_findings,
            interaction_findings=interaction_findings,
            metrics={
                "required_event_coverage": self._coverage(contract, plan),
                "finding_count": float(len(findings)),
                "hard_count": float(sum(finding.severity == "hard" for finding in findings)),
                "error_count": float(sum(finding.severity == "error" for finding in findings)),
                "warning_count": float(sum(finding.severity == "warning" for finding in findings)),
                "unique_root_cause_count": float(len(findings)),
            },
        )

    def evaluate_real(
        self,
        contract: SceneContract,
        plan: TrajectoryPlan,
        telemetry: dict[str, Any],
        artifacts: Any,
        director_plan: DirectorPlan | None = None,
    ) -> DeterministicReport:
        """Evaluate a real Blender run after the ordinary plan-level checks."""
        base = self.evaluate(contract, plan, director_plan=director_plan, telemetry=telemetry)
        findings = list(base.findings)
        findings.extend(_runtime_telemetry_findings(telemetry))
        if getattr(artifacts, "artifact_status", None) != "complete":
            failures = list(getattr(artifacts, "hard_failures", []))
            findings.append(
                Finding(
                    failure_id="incomplete_real_artifacts",
                    owner="proxy_renderer",
                    category="runtime",
                    severity="hard",
                    root_cause_id="runtime_artifact_completeness",
                    message=f"real proxy artifacts are incomplete: {failures}",
                    evidence=failures,
                    repair_route="runtime_repair",
                )
            )

        expected_entities = set(plan.entities)
        observed_entities = set(telemetry.get("objects", {}))
        missing_entities = sorted(expected_entities - observed_entities)
        if missing_entities:
            findings.append(
                Finding(
                    failure_id="telemetry_missing_entity",
                    owner="blender_executor",
                    category="telemetry",
                    severity="hard",
                    root_cause_id="runtime_entity_telemetry",
                    message=f"telemetry is missing planned entities: {missing_entities}",
                    evidence=missing_entities,
                    repair_route="runtime_repair",
                )
            )
        expected_kinds = {entity.id: entity.kind for entity in contract.entities}
        for entity_id, expected_kind in expected_kinds.items():
            observed_kind = telemetry.get("objects", {}).get(entity_id, {}).get("kind")
            # Older synthetic fixtures did not emit semantic kind metadata. Real
            # Blender telemetry does, so validate it whenever it is available.
            if observed_kind is not None and observed_kind != expected_kind:
                findings.append(
                    Finding(
                        failure_id="telemetry_entity_kind_mismatch",
                        owner="scene_parser",
                        category="scene_semantics",
                        severity="hard",
                        root_cause_id=f"scene_entity_kind:{entity_id}",
                        message=(
                            f"rendered entity {entity_id} has kind {observed_kind!r}, "
                            f"expected {expected_kind!r}"
                        ),
                        evidence=[entity_id, str(observed_kind), expected_kind],
                        repair_route="scene_contract_repair",
                    )
                )
        for field, expected in (
            ("frame_start", plan.timebase.frame_start),
            ("frame_end", plan.timebase.frame_end),
            ("fps", plan.timebase.fps),
        ):
            if telemetry.get(field) != expected:
                findings.append(
                    Finding(
                        failure_id="telemetry_timebase_mismatch",
                        owner="blender_executor",
                        category="telemetry",
                        severity="hard",
                        root_cause_id="runtime_timebase",
                        message=f"telemetry {field} does not match plan: {telemetry.get(field)} != {expected}",
                        evidence=[field],
                        repair_route="runtime_repair",
                    )
                )
        if telemetry.get("camera", {}).get("active") is not True:
            findings.append(
                Finding(
                    failure_id="telemetry_inactive_camera",
                    owner="camera_planner",
                    category="camera_coverage",
                    severity="hard",
                    root_cause_id="runtime_active_camera",
                    message="telemetry does not identify the planned camera as active",
                    repair_route="camera_repair",
                )
            )

        findings = deduplicate_findings(findings)
        hard_gate_failed = any(finding.severity == "hard" for finding in findings)
        return DeterministicReport(
            terminal_status="fail" if hard_gate_failed else "pass",
            hard_gate_failed=hard_gate_failed,
            score=self._score_findings(findings),
            director_plan_score=base.director_plan_score,
            director_findings=base.director_findings,
            interaction_findings=base.interaction_findings,
            findings=findings,
            metrics={
                "required_event_coverage": self._coverage(contract, plan),
                "real_readable_frames": float(getattr(artifacts, "readable_frame_count", 0)),
                "finding_count": float(len(findings)),
                "hard_count": float(sum(finding.severity == "hard" for finding in findings)),
                "error_count": float(sum(finding.severity == "error" for finding in findings)),
                "warning_count": float(sum(finding.severity == "warning" for finding in findings)),
                "unique_root_cause_count": float(len(findings)),
            },
        )

    @staticmethod
    def _coverage(contract: SceneContract, plan: TrajectoryPlan) -> float:
        required = set(contract.must_show)
        if not required:
            return 1.0
        covered = {
            event_id
            for shot in plan.camera.shots
            for event_id in shot.required_event_ids
        }
        return len(required & covered) / len(required)

    @staticmethod
    def _score_findings(findings: list[Finding]) -> float:
        return score_findings(findings)
