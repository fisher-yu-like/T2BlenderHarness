"""Visual-primary evaluator used by the new real training protocol."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import VLMJudgeResponse
from .scoring_v7 import SCORING_V7_VERSION, score_v7
from .evidence import build_dimension_evidence


VISUAL_PRIMARY_VERSION = SCORING_V7_VERSION
LEGACY_VISUAL_PRIMARY_VERSION = "visual-primary-v6-independent-channels"
VALID_REVIEW_SOURCES = {
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "human_review",
    "codex_local_visual_review",
    "codex-local-visual-review",
}


class VisualPrimaryScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_version: str = VISUAL_PRIMARY_VERSION
    status: Literal["scored", "skipped", "unavailable", "needs_human_review"]
    source: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    artifact_gate_pass: bool
    semantic_score: float | None = Field(default=None, ge=0, le=100)
    choreography_score: float | None = Field(default=None, ge=0, le=100)
    observability_score: float | None = Field(default=None, ge=0, le=100)
    presentation_score: float | None = Field(default=None, ge=0, le=100)
    camera_effectiveness: float | None = Field(default=None, ge=0, le=100)
    task_score: float | None = Field(default=None, ge=0, le=100)
    realism_score: float | None = Field(default=None, ge=0, le=100)
    overall_vlm_score: float | None = Field(default=None, ge=0, le=100)
    deterministic_score: float | None = None
    semantic_status: Literal["passed", "failed_required_event", "uncertain"] | None = None
    required_event_gate_failed: bool = False
    applicability: dict[str, bool] = Field(default_factory=dict)
    required_event_scores: dict[str, float | None] = Field(default_factory=dict)
    evidence_completeness: float = Field(default=0.0, ge=0, le=1)
    dimension_evidence: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if any(value <= 0 for value in values):
        return 0.0
    return math.prod(values) ** (1.0 / len(values))


DEFAULT_VISUAL_CONFIDENCE_THRESHOLD = 0.6


def score_visual_review(
    response: VLMJudgeResponse | None,
    *,
    artifact_gate_pass: bool,
    source: str,
    applicability: dict[str, bool] | None = None,
    required_event_ids: list[str] | tuple[str, ...] = (),
    required_event_scores: dict[str, float | None] | None = None,
    strict_evidence: bool = False,
    confidence_threshold: float = DEFAULT_VISUAL_CONFIDENCE_THRESHOLD,
    evidence_completeness_threshold: float = 1.0,
) -> VisualPrimaryScore:
    """Return independent task/realism channels with VLM as the main signal."""

    normalized_source = str(source).strip().lower()
    if not artifact_gate_pass:
        return VisualPrimaryScore(
            status="skipped",
            source=normalized_source,
            artifact_gate_pass=False,
            reason="artifact_gate_failed",
        )
    if normalized_source not in VALID_REVIEW_SOURCES:
        return VisualPrimaryScore(
            status="unavailable",
            source=normalized_source,
            artifact_gate_pass=True,
            reason="review_source_not_eligible",
        )
    if response is None:
        return VisualPrimaryScore(
            status="unavailable",
            source=normalized_source,
            artifact_gate_pass=True,
            reason="visual_review_missing",
        )
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be between zero and one")
    if not 0.0 <= float(evidence_completeness_threshold) <= 1.0:
        raise ValueError("evidence_completeness_threshold must be between zero and one")
    if response.confidence < float(confidence_threshold):
        return VisualPrimaryScore(
            status="needs_human_review",
            source=normalized_source,
            confidence=response.confidence,
            artifact_gate_pass=True,
            reason="low_visual_review_confidence",
        )

    evidence_report = build_dimension_evidence(
        response,
        applicability=applicability,
        completeness_threshold=evidence_completeness_threshold,
    )
    if strict_evidence and not evidence_report["complete"]:
        return VisualPrimaryScore(
            status="needs_human_review",
            source=normalized_source,
            confidence=response.confidence,
            artifact_gate_pass=True,
            evidence_completeness=evidence_report["mean_evidence_completeness"],
            dimension_evidence=evidence_report["dimensions"],
            reason="dimension_evidence_incomplete",
        )

    realism_values = [
        response.appearance_detail,
        response.physical_realism,
        response.spatial_consistency,
        response.motion_naturalness,
        response.visual_presentation,
    ]
    if any(value is None for value in realism_values):
        return VisualPrimaryScore(
            status="unavailable",
            source=normalized_source,
            confidence=response.confidence,
            artifact_gate_pass=True,
            reason="missing_realism_dimensions",
        )

    scoring = score_v7(
        response,
        applicability=applicability,
        required_event_ids=required_event_ids,
        required_event_scores=required_event_scores,
    )
    semantic = scoring.semantic_core
    task = scoring.task_score
    realism = scoring.realism_vlm_score
    if task is None or realism is None:
        return VisualPrimaryScore(
            status="unavailable",
            source=normalized_source,
            confidence=response.confidence,
            artifact_gate_pass=True,
            reason="scoring_v7_missing_applicable_dimensions",
        )
    overall = task * 0.70 + realism * 0.30
    return VisualPrimaryScore(
        status="scored",
        source=normalized_source,
        confidence=response.confidence,
        artifact_gate_pass=True,
        semantic_score=semantic,
        choreography_score=scoring.choreography_score,
        observability_score=scoring.observability_score,
        presentation_score=scoring.presentation_score,
        camera_effectiveness=scoring.camera_effectiveness,
        task_score=task,
        realism_score=realism,
        overall_vlm_score=round(overall, 4),
        deterministic_score=None,
        semantic_status=(
            "failed_required_event"
            if scoring.status == "failed_required_event"
            else "uncertain"
            if scoring.status == "uncertain"
            else "passed"
        ),
        required_event_gate_failed=scoring.required_event_gate_failed,
        applicability=scoring.applicability,
        required_event_scores=scoring.required_event_scores,
        evidence_completeness=evidence_report["mean_evidence_completeness"],
        dimension_evidence=evidence_report["dimensions"],
        weights={
            "semantic_core": 0.75,
            "observability": 0.25,
            "presentation": 1.0,
            "required_event_ceiling": 49.0,
            "legacy_overall_task": 0.70,
            "legacy_overall_realism": 0.30,
        },
    )


__all__ = [
    "LEGACY_VISUAL_PRIMARY_VERSION",
    "DEFAULT_VISUAL_CONFIDENCE_THRESHOLD",
    "VISUAL_PRIMARY_VERSION",
    "VisualPrimaryScore",
    "score_visual_review",
]
