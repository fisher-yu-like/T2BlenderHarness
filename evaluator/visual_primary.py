"""Visual-primary evaluator used by the new real training protocol."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import VLMJudgeResponse


VISUAL_PRIMARY_VERSION = "visual-primary-v6-independent-channels"
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
    task_score: float | None = Field(default=None, ge=0, le=100)
    realism_score: float | None = Field(default=None, ge=0, le=100)
    overall_vlm_score: float | None = Field(default=None, ge=0, le=100)
    deterministic_score: float | None = None
    reason: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if any(value <= 0 for value in values):
        return 0.0
    return math.prod(values) ** (1.0 / len(values))


def score_visual_review(
    response: VLMJudgeResponse | None,
    *,
    artifact_gate_pass: bool,
    source: str,
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
    if response.confidence < 0.6:
        return VisualPrimaryScore(
            status="needs_human_review",
            source=normalized_source,
            confidence=response.confidence,
            artifact_gate_pass=True,
            reason="low_visual_review_confidence",
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

    semantic = _geometric_mean(
        [
            response.prompt_compliance,
            response.physical_plausibility,
            response.object_trajectory,
            response.event_timing,
        ]
    )
    choreography = _geometric_mean(
        [
            response.camera_coverage,
            response.camera_innovation,
            response.character_trajectory,
            response.temporal_smoothness,
        ]
    )
    task = semantic * 0.45 + choreography * 0.45 + response.visual_clarity * 0.10
    realism = _geometric_mean([float(value) for value in realism_values if value is not None])
    overall = task * 0.70 + realism * 0.30
    return VisualPrimaryScore(
        status="scored",
        source=normalized_source,
        confidence=response.confidence,
        artifact_gate_pass=True,
        semantic_score=round(semantic, 4),
        choreography_score=round(choreography, 4),
        task_score=round(task, 4),
        realism_score=round(realism, 4),
        overall_vlm_score=round(overall, 4),
        deterministic_score=None,
        weights={"semantic": 0.45, "choreography": 0.45, "visual_clarity": 0.10, "task": 0.70, "realism": 0.30},
    )


__all__ = ["VISUAL_PRIMARY_VERSION", "VisualPrimaryScore", "score_visual_review"]
