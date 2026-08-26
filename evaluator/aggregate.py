"""Blend deterministic and VLM scores with hard-gate ceilings."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

from .deterministic import DeterministicReport
from .schemas import VLMJudgeResponse


class AggregateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_version: str = "aggregate-v2-visual-primary"
    deterministic_score: float = Field(ge=0, le=100)
    vlm_score: float = Field(ge=0, le=100)
    uncapped_score: float = Field(ge=0, le=100)
    final_score: float = Field(ge=0, le=100)
    cap_reason: str | None = None


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.prod(max(0.0, value) for value in values) ** (1.0 / len(values))


def grouped_vlm_score(response: VLMJudgeResponse) -> float:
    """Aggregate correlated dimensions without allowing one high score to hide a failure.

    The weights are an interpretable, pre-human-calibration baseline. Semantic
    fidelity and choreography are grouped separately; choreography includes
    camera and character/object trajectory dimensions explicitly.
    """
    semantic = _geometric_mean(
        [response.prompt_compliance, response.physical_plausibility, response.object_trajectory, response.event_timing]
    )
    choreography = _geometric_mean(
        [response.camera_coverage, response.camera_innovation, response.character_trajectory, response.temporal_smoothness]
    )
    presentation = response.visual_clarity
    return round(semantic * 0.45 + choreography * 0.45 + presentation * 0.10, 4)


def weighted_vlm_score(response: VLMJudgeResponse) -> float:
    """Backward-compatible name for the calibrated grouped aggregation."""
    return grouped_vlm_score(response)


def grouped_realism_vlm_score(response: VLMJudgeResponse) -> float | None:
    """Aggregate the independent visual-realism dimensions from one review."""
    values = [
        response.appearance_detail,
        response.physical_realism,
        response.spatial_consistency,
        response.motion_naturalness,
        response.visual_presentation,
    ]
    if any(value is None for value in values):
        return None
    return round(_geometric_mean([float(value) for value in values if value is not None]), 4)


def aggregate_scores(
    deterministic: DeterministicReport,
    vlm: VLMJudgeResponse,
    *,
    deterministic_weight: float = 0.2,
    vlm_weight: float = 0.8,
) -> AggregateScore:
    if abs(deterministic_weight + vlm_weight - 1.0) > 1e-9:
        raise ValueError("deterministic_weight and vlm_weight must sum to 1")
    vlm_score = weighted_vlm_score(vlm)
    uncapped = round(
        deterministic.score * deterministic_weight + vlm_score * vlm_weight,
        4,
    )
    cap_reason = "hard_gate" if deterministic.hard_gate_failed else None
    final_score = min(uncapped, 49.0) if cap_reason else uncapped
    return AggregateScore(
        deterministic_score=deterministic.score,
        vlm_score=vlm_score,
        uncapped_score=uncapped,
        final_score=final_score,
        cap_reason=cap_reason,
    )
