"""Frozen scoring-v7 aggregation for independent visual review.

The protocol keeps task semantics, observability, and presentation explicit.
Applicability is supplied by the prompt/Director contract; non-applicable
dimensions are omitted from the geometric means instead of being filled with
an invented perfect score or a penalty.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import VLMJudgeResponse


SCORING_V7_VERSION = "scoring-v7-independent-channels"
REQUIRED_EVENT_SCORE_THRESHOLD = 25.0
REQUIRED_EVENT_TASK_CEILING = 49.0


class ScoringV7Result(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_version: str = SCORING_V7_VERSION
    status: Literal["scored", "uncertain", "failed_required_event"]
    semantic_core: float | None = Field(default=None, ge=0, le=100)
    choreography_score: float | None = Field(default=None, ge=0, le=100)
    observability_score: float | None = Field(default=None, ge=0, le=100)
    task_score: float | None = Field(default=None, ge=0, le=100)
    presentation_score: float | None = Field(default=None, ge=0, le=100)
    realism_vlm_score: float | None = Field(default=None, ge=0, le=100)
    camera_effectiveness: float | None = Field(default=None, ge=0, le=100)
    required_event_gate_failed: bool = False
    required_event_scores: dict[str, float | None] = Field(default_factory=dict)
    applicability: dict[str, bool] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)


def _value(response: VLMJudgeResponse, name: str) -> float | None:
    value = getattr(response, name, None)
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def _gm(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    if any(value <= 0 for value in present):
        return 0.0
    return math.prod(present) ** (1.0 / len(present))


def _mean(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


def score_v7(
    response: VLMJudgeResponse,
    *,
    applicability: Mapping[str, bool] | None = None,
    required_event_ids: Iterable[str] = (),
    required_event_scores: Mapping[str, float | None] | None = None,
) -> ScoringV7Result:
    """Score one blind review using the pre-calibration scoring-v7 policy.

    ``camera_motion`` controls whether ``camera_innovation`` is applicable.
    ``character_trajectory`` is normally false for object-only prompts.  A
    missing required-event score makes the result uncertain; it does not
    become zero.  A score below 25 is an explicit required-event failure and
    caps the task channel at 49.
    """

    active = {
        "prompt_compliance": True,
        "physical_plausibility": True,
        "object_trajectory": True,
        "character_trajectory": True,
        "event_timing": True,
        "camera_coverage": True,
        "camera_innovation": True,
        "camera_motion": True,
        "visual_clarity": True,
        "temporal_smoothness": True,
        "visual_presentation": True,
        "appearance_detail": True,
    }
    if applicability:
        active.update({str(key): bool(value) for key, value in applicability.items()})
    if not active.get("camera_motion", True):
        active["camera_innovation"] = False

    semantic_names = [
        name
        for name in ("prompt_compliance", "physical_plausibility", "object_trajectory", "character_trajectory", "event_timing")
        if active.get(name, True)
    ]
    semantic_core = _gm(_value(response, name) for name in semantic_names)
    camera_effectiveness = (
        _gm((_value(response, "camera_coverage"), _value(response, "camera_innovation")))
        if active.get("camera_motion", True) and active.get("camera_innovation", True)
        else _value(response, "camera_coverage")
    )
    choreography_values: list[float | None] = [
        _value(response, "camera_coverage"),
        _value(response, "camera_innovation")
        if active.get("camera_motion", True) and active.get("camera_innovation", True)
        else None,
        _value(response, "character_trajectory") if active.get("character_trajectory", True) else None,
        _value(response, "temporal_smoothness"),
    ]
    choreography = _gm(choreography_values)
    observability = _gm((_value(response, "camera_coverage"), _value(response, "visual_clarity")))
    task = None if semantic_core is None or observability is None else 0.75 * semantic_core + 0.25 * observability
    presentation = _mean(
        value
        for name, value in (
            ("temporal_smoothness", _value(response, "temporal_smoothness")),
            ("camera_effectiveness", camera_effectiveness),
            ("visual_presentation", _value(response, "visual_presentation")),
            ("appearance_detail", _value(response, "appearance_detail")),
        )
        if active.get(name, True)
    )
    realism = _gm(
        _value(response, name)
        for name in (
            "appearance_detail",
            "physical_realism",
            "spatial_consistency",
            "motion_naturalness",
            "visual_presentation",
        )
    )

    event_ids = [str(event_id) for event_id in required_event_ids if str(event_id).strip()]
    bound_event_scores = required_event_scores
    if bound_event_scores is None and getattr(response, "event_scores", None) is not None:
        bound_event_scores = response.event_scores
    event_scores = {
        event_id: (bound_event_scores.get(event_id) if bound_event_scores is not None else None)
        for event_id in event_ids
    }
    reasons: list[str] = []
    if event_ids and bound_event_scores is None:
        status = "uncertain"
        reasons.append("required_event_evidence_unavailable")
    elif any(value is not None and float(value) < REQUIRED_EVENT_SCORE_THRESHOLD for value in event_scores.values()):
        status = "failed_required_event"
        reasons.append("required_event_score_below_25")
    elif any(value is None for value in event_scores.values()):
        status = "uncertain"
        reasons.append("required_event_evidence_incomplete")
    else:
        status = "scored"

    required_event_gate_failed = status == "failed_required_event"
    if required_event_gate_failed:
        semantic_core = min(semantic_core, REQUIRED_EVENT_TASK_CEILING) if semantic_core is not None else None
        task = min(task, REQUIRED_EVENT_TASK_CEILING) if task is not None else None

    return ScoringV7Result(
        status=status,
        semantic_core=None if semantic_core is None else round(semantic_core, 4),
        choreography_score=None if choreography is None else round(choreography, 4),
        observability_score=None if observability is None else round(observability, 4),
        task_score=None if task is None else round(task, 4),
        presentation_score=None if presentation is None else round(presentation, 4),
        realism_vlm_score=None if realism is None else round(realism, 4),
        camera_effectiveness=None if camera_effectiveness is None else round(camera_effectiveness, 4),
        required_event_gate_failed=required_event_gate_failed,
        required_event_scores={key: None if value is None else float(value) for key, value in event_scores.items()},
        applicability={
            key: bool(active.get(key, True))
            for key in (
                "object_trajectory",
                "character_trajectory",
                "camera_coverage",
                "camera_innovation",
                "visual_clarity",
                "camera_motion",
            )
        },
        weights={
            "semantic_core": 0.75,
            "observability": 0.25,
            "presentation": 1.0,
            "required_event_ceiling": REQUIRED_EVENT_TASK_CEILING,
        },
        reasons=reasons,
    )


__all__ = [
    "REQUIRED_EVENT_SCORE_THRESHOLD",
    "REQUIRED_EVENT_TASK_CEILING",
    "SCORING_V7_VERSION",
    "ScoringV7Result",
    "score_v7",
]
