"""Three-layer result contract for real-video evaluation.

Execution validity, semantic completion, and presentation quality are kept as
separate states.  This module is intentionally conservative: an unavailable
observation never becomes a zero, and a failed required event never makes a
valid-looking presentation score disappear.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


RESULT_CONTRACT_VERSION = "evaluation-result-v1"
REQUIRED_EVENT_SCORE_THRESHOLD = 25.0
REQUIRED_EVENT_TASK_CEILING = 49.0


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = RESULT_CONTRACT_VERSION
    execution_status: Literal["valid", "invalid", "unavailable"]
    semantic_status: Literal["passed", "failed_required_event", "uncertain"]
    quality_status: Literal["scored", "unavailable"]
    semantic_score: float | None = Field(default=None, ge=0, le=100)
    observability_score: float | None = Field(default=None, ge=0, le=100)
    presentation_score: float | None = Field(default=None, ge=0, le=100)
    task_score: float | None = Field(default=None, ge=0, le=100)
    realism_score: float | None = Field(default=None, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence_completeness: float = Field(ge=0, le=1)
    reasons: list[str] = Field(default_factory=list)


def _score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return None


def _execution_status(
    *, artifact_status: str | None, video_probe: Mapping[str, Any] | None, runtime_observation_count: int | None
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if str(artifact_status or "").lower() == "unavailable":
        return "unavailable", ["artifact_evidence_unavailable"]
    if str(artifact_status or "").lower() != "complete":
        reasons.append("artifact_gate_incomplete")
    if not isinstance(video_probe, Mapping) or video_probe.get("playable") is not True:
        reasons.append("proxy_mp4_not_playable")
    if runtime_observation_count is None or int(runtime_observation_count) <= 0:
        reasons.append("runtime_observations_missing")
    if reasons:
        return "invalid", reasons
    return "valid", reasons


def build_evaluation_result(
    *,
    artifact_status: str | None,
    video_probe: Mapping[str, Any] | None,
    runtime_observation_count: int | None,
    visual_status: str,
    semantic_score: float | None,
    observability_score: float | None,
    presentation_score: float | None,
    task_score: float | None,
    realism_score: float | None,
    required_event_scores: Mapping[str, float | None] | None = None,
    confidence: float | None = None,
) -> EvaluationResult:
    """Build a result without collapsing missing evidence into a numeric score.

    ``required_event_scores`` is an evidence-bound mapping.  ``None`` means
    the event evidence was not available; an empty mapping means the contract
    has no required events.  A score below 25 is a failed required event and
    caps the task channel at 49 while leaving a valid presentation channel
    available.
    """

    execution_status, reasons = _execution_status(
        artifact_status=artifact_status,
        video_probe=video_probe,
        runtime_observation_count=runtime_observation_count,
    )
    raw_semantic = _score(semantic_score)
    raw_observability = _score(observability_score)
    raw_presentation = _score(presentation_score)
    raw_task = _score(task_score)
    raw_realism = _score(realism_score)

    if execution_status != "valid":
        return EvaluationResult(
            execution_status=execution_status,  # type: ignore[arg-type]
            semantic_status="uncertain",
            quality_status="unavailable",
            confidence=None,
            evidence_completeness=0.0,
            reasons=reasons,
        )

    if required_event_scores is None:
        semantic_status = "uncertain"
        reasons.append("required_event_evidence_unavailable")
    elif any(value is not None and float(value) < REQUIRED_EVENT_SCORE_THRESHOLD for value in required_event_scores.values()):
        semantic_status = "failed_required_event"
        reasons.append("required_event_evidence_below_threshold")
    elif any(value is None for value in required_event_scores.values()):
        semantic_status = "uncertain"
        reasons.append("required_event_evidence_incomplete")
    elif str(visual_status).lower() in {"unavailable", "needs_human_review"}:
        semantic_status = "uncertain"
        reasons.append("visual_review_not_final")
    else:
        semantic_status = "passed"

    quality_status = "scored" if str(visual_status).lower() == "scored" and raw_presentation is not None else "unavailable"
    if quality_status != "scored":
        raw_presentation = None
        raw_realism = None

    if semantic_status == "failed_required_event":
        raw_semantic = min(raw_semantic, REQUIRED_EVENT_TASK_CEILING) if raw_semantic is not None else None
        raw_task = min(raw_task, REQUIRED_EVENT_TASK_CEILING) if raw_task is not None else None

    observed_fields = sum(value is not None for value in (raw_semantic, raw_observability, raw_presentation, raw_task, raw_realism))
    completeness = observed_fields / 5.0
    return EvaluationResult(
        execution_status="valid",
        semantic_status=semantic_status,  # type: ignore[arg-type]
        quality_status=quality_status,  # type: ignore[arg-type]
        semantic_score=raw_semantic,
        observability_score=raw_observability,
        presentation_score=raw_presentation,
        task_score=raw_task,
        realism_score=raw_realism,
        confidence=_score(confidence) / 100.0 if confidence is not None and float(confidence) > 1.0 else confidence,
        evidence_completeness=round(completeness, 4),
        reasons=reasons,
    )


__all__ = [
    "EvaluationResult",
    "REQUIRED_EVENT_SCORE_THRESHOLD",
    "REQUIRED_EVENT_TASK_CEILING",
    "RESULT_CONTRACT_VERSION",
    "build_evaluation_result",
]
