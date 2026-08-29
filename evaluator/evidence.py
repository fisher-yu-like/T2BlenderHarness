"""Per-dimension evidence and uncertainty records for visual review."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schemas import VLMJudgeResponse


EVIDENCE_CONTRACT_VERSION = "dimension-evidence-v1"
DEFAULT_EVIDENCE_COMPLETENESS_THRESHOLD = 1.0
REVIEW_DIMENSIONS = (
    "prompt_compliance",
    "physical_plausibility",
    "camera_coverage",
    "camera_innovation",
    "character_trajectory",
    "object_trajectory",
    "event_timing",
    "temporal_smoothness",
    "visual_clarity",
    "appearance_detail",
    "physical_realism",
    "spatial_consistency",
    "motion_naturalness",
    "visual_presentation",
)


def _bounded(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _dimension_item(response: VLMJudgeResponse, name: str, *, applicable: bool = True) -> dict[str, Any]:
    score = getattr(response, name, None)
    score = None if score is None else max(0.0, min(100.0, float(score)))
    raw = (response.dimension_evidence or {}).get(name) if response.dimension_evidence else None
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    raw = raw if isinstance(raw, Mapping) else {}
    confidence = _bounded(raw.get("confidence"), 0.0)
    completeness = _bounded(raw.get("evidence_completeness"), 0.0)
    refs = raw.get("evidence_refs")
    refs = [str(item) for item in refs if str(item).strip()] if isinstance(refs, list) else []
    # The interval is an audit aid, not a replacement score.  It widens when
    # either confidence or concrete frame coverage is missing.
    half_width = 5.0 + 25.0 * (1.0 - confidence * completeness)
    interval = None if score is None else [
        round(max(0.0, score - half_width), 4),
        round(min(100.0, score + half_width), 4),
    ]
    return {
        "score": None if score is None else round(score, 4),
        "applicability": bool(applicable),
        "confidence": round(confidence, 4),
        "evidence_completeness": round(completeness, 4),
        "evidence_refs": refs,
        "interval": interval,
    }


def build_dimension_evidence(
    response: VLMJudgeResponse,
    *,
    applicability: Mapping[str, bool] | None = None,
    completeness_threshold: float = DEFAULT_EVIDENCE_COMPLETENESS_THRESHOLD,
) -> dict[str, Any]:
    if not 0.0 <= float(completeness_threshold) <= 1.0:
        raise ValueError("completeness_threshold must be between zero and one")
    active = {name: True for name in REVIEW_DIMENSIONS}
    if applicability:
        active.update({str(name): bool(value) for name, value in applicability.items() if str(name) in active})
    if applicability is not None and not bool(applicability.get("camera_motion", True)):
        active["camera_innovation"] = False
    dimensions = {
        name: _dimension_item(response, name, applicable=active[name])
        for name in REVIEW_DIMENSIONS
    }
    values = [item for name, item in dimensions.items() if active[name]]
    complete = all(
        item["score"] is not None
        and item["evidence_completeness"] >= float(completeness_threshold)
        and bool(item["evidence_refs"])
        for item in values
    )
    return {
        "contract_version": EVIDENCE_CONTRACT_VERSION,
        "complete": complete,
        "applicability": active,
        "mean_confidence": round(sum(item["confidence"] for item in values) / len(values), 4) if values else 0.0,
        "mean_evidence_completeness": round(
            sum(item["evidence_completeness"] for item in values) / len(values), 4
        ) if values else 0.0,
        "completeness_threshold": float(completeness_threshold),
        "dimensions": dimensions,
    }


__all__ = [
    "DEFAULT_EVIDENCE_COMPLETENESS_THRESHOLD",
    "EVIDENCE_CONTRACT_VERSION",
    "REVIEW_DIMENSIONS",
    "build_dimension_evidence",
]
