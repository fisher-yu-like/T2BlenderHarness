"""One visual-review interface shared by external VLM and local Codex review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .aggregate import aggregate_scores
from .deterministic import DeterministicReport
from .realism import score_realism
from .schemas import VLMJudgeResponse


REALISM_REVIEW_FIELDS = (
    "appearance_detail",
    "physical_realism",
    "spatial_consistency",
    "motion_naturalness",
    "visual_presentation",
)


def _review_payload(response: VLMJudgeResponse, source: str) -> dict[str, Any]:
    return {
        "status": "complete",
        "source": source,
        "confidence": response.confidence,
        "scores": {
            name: getattr(response, name)
            for name in REALISM_REVIEW_FIELDS
            if getattr(response, name) is not None
        },
    }


def score_shared_visual_review(
    run_dir: str | Path,
    *,
    deterministic: DeterministicReport,
    response: VLMJudgeResponse,
    source: str,
    frame_paths: list[str | Path],
    video_probe: dict[str, Any],
    model: str | None = None,
    raw_response_id: str | None = None,
    review_source_label: str | None = None,
) -> dict[str, Any]:
    """Score task and realism from one validated response, without adding them."""
    root = Path(run_dir)
    task_aggregate = aggregate_scores(deterministic, response)
    geometry_path = root / "geometry_report.json"
    visual_path = root / "visual_evidence.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.is_file() else {}
    visual = json.loads(visual_path.read_text(encoding="utf-8")) if visual_path.is_file() else {}
    realism = score_realism(geometry, visual, _review_payload(response, source))
    result = {
        "status": "scored",
        "review_source": review_source_label or source,
        "vlm_model": model,
        "video_probe": video_probe,
        "vlm_response": response.model_dump(mode="json"),
        "aggregate": task_aggregate.model_dump(mode="json"),
        "task_score": task_aggregate.final_score,
        "realism": realism,
        "realism_score": realism.get("score"),
        "realism_score_kind": realism.get("score_kind"),
        "frame_count": len(frame_paths),
        "sampled_frames": [str(Path(path).resolve()) for path in frame_paths],
        "raw_response_id": raw_response_id,
        "score_channels": {
            "task_score": "aggregate deterministic/VLM task score",
            "realism_score": "geometry/PNG plus the realism dimensions in this same review",
            "combined": False,
        },
    }
    (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
