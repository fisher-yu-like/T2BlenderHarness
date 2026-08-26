"""Codex-local, human-in-the-loop review for real proxy video frames.

This module is deliberately not an automatic substitute for visual judgment. It
creates an auditable review request and accepts only an explicit review payload
written after a local frame inspection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evaluator.aggregate import aggregate_scores
from evaluator.deterministic import DeterministicReport
from evaluator.shared_review import score_shared_visual_review
from evaluator.schemas import VLMJudgeResponse
from videoact.real_artifacts import probe_mp4


ASSISTANT_REVIEW_VERSION = "assistant-local-v1"
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_assistant_review_request(
    run_dir: str | Path,
    *,
    prompt: str,
    scene_contract: Any,
    deterministic_findings: list[Any],
    frame_paths: list[str | Path],
    video_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    if hasattr(scene_contract, "model_dump"):
        scene_contract = scene_contract.model_dump(mode="json")
    findings = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in deterministic_findings
    ]
    return {
        "review_version": ASSISTANT_REVIEW_VERSION,
        "review_source": "assistant_local_review",
        "reviewer": "codex-assistant",
        "case_id": _manifest_case_id(root),
        "prompt": prompt,
        "scene_contract": scene_contract,
        "deterministic_findings": findings,
        "video_probe": video_probe or probe_mp4(root / "proxy.mp4", minimum_frames=3),
        "sampled_frames": [str(Path(path).resolve()) for path in frame_paths],
        "dimensions": list(REVIEW_DIMENSIONS),
        "instructions": {
            "frame_order": "chronological",
            "evidence_policy": "score only what is visible in the sampled frames and contract; missing evidence lowers the dimension",
            "score_range": "0-100 integer for every dimension",
            "required_output": "scores, visible_evidence, weaknesses, confidence",
            "trajectory_focus": "check camera follow/orbit/dolly/reveal and character/object phase transitions, not only final composition",
        },
    }


def _manifest_case_id(root: Path) -> str | None:
    path = root / "run_manifest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("case_id")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def write_assistant_review_request(run_dir: str | Path, request: dict[str, Any]) -> Path:
    root = Path(run_dir)
    path = root / "assistant_review_request.json"
    _write_json(path, request)
    return path


def _validated_review(payload: dict[str, Any], expected_frames: list[str | Path]) -> VLMJudgeResponse:
    if payload.get("review_version") != ASSISTANT_REVIEW_VERSION:
        raise ValueError("invalid assistant review version")
    if payload.get("reviewer") != "codex-assistant":
        raise ValueError("assistant review must identify codex-assistant")
    requested_frames = [str(Path(path).resolve()) for path in payload.get("sampled_frames", [])]
    actual_frames = [str(Path(path).resolve()) for path in expected_frames]
    if requested_frames != actual_frames:
        raise ValueError("assistant review frames do not match the evaluator sampling")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError("assistant review must contain a scores object")
    if set(REVIEW_DIMENSIONS) - set(scores):
        missing = sorted(set(REVIEW_DIMENSIONS) - set(scores))
        raise ValueError(f"assistant review is missing dimensions: {missing}")
    response = VLMJudgeResponse.model_validate(scores)
    if not response.visible_evidence:
        raise ValueError("assistant review requires visible_evidence")
    if not 0.0 <= response.confidence <= 1.0:
        raise ValueError("assistant review confidence must be between 0 and 1")
    return response


def score_assistant_local_review(
    run_dir: str | Path,
    *,
    deterministic: DeterministicReport,
    frame_paths: list[str | Path],
    review: dict[str, Any],
    video_probe: dict[str, Any],
) -> dict[str, Any]:
    response = _validated_review(review, frame_paths)
    if response.confidence < 0.6:
        result = {
            "status": "needs_human_review",
            "review_source": "assistant_local_review",
            "reason": "low_visual_review_confidence",
            "confidence": response.confidence,
            "video_probe": video_probe,
            "sampled_frames": [str(Path(path).resolve()) for path in frame_paths],
            "vlm_response": response.model_dump(mode="json"),
        }
        _write_json(Path(run_dir) / "vlm_report.json", result)
        return result
    result = score_shared_visual_review(
        run_dir,
        deterministic=deterministic,
        response=response,
        source="assistant_local_review",
        frame_paths=frame_paths,
        video_probe=video_probe,
        model=None,
    )
    result["vlm_model_alias"] = None
    result["review_confidence"] = response.confidence
    _write_json(Path(run_dir) / "vlm_report.json", result)
    return result
