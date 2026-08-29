"""Evaluate deterministic-pass real proxy runs with an optional VLM provider."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator.aggregate import aggregate_scores  # noqa: E402
from evaluator.assistant_local import (  # noqa: E402
    build_assistant_review_request,
    score_assistant_local_review,
    write_assistant_review_request,
)
from evaluator.deterministic import DeterministicReport  # noqa: E402
from evaluator.openai_vlm import OpenAIVLMProvider, VLMUnavailable, canonical_vlm_name  # noqa: E402
from evaluator.vlm_providers import OpenAICompatibleVLMProvider  # noqa: E402
from evaluator.shared_review import score_shared_visual_review  # noqa: E402
from evaluator.realism import score_realism  # noqa: E402
from evaluator.real_video_metrics import (  # noqa: E402
    LOCAL_REVIEW_SOURCE,
    build_local_vlm_response,
    evaluate_real_video,
)
from evaluator.schemas import VLMJudgeResponse  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION, score_visual_review  # noqa: E402
from videoact.real_artifacts import probe_mp4, sample_event_aligned_frame_paths  # noqa: E402
from scripts.evaluate_real_runs import discover_run_dirs  # noqa: E402


def evaluate_vlm_run(
    run_dir: str | Path,
    *,
    prompt: str,
    scene_contract: Any,
    provider: Any | None = None,
    assistant_local: bool = False,
    assistant_review: dict[str, Any] | None = None,
    scoring_policy: str = VISUAL_PRIMARY_VERSION,
) -> dict[str, Any]:
    root = Path(run_dir)
    use_visual_primary = scoring_policy in {VISUAL_PRIMARY_VERSION, "visual-primary-v6"}
    deterministic = DeterministicReport.model_validate(
        json.loads((root / "deterministic_report.json").read_text(encoding="utf-8"))
    )
    if deterministic.terminal_status != "pass" or deterministic.hard_gate_failed:
        result = {"status": "skipped", "reason": "deterministic_gate_failed"}
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    geometry_report_path = root / "geometry_report.json"
    if geometry_report_path.is_file():
        geometry_report = json.loads(geometry_report_path.read_text(encoding="utf-8"))
        if geometry_report.get("hard_gate_failed"):
            result = {
                "status": "skipped",
                "reason": "realism_geometry_gate_failed",
                "geometry_findings": [finding.get("failure_id") for finding in geometry_report.get("findings", [])],
            }
            (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
            return result
    video_probe = probe_mp4(root / "proxy.mp4", minimum_frames=3)
    if not video_probe["playable"]:
        result = {
            "status": "skipped",
            "reason": "unplayable_proxy_video",
            "video_probe": video_probe,
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    frames = sample_event_aligned_frame_paths(root, scene_contract, max_frames=8)
    if not frames:
        result = {"status": "skipped", "reason": "no_sample_frames"}
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    if assistant_review is not None and assistant_review.get("review_source") == "frame_statistics":
        result = {
            "status": "unavailable",
            "review_source": "frame_statistics",
            "reason": "frame_statistics_not_eligible",
            "video_probe": video_probe,
            "frame_count": len(frames),
            "sampled_frames": [str(path.resolve()) for path in frames],
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    if assistant_review is not None:
        if use_visual_primary:
            try:
                review_payload = dict(assistant_review.get("scores") or assistant_review)
                response = VLMJudgeResponse.model_validate(review_payload)
            except (TypeError, ValueError) as exc:
                result = {
                    "status": "unavailable",
                    "review_source": "human_review",
                    "reason": f"human_review_schema_error:{type(exc).__name__}",
                    "video_probe": video_probe,
                    "sampled_frames": [str(path.resolve()) for path in frames],
                }
                (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
                return result
            return _score_visual_primary(
                root,
                deterministic=deterministic,
                response=response,
                source=str(assistant_review.get("review_source") or "human_review").lower(),
                review_source_label=str(assistant_review.get("review_source") or "human_review").lower(),
                frame_paths=frames,
                video_probe=video_probe,
                raw_response_id=None,
            )
        return score_assistant_local_review(
            root,
            deterministic=deterministic,
            frame_paths=frames,
            review=assistant_review,
            video_probe=video_probe,
        )
    if assistant_local:
        trajectory_path = root / "trajectory.json"
        telemetry_path = root / "telemetry.json"
        if trajectory_path.is_file() and telemetry_path.is_file():
            try:
                trajectory_plan = json.loads(trajectory_path.read_text(encoding="utf-8"))
                telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
                local_evidence = evaluate_real_video(
                    root,
                    prompt=prompt,
                    scene_contract=scene_contract,
                    trajectory_plan=trajectory_plan,
                    telemetry=telemetry,
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                local_evidence = None
            if local_evidence and local_evidence.get("status") == "scored":
                local_response = build_local_vlm_response(local_evidence)
                return _score_visual_primary(
                    root,
                    deterministic=deterministic,
                    response=local_response,
                    source=LOCAL_REVIEW_SOURCE,
                    review_source_label=LOCAL_REVIEW_SOURCE,
                    frame_paths=frames,
                    video_probe=video_probe,
                    model="codex-local",
                    local_video_evidence=local_evidence,
                )
        request = build_assistant_review_request(
            root,
            prompt=prompt,
            scene_contract=scene_contract,
            deterministic_findings=deterministic.findings,
            frame_paths=frames,
            video_probe=video_probe,
        )
        request_path = write_assistant_review_request(root, request)
        result = {
            "status": "awaiting_assistant_review",
            "review_source": "assistant_local_review",
            "reason": "assistant_local_review_required",
            "review_request": str(request_path.resolve()),
            "video_probe": video_probe,
            "frame_count": len(frames),
            "sampled_frames": [str(path.resolve()) for path in frames],
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    provider = provider or OpenAICompatibleVLMProvider()
    try:
        vlm_response, raw_response = provider.evaluate(
            prompt=prompt,
            scene_contract=scene_contract,
            frame_paths=frames,
            deterministic_findings=deterministic.findings,
        )
    except VLMUnavailable as exc:
        result = {"status": "unavailable", "reason": str(exc)}
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    if vlm_response.confidence < 0.6:
        result = {
            "status": "needs_human_review",
            "review_source": "external_vlm",
            "reason": "low_visual_review_confidence",
            "confidence": vlm_response.confidence,
            "video_probe": video_probe,
            "sampled_frames": [str(path.resolve()) for path in frames],
            "vlm_response": vlm_response.model_dump(mode="json"),
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    review_source = getattr(provider, "model_alias", None) or canonical_vlm_name(
        getattr(provider, "model", None) or "gpt-5.6-luna"
    )
    if use_visual_primary:
        return _score_visual_primary(
            root,
            deterministic=deterministic,
            response=vlm_response,
            source=str(review_source).lower(),
            review_source_label="external_vlm",
            model=str(review_source).lower(),
            frame_paths=frames,
            video_probe=video_probe,
            raw_response_id=raw_response.get("id"),
        )
    result = score_shared_visual_review(
        root,
        deterministic=deterministic,
        response=vlm_response,
        source=getattr(provider, "model_alias", None) or canonical_vlm_name(getattr(provider, "model", None) or "gpt-5.6-luna"),
        review_source_label="external_vlm",
        frame_paths=frames,
        video_probe=video_probe,
        model=getattr(provider, "model_alias", None) or getattr(provider, "model", None),
        raw_response_id=raw_response.get("id"),
    )
    result["vlm_model_alias"] = getattr(provider, "model_alias", None)
    (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _score_visual_primary(
    run_dir: str | Path,
    *,
    deterministic: DeterministicReport,
    response: VLMJudgeResponse,
    source: str,
    review_source_label: str,
    frame_paths: list[str | Path],
    video_probe: dict[str, Any],
    model: str | None = None,
    raw_response_id: str | None = None,
    local_video_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist the new evaluator channels without legacy deterministic fusion."""

    root = Path(run_dir)
    if local_video_evidence is not None:
        (root / "local_video_evidence.json").write_text(
            json.dumps(local_video_evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    primary = score_visual_review(response, artifact_gate_pass=True, source=source)
    if primary.status != "scored":
        result = {
            "status": primary.status,
            "review_source": review_source_label,
            "vlm_model": model,
            "reason": primary.reason,
            "confidence": primary.confidence,
            "video_probe": video_probe,
            "visual_primary": primary.model_dump(mode="json"),
            "sampled_frames": [str(Path(path).resolve()) for path in frame_paths],
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    geometry_path = root / "geometry_report.json"
    visual_path = root / "visual_evidence.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8")) if geometry_path.is_file() else {}
    visual_evidence = json.loads(visual_path.read_text(encoding="utf-8")) if visual_path.is_file() else {}
    independent_review = {
        "status": "complete",
        "source": source,
        "confidence": response.confidence,
        "scores": {
            name: getattr(response, name)
            for name in (
                "appearance_detail",
                "physical_realism",
                "spatial_consistency",
                "motion_naturalness",
                "visual_presentation",
            )
        },
    }
    realism = score_realism(geometry, visual_evidence, independent_review)
    result = {
        "status": "scored",
        "evaluator_version": VISUAL_PRIMARY_VERSION,
        "review_source": review_source_label,
        "vlm_model": model,
        "confidence": response.confidence,
        "video_probe": video_probe,
        "vlm_response": response.model_dump(mode="json"),
        "visual_primary": primary.model_dump(mode="json"),
        "task_score": primary.task_score,
        "task_final_score": primary.task_score,
        "realism_vlm_score": primary.realism_score,
        "realism": realism,
        "realism_score": realism.get("score"),
        "realism_score_kind": realism.get("score_kind"),
        "overall_vlm_score": primary.overall_vlm_score,
        "deterministic_score": deterministic.score,
        "frame_count": len(frame_paths),
        "sampled_frames": [str(Path(path).resolve()) for path in frame_paths],
        "raw_response_id": raw_response_id,
        "local_video_evidence": local_video_evidence,
        "score_channels": {
            "task_score": "visual-primary VLM task channel",
            "realism_score": "independent realism channel with VLM as primary evidence",
            "deterministic_score": "artifact/contract gate and diagnostics only",
            "combined": False,
        },
    }
    (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def evaluate_split(
    run_root: str | Path,
    dataset_root: str | Path = "dataset/trajectory-v5-agent-codegen",
    *,
    assistant_local: bool = False,
    assistant_review_dir: str | Path | None = None,
    scoring_policy: str = VISUAL_PRIMARY_VERSION,
) -> list[dict[str, Any]]:
    root = Path(run_root)
    records = [
        json.loads(line)
        for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["case_id"]: record for record in records}
    results = []
    for run_dir in discover_run_dirs(root):
        case_id = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["case_id"]
        record = by_id[case_id]
        review = None
        if assistant_review_dir is not None:
            review_path = Path(assistant_review_dir) / f"{case_id}.json"
            if review_path.is_file():
                review = json.loads(review_path.read_text(encoding="utf-8"))
        result = evaluate_vlm_run(
            run_dir,
            prompt=record["prompt"],
            scene_contract=json.loads((run_dir / "scene_contract.json").read_text(encoding="utf-8")),
            assistant_local=assistant_local,
            assistant_review=review,
            scoring_policy=scoring_policy,
        )
        results.append({"case_id": case_id, **result})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--assistant-local", action="store_true")
    parser.add_argument("--assistant-review-dir")
    parser.add_argument("--scoring-policy", default=VISUAL_PRIMARY_VERSION, choices=["legacy-aggregate", "visual-primary-v6", VISUAL_PRIMARY_VERSION])
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_split(
                args.run_root,
                args.dataset_root,
                assistant_local=args.assistant_local,
                assistant_review_dir=args.assistant_review_dir,
                scoring_policy=args.scoring_policy,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
