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
from evaluator.shared_review import score_shared_visual_review  # noqa: E402
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
) -> dict[str, Any]:
    root = Path(run_dir)
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
    if assistant_review is not None:
        return score_assistant_local_review(
            root,
            deterministic=deterministic,
            frame_paths=frames,
            review=assistant_review,
            video_probe=video_probe,
        )
    if assistant_local:
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
    provider = provider or OpenAIVLMProvider()
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


def evaluate_split(
    run_root: str | Path,
    dataset_root: str | Path = "dataset",
    *,
    assistant_local: bool = False,
    assistant_review_dir: str | Path | None = None,
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
        )
        results.append({"case_id": case_id, **result})
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--assistant-local", action="store_true")
    parser.add_argument("--assistant-review-dir")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_split(
                args.run_root,
                args.dataset_root,
                assistant_local=args.assistant_local,
                assistant_review_dir=args.assistant_review_dir,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
