"""Evaluate deterministic-pass real proxy runs with an optional VLM provider."""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    evaluate_real_video,
)
from evaluator.result_contract import build_evaluation_result  # noqa: E402
from evaluator.scoring_v7 import score_v7  # noqa: E402
from evaluator.schemas import VLMJudgeResponse  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION, score_visual_review  # noqa: E402
from videoact.real_artifacts import probe_mp4, sample_event_aligned_frame_paths  # noqa: E402
from videoact.observer_contract import read_trusted_observer_output  # noqa: E402
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
    visual_frame_budget: int | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    use_visual_primary = scoring_policy in {VISUAL_PRIMARY_VERSION, "visual-primary-v6"}
    deterministic_report_path = root / "deterministic_report.json"
    if not deterministic_report_path.is_file():
        # Failed preparation/render cases can still have a run manifest and
        # therefore be discovered by evaluate_split.  They have no visual
        # evidence to review; preserve that boundary as an explicit
        # unavailable result instead of inventing a report or raising while
        # trying to read one.
        result = {
            "status": "unavailable",
            "review_source": "not_evaluated",
            "reason": "deterministic_report_missing",
        }
        (root / "vlm_report.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        return result
    deterministic = DeterministicReport.model_validate(
        json.loads(deterministic_report_path.read_text(encoding="utf-8"))
    )
    # The explicit hard-gate bit is authoritative.  A report may carry a
    # non-pass terminal label for a soft/diagnostic finding while still being
    # eligible for visual evidence collection; do not suppress the Codex VLM
    # solely because those two fields are inconsistent.
    if deterministic.hard_gate_failed:
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
    frame_budget = visual_frame_budget
    if frame_budget is None and provider is not None:
        frame_budget = getattr(provider, "visual_frame_budget", None)
    frame_budget = int(frame_budget or 8)
    if frame_budget < 1:
        raise ValueError("visual_frame_budget must be a positive integer")
    frames = sample_event_aligned_frame_paths(root, scene_contract, max_frames=frame_budget)
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
                scene_contract=scene_contract,
                frame_paths=frames,
                video_probe=video_probe,
                raw_response_id=None,
                strict_evidence=use_visual_primary and scoring_policy == VISUAL_PRIMARY_VERSION,
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
        runtime_observations = _runtime_observations_for_evaluation(root)
        if trajectory_path.is_file() and runtime_observations is not None:
            try:
                trajectory_plan = json.loads(trajectory_path.read_text(encoding="utf-8"))
                local_evidence = evaluate_real_video(
                    root,
                    prompt=prompt,
                    scene_contract=scene_contract,
                    trajectory_plan=trajectory_plan,
                    telemetry={"runtime_observations": runtime_observations},
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                local_evidence = None
            if local_evidence and local_evidence.get("status") == "scored":
                # These measurements are useful diagnostics, but they are not
                # an independent visual judge. Never convert them into a
                # VLMJudgeResponse or a formal task score.
                (root / "deterministic_video_proxy_metrics.json").write_text(
                    json.dumps(local_evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                result = {
                    "status": "unavailable",
                    "review_source": LOCAL_REVIEW_SOURCE,
                    "reason": "deterministic_video_proxy_metrics_not_a_visual_judge",
                    "video_probe": video_probe,
                    "frame_count": len(frames),
                    "sampled_frames": [str(path.resolve()) for path in frames],
                    "deterministic_video_proxy_metrics": local_evidence,
                    "task_score": None,
                    "realism_score": None,
                }
                (root / "vlm_report.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return result
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
            frame_paths=frames,
            video_path=root / "proxy.mp4",
            frame_metadata=_blind_frame_metadata(frames, video_probe),
        )
    except VLMUnavailable as exc:
        result = {
            "status": "unavailable",
            "review_source": getattr(provider, "review_source", "external_vlm"),
            "provider_kind": getattr(provider, "provider_kind", None),
            "reason": str(exc),
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    review_source = getattr(provider, "model_alias", None) or canonical_vlm_name(
        getattr(provider, "model", None) or "gpt-5.6-luna"
    )
    review_source_label = getattr(provider, "review_source", "external_vlm")
    if vlm_response.confidence < 0.6:
        result = {
            "status": "needs_human_review",
            "review_source": review_source_label,
            "provider_kind": getattr(provider, "provider_kind", None),
            "reason": "low_visual_review_confidence",
            "confidence": vlm_response.confidence,
            "video_probe": video_probe,
            "sampled_frames": [str(path.resolve()) for path in frames],
            "vlm_response": vlm_response.model_dump(mode="json"),
        }
        (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    if use_visual_primary:
        return _score_visual_primary(
            root,
            deterministic=deterministic,
            response=vlm_response,
            source=str(review_source).lower(),
            review_source_label=review_source_label,
            model=str(review_source).lower(),
            scene_contract=scene_contract,
            frame_paths=frames,
            video_probe=video_probe,
            raw_response_id=raw_response.get("id"),
            strict_evidence=use_visual_primary and scoring_policy == VISUAL_PRIMARY_VERSION,
        )
    result = score_shared_visual_review(
        root,
        deterministic=deterministic,
        response=vlm_response,
        source=getattr(provider, "model_alias", None) or canonical_vlm_name(getattr(provider, "model", None) or "gpt-5.6-luna"),
        review_source_label=review_source_label,
        frame_paths=frames,
        video_probe=video_probe,
        model=getattr(provider, "model_alias", None) or getattr(provider, "model", None),
        raw_response_id=raw_response.get("id"),
    )
    result["vlm_model_alias"] = getattr(provider, "model_alias", None)
    (root / "vlm_report.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _runtime_observations_for_evaluation(root: Path) -> list[Any] | None:
    """Return observations only from the trusted observer on formal runs.

    Legacy diagnostic fixtures without ``trusted_observer_required`` retain
    their historical local-metric behavior.  A formal run is different: a
    generated ``telemetry.json`` is explicitly untrusted and cannot contribute
    runtime evidence, even when it contains plausible semantic-looking data.
    """

    manifest_path = root / "run_manifest.json"
    trusted_required = False
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            trusted_required = isinstance(manifest, dict) and manifest.get("trusted_observer_required") is True
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            trusted_required = True
    if trusted_required:
        report = read_trusted_observer_output(
            root,
            observer_source_path=ROOT / "blender" / "trusted_observer.py",
        )
        if report.get("status") != "pass":
            return None
        telemetry = report.get("telemetry")
    else:
        telemetry_path = root / "telemetry.json"
        try:
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8")) if telemetry_path.is_file() else {}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(telemetry, dict):
        return None
    observations = telemetry.get("observations") or telemetry.get("runtime_observations") or []
    return observations if isinstance(observations, list) else None


def _blind_frame_metadata(
    frame_paths: list[str | Path], video_probe: dict[str, Any]
) -> list[dict[str, Any]]:
    """Materialize only frame numbers/timecodes for the primary Judge."""

    fps = float(video_probe.get("fps") or 24.0)
    if fps <= 0:
        fps = 24.0
    metadata: list[dict[str, Any]] = []
    for path in frame_paths:
        match = re.search(r"frame_(\d+)$", Path(path).stem)
        frame = int(match.group(1)) if match else None
        time_s = (frame - 1) / fps if frame is not None else None
        if time_s is None:
            timecode = None
        else:
            whole = int(time_s)
            hours, remainder = divmod(whole, 3600)
            minutes, seconds = divmod(remainder, 60)
            millis = int(round((time_s - whole) * 1000.0))
            if millis >= 1000:
                seconds += 1
                millis = 0
            timecode = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
        metadata.append({"frame": frame, "time_s": time_s, "timecode": timecode})
    return metadata


def _run_obligation_ids(root: Path) -> list[str]:
    """Read identity-only obligation anchors after the blind Judge call."""

    path = root / "run_manifest.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    values = payload.get("obligation_ids", []) if isinstance(payload, dict) else []
    return list(dict.fromkeys(str(item) for item in values if isinstance(item, str) and item.strip()))


def _event_timing_applicable(
    contract: dict[str, Any],
    required_event_ids: list[str],
    *,
    camera_motion: bool,
) -> bool:
    """Require timing evidence for discrete events, not a full-scene hold."""

    if not required_event_ids:
        return False
    if camera_motion:
        return True
    duration_s = float(contract.get("duration_s") or 0.0)
    required = set(required_event_ids)
    for event in contract.get("events", []) or []:
        if not isinstance(event, dict) or str(event.get("id") or "") not in required:
            continue
        label = str(event.get("action") or event.get("description") or "").strip().casefold()
        start_s = float(event.get("start") or 0.0)
        end_s = float(event.get("end") or start_s)
        full_scene_observe = (
            label in {"observe", "scene remains observable", "scene remains visible"}
            and start_s <= 0.0
            and (duration_s <= 0.0 or end_s >= duration_s)
        )
        if not full_scene_observe:
            return True
    return False


def _score_visual_primary(
    run_dir: str | Path,
    *,
    deterministic: DeterministicReport,
    response: VLMJudgeResponse,
    source: str,
    review_source_label: str,
    scene_contract: Any,
    frame_paths: list[str | Path],
    video_probe: dict[str, Any],
    model: str | None = None,
    raw_response_id: str | None = None,
    local_video_evidence: dict[str, Any] | None = None,
    strict_evidence: bool = False,
) -> dict[str, Any]:
    """Persist the new evaluator channels without legacy deterministic fusion."""

    root = Path(run_dir)
    if local_video_evidence is not None:
        (root / "local_video_evidence.json").write_text(
            json.dumps(local_video_evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    contract = scene_contract.model_dump(mode="json") if hasattr(scene_contract, "model_dump") else scene_contract
    contract = contract if isinstance(contract, dict) else {}
    entities = [item for item in contract.get("entities", []) if isinstance(item, dict)]
    actor_present = any(str(item.get("kind") or "").lower() in {"actor", "character"} for item in entities)
    prop_present = any(str(item.get("kind") or "").lower() == "prop" for item in entities)
    camera_constraints = " ".join(str(item) for item in contract.get("camera_constraints", []) or []).lower()
    camera_motion = any(token in camera_constraints for token in ("orbit", "dolly", "zoom", "pan", "tilt", "follow", "reframe"))
    required_event_ids = [
        str(event.get("id"))
        for event in contract.get("events", []) or []
        if isinstance(event, dict) and str(event.get("id") or "").strip()
        and (
            not contract.get("must_show")
            or str(event.get("id")) in {str(item) for item in contract.get("must_show", [])}
        )
    ]
    applicability = {
        "character_trajectory": actor_present,
        "object_trajectory": prop_present,
        "camera_motion": camera_motion,
        "event_timing": _event_timing_applicable(
            contract,
            required_event_ids,
            camera_motion=camera_motion,
        ),
    }
    primary = score_visual_review(
        response,
        artifact_gate_pass=True,
        source=source,
        applicability=applicability,
        required_event_ids=required_event_ids,
        required_event_scores=response.event_scores,
        strict_evidence=strict_evidence,
    )
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
    evaluation_result = None
    obligation_ids = _run_obligation_ids(root)
    artifact_report_path = root / "artifact_report.json"
    if artifact_report_path.is_file():
        try:
            artifact_payload = json.loads(artifact_report_path.read_text(encoding="utf-8"))
            observations = _runtime_observations_for_evaluation(root)
            evaluation_result = build_evaluation_result(
                artifact_status=artifact_payload.get("artifact_status"),
                video_probe=video_probe,
                runtime_observation_count=len(observations) if isinstance(observations, list) else 0,
                visual_status=primary.status,
                semantic_score=primary.semantic_score,
                observability_score=primary.observability_score,
                presentation_score=primary.presentation_score,
                task_score=primary.task_score,
                realism_score=realism.get("score"),
                required_event_scores=primary.required_event_scores,
                confidence=response.confidence,
                obligation_ids=obligation_ids,
            ).model_dump(mode="json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            evaluation_result = build_evaluation_result(
                artifact_status="unavailable",
                video_probe=video_probe,
                runtime_observation_count=None,
                visual_status=primary.status,
                semantic_score=None,
                observability_score=None,
                presentation_score=None,
                task_score=None,
                realism_score=None,
                obligation_ids=obligation_ids,
            ).model_dump(mode="json")
    result = {
        "status": "scored",
        "evaluator_version": VISUAL_PRIMARY_VERSION,
        "review_source": review_source_label,
        "vlm_model": model,
        "confidence": response.confidence,
        "video_probe": video_probe,
        "vlm_response": response.model_dump(mode="json"),
        "visual_primary": primary.model_dump(mode="json"),
        "scoring_v7": score_v7(
            response,
            applicability=applicability,
            required_event_ids=required_event_ids,
            required_event_scores=response.event_scores,
        ).model_dump(mode="json"),
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
        "evaluation_result": evaluation_result,
        "obligation_ids": obligation_ids,
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
    provider: Any | None = None,
    scoring_policy: str = VISUAL_PRIMARY_VERSION,
    max_workers: int = 1,
    case_ids: list[str] | None = None,
    visual_frame_budget: int | None = None,
) -> list[dict[str, Any]]:
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer")
    root = Path(run_root)
    records = [
        json.loads(line)
        for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["case_id"]: record for record in records}
    requested_case_ids = [str(case_id) for case_id in case_ids] if case_ids is not None else None
    if requested_case_ids is not None and len(requested_case_ids) != len(set(requested_case_ids)):
        raise ValueError("case_ids must be unique")
    discovered = discover_run_dirs(root)
    if requested_case_ids is not None:
        discovered_by_id = {
            str(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["case_id"]): run_dir
            for run_dir in discovered
        }
        missing = sorted(set(requested_case_ids) - set(discovered_by_id))
        if missing:
            raise ValueError(f"requested case IDs have no rendered run: {missing}")
        run_dirs = [discovered_by_id[case_id] for case_id in requested_case_ids]
    else:
        run_dirs = discovered
    entries = []
    for run_dir in run_dirs:
        case_id = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["case_id"]
        record = by_id[case_id]
        review = None
        if assistant_review_dir is not None:
            review_path = Path(assistant_review_dir) / f"{case_id}.json"
            if review_path.is_file():
                review = json.loads(review_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "case_id": case_id,
                "run_dir": run_dir,
                "prompt": record["prompt"],
                "scene_contract": json.loads((run_dir / "scene_contract.json").read_text(encoding="utf-8")),
                "assistant_review": review,
            }
        )

    def evaluate_entry(entry: dict[str, Any]) -> dict[str, Any]:
        # A provider session owns its call-record cursor.  Clone it per case
        # before concurrent evaluation so one case cannot attach another
        # case's call provenance to its vlm_report.json.  Providers without a
        # clone boundary remain on the safe sequential path below.
        case_provider = provider.clone() if provider is not None and hasattr(provider, "clone") else provider
        result = evaluate_vlm_run(
            entry["run_dir"],
            prompt=entry["prompt"],
            scene_contract=entry["scene_contract"],
            assistant_local=assistant_local,
            assistant_review=entry["assistant_review"],
            provider=case_provider,
            scoring_policy=scoring_policy,
            visual_frame_budget=visual_frame_budget,
        )
        return {"case_id": entry["case_id"], **result}

    if (
        max_workers == 1
        or provider is None
        or not hasattr(provider, "clone")
        or getattr(provider, "parallel_safe", True) is False
        or len(entries) <= 1
    ):
        return [evaluate_entry(entry) for entry in entries]
    results: list[dict[str, Any] | None] = [None] * len(entries)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(entries))) as executor:
        futures = {executor.submit(evaluate_entry, entry): index for index, entry in enumerate(entries)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result for result in results if result is not None]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--assistant-local", action="store_true")
    parser.add_argument("--assistant-review-dir")
    parser.add_argument("--scoring-policy", default=VISUAL_PRIMARY_VERSION, choices=["legacy-aggregate", "visual-primary-v6", VISUAL_PRIMARY_VERSION])
    parser.add_argument("--visual-frame-budget", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_split(
                args.run_root,
                args.dataset_root,
                assistant_local=args.assistant_local,
                assistant_review_dir=args.assistant_review_dir,
                scoring_policy=args.scoring_policy,
                visual_frame_budget=args.visual_frame_budget,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
