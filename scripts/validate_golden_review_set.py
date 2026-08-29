"""Validate the human golden-review bundle without reading hidden arm labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from videoact.real_artifacts import probe_mp4


GOLDEN_DIMENSIONS = (
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
LEAK_KEYS = {"arm", "variant", "branch", "commit", "score", "source_arm", "harness_version"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [_read_json_line(line, path, index) for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip()]


def _read_json_line(line: str, path: Path, index: int) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}:{index} is not an object")
    return value


def _leak_keys(payload: dict[str, Any]) -> set[str]:
    return {key for key in payload if key.lower() in LEAK_KEYS}


def _bundle_path(bundle: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = bundle / path
    try:
        resolved = path.resolve()
        resolved.relative_to(bundle.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def validate_golden_review_set(root: str | Path) -> dict[str, Any]:
    bundle = Path(root)
    errors: list[str] = []
    manifest_path = bundle / "manifest.jsonl"
    scores_path = bundle / "human_scores.jsonl"
    metadata_path = bundle / "metadata.json"
    try:
        manifest = _read_jsonl(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = []
        errors.append(f"manifest: {exc}")
    try:
        score_rows = _read_jsonl(scores_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        score_rows = []
        errors.append(f"human_scores: {exc}")
    try:
        metadata = _read_json(metadata_path)
        if not isinstance(metadata, dict):
            raise ValueError("metadata is not an object")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        metadata = {}
        errors.append(f"metadata: {exc}")

    case_ids = [str(row.get("case_id", "")) for row in manifest]
    if len(case_ids) < 30 or len(case_ids) > 50:
        errors.append(f"case count must be between 30 and 50, got {len(case_ids)}")
    if len(case_ids) != len(set(case_ids)) or "" in case_ids:
        errors.append("manifest case IDs must be non-empty and unique")
    for row in manifest:
        leaked = _leak_keys(row)
        if leaked:
            errors.append(f"manifest case {row.get('case_id')}: arm/experiment metadata leak: {sorted(leaked)}")
        prompt = row.get("prompt")
        prompt_en = row.get("prompt_en", prompt)
        if not isinstance(prompt, str) or not prompt.strip():
            errors.append(f"manifest case {row.get('case_id')}: prompt must be a non-empty string")
        if not isinstance(prompt_en, str) or not prompt_en.strip():
            errors.append(f"manifest case {row.get('case_id')}: prompt_en must be a non-empty string")
        elif prompt_en != prompt:
            errors.append(f"manifest case {row.get('case_id')}: prompt_en must equal prompt for blind review")
        if "source_prompt" in row and row.get("source_prompt") != prompt:
            errors.append(f"manifest case {row.get('case_id')}: source_prompt must equal prompt")
        if "prompt_origin" in row and row.get("prompt_origin") not in {"benchmark_verbatim", "dataset_record"}:
            errors.append(f"manifest case {row.get('case_id')}: unknown prompt_origin")

    samples_by_case: dict[str, set[str]] = {}
    for row in manifest:
        case_id = str(row.get("case_id", ""))
        frames = row.get("sampled_frames")
        videos = row.get("sampled_videos")
        if not isinstance(frames, dict) or not isinstance(videos, dict):
            errors.append(f"manifest case {case_id}: sampled_frames and sampled_videos must be objects keyed by sample_id")
            continue
        frame_labels = {str(label) for label in frames}
        video_labels = {str(label) for label in videos}
        if frame_labels != video_labels:
            errors.append(f"manifest case {case_id}: frame/video sample labels differ")
        if len(frame_labels) < 3:
            errors.append(f"manifest case {case_id}: at least three blind samples are required")
        samples_by_case[case_id] = frame_labels & video_labels
        for sample_id, video_value in videos.items():
            video_path = _bundle_path(bundle, video_value)
            if video_path is None or not video_path.is_file() or video_path.stat().st_size == 0:
                errors.append(f"manifest case {case_id}/{sample_id}: sampled video must be a non-empty path inside the bundle")
                continue
            probe = probe_mp4(video_path)
            if not probe["playable"]:
                errors.append(f"manifest case {case_id}/{sample_id}: sampled video is not playable")

    annotators_by_sample: dict[tuple[str, str], set[str]] = {}
    seen_rows: set[tuple[str, str, str]] = set()
    for row in score_rows:
        case_id = str(row.get("case_id", ""))
        sample_id = str(row.get("sample_id", ""))
        annotator = str(row.get("annotator_id", ""))
        if not sample_id:
            errors.append(f"score row {case_id}/{annotator}: sample_id is required")
        elif case_id not in samples_by_case or sample_id not in samples_by_case[case_id]:
            errors.append(f"score row {case_id}/{sample_id}/{annotator}: unknown sample_id")
        if not annotator:
            errors.append(f"score row {case_id}/{sample_id}: annotator_id is required")
        row_key = (case_id, sample_id, annotator)
        if row_key in seen_rows:
            errors.append(f"duplicate score row {case_id}/{sample_id}/{annotator}")
        seen_rows.add(row_key)
        annotators_by_sample.setdefault((case_id, sample_id), set()).add(annotator)
        leaked = _leak_keys(row)
        if leaked:
            errors.append(f"score row {case_id}/{sample_id}/{annotator}: arm/experiment metadata leak: {sorted(leaked)}")
        scores = row.get("scores")
        if not isinstance(scores, dict):
            errors.append(f"score row {case_id}/{sample_id}/{annotator}: scores must be an object")
            continue
        missing = sorted(set(GOLDEN_DIMENSIONS) - set(scores))
        extra = sorted(set(scores) - set(GOLDEN_DIMENSIONS))
        if missing:
            errors.append(f"score row {case_id}/{sample_id}/{annotator}: missing dimensions {missing}")
        if extra:
            errors.append(f"score row {case_id}/{sample_id}/{annotator}: unknown dimensions {extra}")
        for dimension in GOLDEN_DIMENSIONS:
            if dimension in scores and (
                isinstance(scores[dimension], bool)
                or not isinstance(scores[dimension], (int, float))
                or not math.isfinite(float(scores[dimension]))
            ):
                errors.append(f"score row {case_id}/{sample_id}/{annotator}: {dimension} is not numeric")
            elif dimension in scores and not 0 <= float(scores[dimension]) <= 100:
                errors.append(f"score row {case_id}/{sample_id}/{annotator}: {dimension} outside 0-100")

    missing_samples = sorted(
        (case_id, sample_id)
        for case_id, sample_ids in samples_by_case.items()
        for sample_id in sample_ids
        if (case_id, sample_id) not in annotators_by_sample
    )
    if missing_samples:
        errors.append(f"videos without human scores: {missing_samples[:5]}")
    low_coverage = sorted(
        (case_id, sample_id)
        for case_id, sample_ids in samples_by_case.items()
        for sample_id in sample_ids
        if len(annotators_by_sample.get((case_id, sample_id), set())) < 2
    )
    if low_coverage:
        errors.append(f"each blind video needs at least two independent annotators: {low_coverage[:5]}")
    agreement = metadata.get("inter_rater_agreement")
    if not isinstance(agreement, dict) or any(dimension not in agreement for dimension in GOLDEN_DIMENSIONS):
        errors.append("metadata must report inter-rater agreement for all 14 dimensions")
    elif any(
        not isinstance(agreement.get(dimension), dict)
        or not isinstance(agreement[dimension].get("metric"), str)
        or not agreement[dimension].get("metric")
        or isinstance(agreement[dimension].get("value"), bool)
        or not isinstance(agreement[dimension].get("value"), (int, float))
        or not math.isfinite(float(agreement[dimension].get("value")))
        or not -1 <= float(agreement[dimension].get("value")) <= 1
        for dimension in GOLDEN_DIMENSIONS
    ):
        errors.append("metadata inter-rater agreement entries must have a metric and finite value in [-1, 1]")
    if metadata.get("patch_selection_allowed") is not False:
        errors.append("golden review set must declare patch_selection_allowed=false")
    if metadata.get("arms_hidden") is not True:
        errors.append("golden review set must declare arms_hidden=true")
    mismatch_count = metadata.get("render_prompt_mismatch_count", 0)
    if isinstance(mismatch_count, bool) or not isinstance(mismatch_count, int) or mismatch_count < 0:
        errors.append("render_prompt_mismatch_count must be a non-negative integer")
    elif mismatch_count:
        errors.append("render prompt differs from the displayed source prompt; bundle is comparison-only")
    if metadata.get("comparison_only") is True and mismatch_count == 0:
        errors.append("comparison_only=true requires a non-zero render_prompt_mismatch_count")
    annotator_counts = [
        len(annotators_by_sample.get((case_id, sample_id), set()))
        for case_id, sample_ids in samples_by_case.items()
        for sample_id in sample_ids
    ]
    report = {
        "status": "pass" if not errors else "fail",
        "case_count": len(case_ids),
        "score_row_count": len(score_rows),
        "annotators_per_case": min(annotator_counts) if annotator_counts else 0,
        "annotators_per_sample": min(annotator_counts) if annotator_counts else 0,
        "sample_count": sum(len(values) for values in samples_by_case.values()),
        "annotator_count": len({annotator for values in annotators_by_sample.values() for annotator in values}),
        "dimensions": list(GOLDEN_DIMENSIONS),
        "patch_selection_allowed": metadata.get("patch_selection_allowed"),
        "errors": errors,
        "root": str(bundle.resolve()),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root_positional", nargs="?", help="golden-review bundle directory")
    parser.add_argument("--root", dest="root_option", help="golden-review bundle directory")
    args = parser.parse_args()
    root = args.root_option or args.root_positional or "dataset/golden-review-exact-v2"
    report = validate_golden_review_set(root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
