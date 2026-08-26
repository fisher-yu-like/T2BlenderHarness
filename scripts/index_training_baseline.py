"""Index retained single-entity training evidence without copying artifacts.

The report under ``round-01/attempt_report.json`` is the immutable source of
truth.  This module extracts a small summary, fingerprints the report bytes,
and refuses to relabel an incomplete visual review as a completed score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


EXPECTED_RUN_ROOT = "single-five-rounds-v1"
EXPECTED_ROUND = 1
EXPECTED_ATTEMPT = 3
EXPECTED_CASE_COUNT = 10
EXPECTED_DETERMINISTIC_MEAN = 100.0
EXPECTED_SPLIT_STATUS = "incomplete_vlm_scoring"
EXPECTED_SCORING_MODE = "real_blender_video_vlm"
EXPECTED_VISUAL_REVIEW_STATUS = "unavailable"


class BaselineEvidenceError(ValueError):
    """Raised when retained baseline evidence is missing or inconsistent."""


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineEvidenceError(f"{label} must be an object")
    return value


def _main_checkout_report(requested: Path) -> Path | None:
    """Find a retained report next to a worktree for a relative request.

    The source ``out/`` tree is intentionally ignored by Git and lives in the
    main checkout.  A relative CLI argument from a worktree therefore needs a
    read-only fallback to that checkout, while the requested path remains the
    display value in the summary.
    """

    if requested.is_absolute():
        return None
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if parent.name != ".worktrees":
            continue
        candidate = parent.parent / requested / "round-01" / "attempt_report.json"
        if candidate.is_file():
            return candidate
    return None


def _resolve_report(run_root: str | Path) -> tuple[Path, Path]:
    requested = Path(run_root)
    display_root = requested.resolve()
    if display_root.name != EXPECTED_RUN_ROOT:
        raise BaselineEvidenceError(
            f"unexpected run root {display_root.name!r}; expected {EXPECTED_RUN_ROOT!r}"
        )

    report_path = display_root / "round-01" / "attempt_report.json"
    if not report_path.is_file():
        retained = _main_checkout_report(requested)
        if retained is not None:
            report_path = retained
    if not report_path.is_file():
        raise BaselineEvidenceError(f"source report is missing: {report_path}")
    return display_root, report_path.resolve()


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineEvidenceError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise BaselineEvidenceError(f"{label} must be finite")
    return number


def _validate_split(split_name: str, split: Mapping[str, Any]) -> dict[str, Any]:
    case_count = split.get("case_count")
    if case_count != EXPECTED_CASE_COUNT:
        raise BaselineEvidenceError(
            f"{split_name} case count must be {EXPECTED_CASE_COUNT}, got {case_count!r}"
        )
    cases = split.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_CASE_COUNT:
        raise BaselineEvidenceError(f"{split_name} cases must contain exactly 10 entries")

    aggregate = _as_mapping(split.get("aggregate"), f"{split_name}.aggregate")
    deterministic_mean = _require_number(
        aggregate.get("mean_deterministic_score"),
        f"{split_name}.aggregate.mean_deterministic_score",
    )
    if deterministic_mean != EXPECTED_DETERMINISTIC_MEAN:
        raise BaselineEvidenceError(
            f"{split_name} deterministic mean must be 100.0, got {deterministic_mean!r}"
        )
    artifact_mean = _require_number(
        aggregate.get("mean_artifact_only_realism_score"),
        f"{split_name}.aggregate.mean_artifact_only_realism_score",
    )
    if aggregate.get("mean_final_score") is not None:
        raise BaselineEvidenceError(f"{split_name} has a completed final visual score")
    if aggregate.get("mean_task_final_score") is not None:
        raise BaselineEvidenceError(f"{split_name} has a completed task score")

    status = split.get("status")
    if status != EXPECTED_SPLIT_STATUS:
        raise BaselineEvidenceError(
            f"{split_name} visual review status is not unavailable: {status!r}"
        )
    if split.get("scoring_mode") != EXPECTED_SCORING_MODE:
        raise BaselineEvidenceError(
            f"{split_name} scoring mode is not the retained VLM mode: {split.get('scoring_mode')!r}"
        )
    if split.get("vlm_scored_count") != 0:
        raise BaselineEvidenceError(f"{split_name} has completed visual scores")

    for index, case in enumerate(cases):
        case_mapping = _as_mapping(case, f"{split_name}.cases[{index}]")
        if case_mapping.get("vlm_status") != EXPECTED_VISUAL_REVIEW_STATUS:
            raise BaselineEvidenceError(
                f"{split_name} case {index} visual review is not unavailable"
            )
        if case_mapping.get("vlm_score") is not None:
            raise BaselineEvidenceError(f"{split_name} case {index} has a completed visual score")
        if case_mapping.get("video_score") is not None:
            raise BaselineEvidenceError(f"{split_name} case {index} has a completed video score")
        if case_mapping.get("task_final_score") is not None:
            raise BaselineEvidenceError(f"{split_name} case {index} has a completed task score")

    # Keep report fields that communicate how the split was scored, but never
    # retain per-case paths or copy any artifact metadata into the baseline.
    summary = {
        "case_count": EXPECTED_CASE_COUNT,
        "aggregate": dict(aggregate),
        "mean_deterministic_score": deterministic_mean,
        "mean_artifact_only_realism_score": artifact_mean,
        "mean_task_final_score": None,
        "mean_final_score": None,
        "status": status,
        "scoring_mode": split["scoring_mode"],
        "vlm_call_policy": split.get("vlm_call_policy"),
        "vlm_model": split.get("vlm_model"),
        "vlm_scored_count": 0,
        "visual_review_status": EXPECTED_VISUAL_REVIEW_STATUS,
        "task_score": None,
    }
    return summary


def index_baseline(run_root: str | Path, out: str | Path) -> dict[str, Any]:
    """Validate and index the retained attempt report.

    ``run_root`` is resolved for display, while the SHA-256 is computed from
    the actual report file bytes.  Only ``out`` is written.
    """

    display_root, report_path = _resolve_report(run_root)
    output_path = Path(out).resolve()
    if output_path == report_path:
        raise BaselineEvidenceError("output must not overwrite the source report")

    raw_report = report_path.read_bytes()
    try:
        report = json.loads(raw_report.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineEvidenceError(f"source report is not valid JSON: {report_path}") from exc
    report_mapping = _as_mapping(report, "source report")

    if report_mapping.get("round") != EXPECTED_ROUND:
        raise BaselineEvidenceError(
            f"report round must be {EXPECTED_ROUND}, got {report_mapping.get('round')!r}"
        )
    if report_mapping.get("attempt") != EXPECTED_ATTEMPT:
        raise BaselineEvidenceError(
            f"report attempt must be {EXPECTED_ATTEMPT}, got {report_mapping.get('attempt')!r}"
        )
    batch_value = report_mapping.get("batch")
    if batch_value is not None:
        batch = _as_mapping(batch_value, "report.batch")
        if batch.get("round") != EXPECTED_ROUND:
            raise BaselineEvidenceError("batch round does not match retained round")
        for split_name in ("train", "dev"):
            batch_ids = batch.get(split_name)
            if not isinstance(batch_ids, list) or len(batch_ids) != EXPECTED_CASE_COUNT:
                raise BaselineEvidenceError(f"batch {split_name} must contain exactly 10 cases")

    splits = _as_mapping(report_mapping.get("splits"), "report.splits")
    split_summaries = {
        split_name: _validate_split(split_name, _as_mapping(splits.get(split_name), f"splits.{split_name}"))
        for split_name in ("train", "dev")
    }
    source_sha256 = hashlib.sha256(raw_report).hexdigest()
    summary = {
        "run_root": str(display_root),
        "round": EXPECTED_ROUND,
        "attempt": EXPECTED_ATTEMPT,
        "train_count": EXPECTED_CASE_COUNT,
        "dev_count": EXPECTED_CASE_COUNT,
        "splits": split_summaries,
        "visual_review_status": EXPECTED_VISUAL_REVIEW_STATUS,
        "visual_review": {"status": EXPECTED_VISUAL_REVIEW_STATUS, "scored_count": 0},
        "task_score": None,
        "source_report": {"path": str(report_path), "sha256": source_sha256},
        "source_report_sha256": source_sha256,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="retained training run root")
    parser.add_argument("--out", required=True, help="summary JSON destination")
    args = parser.parse_args(argv)
    try:
        summary = index_baseline(args.run_root, args.out)
    except BaselineEvidenceError as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
