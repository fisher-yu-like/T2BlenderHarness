"""Run the real Blender/VLM Harness protocols without synthetic scores.

The round roots produced by this protocol are immutable job
batches.  This runner renders those jobs with Blender CLI, evaluates the real
artifacts, calls the configured VLM on sampled frames, and writes one unified
report per batch.  It can also prepare and evaluate the complete train split.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_real_runs import evaluate_real_split  # noqa: E402
from scripts.evaluate_real_videos import evaluate_split  # noqa: E402
from scripts.prepare_real_jobs import prepare_jobs  # noqa: E402
from scripts.render_proxy_jobs_parallel import render_jobs  # noqa: E402
from evaluator.openai_vlm import canonical_vlm_name  # noqa: E402

_MISSING = object()


def _is_missing(value: Any) -> bool:
    return value is _MISSING or value is None or (isinstance(value, str) and not value.strip())


def _first_non_missing(*values: Any) -> Any:
    return next((value for value in values if not _is_missing(value)), _MISSING)


def _numbered_path_part(parts: tuple[str, ...], prefix: str) -> Any:
    part = next((part for part in parts if part.startswith(prefix)), None)
    if part is None:
        return _MISSING
    suffix = part.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else part


def _group_case_ids(case_ids: list[str], *, expected_groups: int = 6) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for case_id in case_ids:
        family = case_id.rsplit("-", 1)[0]
        groups.setdefault(family, []).append(case_id)
    ordered = []
    for family in sorted(groups, key=lambda value: int(value.rsplit("-", 1)[-1])):
        values = sorted(groups[family])
        if len(values) != 10:
            raise ValueError(f"each round family must contain 10 cases: {family} has {len(values)}")
        ordered.append(values)
    if len(ordered) != expected_groups:
        raise ValueError(f"expected {expected_groups} round families, got {len(ordered)}")
    return ordered


def build_six_round_batches(train_ids: list[str], dev_ids: list[str]) -> list[dict[str, list[str]]]:
    """Pair six disjoint 10-case train families with six disjoint dev families."""
    if len(train_ids) != 60 or len(dev_ids) != 60:
        raise ValueError("six-round protocol requires exactly 60 train and 60 dev case IDs")
    if set(train_ids) & set(dev_ids):
        raise ValueError("train and dev case IDs must be disjoint")
    return build_round_batches(train_ids, dev_ids)


def build_round_batches(train_ids: list[str], dev_ids: list[str]) -> list[dict[str, list[str]]]:
    """Pair balanced 10-case train/dev families for the active phase."""
    if len(train_ids) != len(dev_ids) or not train_ids or len(train_ids) % 10:
        raise ValueError("train and dev must have equal non-zero counts divisible by 10")
    expected_groups = len(train_ids) // 10
    if set(train_ids) & set(dev_ids):
        raise ValueError("train and dev case IDs must be disjoint")
    train_groups = _group_case_ids(train_ids, expected_groups=expected_groups)
    dev_groups = _group_case_ids(dev_ids, expected_groups=expected_groups)
    return [
        {"round": index + 1, "train": train_group, "dev": dev_groups[index]}
        for index, train_group in enumerate(train_groups)
    ]


def build_protocol_manifest(
    train_ids: list[str], dev_ids: list[str], test_ids: list[str], *, dataset_fingerprint: str
) -> dict[str, Any]:
    batches = build_round_batches(train_ids, dev_ids)
    if len(test_ids) != 20 or set(test_ids) & (set(train_ids) | set(dev_ids)):
        raise ValueError("test split must contain 20 frozen, disjoint cases")
    cumulative_train: list[str] = []
    cumulative_dev: list[str] = []
    rounds: list[dict[str, Any]] = []
    for batch in batches:
        cumulative_train.extend(batch["train"])
        cumulative_dev.extend(batch["dev"])
        rounds.append(
            {
                "round": batch["round"],
                "train": batch["train"],
                "dev": batch["dev"],
                "overall_evaluation": {
                    "scope": "cumulative_train_and_dev",
                    "train_cases": list(cumulative_train),
                    "dev_cases": list(cumulative_dev),
                    "blind_test": False,
                },
            }
        )
    return {
        "protocol_version": "real-round-v4-shared-review",
        "dataset_fingerprint": dataset_fingerprint,
        "round_count": len(batches),
        "train_count": len(train_ids),
        "dev_count": len(dev_ids),
        "test_count": len(test_ids),
        "attempts_per_round_max": 5,
        "attempt_policy": build_attempt_policy(overall_case_count=len(train_ids) + len(dev_ids)),
        "batch_case_count": 20,
        "overall_case_count": len(train_ids) + len(dev_ids),
        "videos_per_round_max": 5 * 20 + len(train_ids) + len(dev_ids),
        "videos_total_max": len(batches) * (5 * 20 + len(train_ids) + len(dev_ids)),
        "selection_policy": f"{len(batches)} disjoint train families paired with {len(batches)} disjoint dev families",
        "patch_scope": "Harness-only under src/videoact; Blender, dataset, evaluator, and generated plans are immutable",
        "rounds": rounds,
        "final_evaluation": {
            "scope": "all_train_and_all_dev",
            "train_cases": train_ids,
            "dev_cases": dev_ids,
            "blind_test_cases": test_ids,
        },
    }


def build_multi_five_round_manifest(
    train_ids: list[str],
    dev_ids: list[str],
    test_ids: list[str],
    *,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    """Build the trajectory-v4-multi five-round outer-loop protocol.

    Each attempt pairs ten train cases with ten dev cases.  The round-end
    overall evaluation uses all train cases seen so far and all sixty dev
    cases, so the held-out dev set cannot be silently narrowed to the paired
    batch.
    """
    if len(train_ids) != 50 or len(dev_ids) != 60 or len(test_ids) != 30:
        raise ValueError("multi-five-round protocol requires 50 train, 60 dev, and 30 test cases")
    if set(train_ids) & set(dev_ids) or set(train_ids) & set(test_ids) or set(dev_ids) & set(test_ids):
        raise ValueError("multi-five-round splits must be disjoint")
    def fixed_groups(case_ids: list[str], group_count: int) -> list[list[str]]:
        ordered = sorted(case_ids)
        if len(ordered) % group_count:
            raise ValueError("multi-five-round IDs cannot be split into ten-case groups")
        size = len(ordered) // group_count
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]

    train_groups = fixed_groups(train_ids, 5)
    dev_groups = fixed_groups(dev_ids, 6)
    rounds: list[dict[str, Any]] = []
    cumulative_train: list[str] = []
    for index, train_group in enumerate(train_groups):
        paired_dev = dev_groups[index]
        cumulative_train.extend(train_group)
        rounds.append(
            {
                "round": index + 1,
                "train": train_group,
                "dev": paired_dev,
                "overall_evaluation": {
                    "scope": "cumulative_train_and_all_dev",
                    "train_cases": list(cumulative_train),
                    "dev_cases": list(dev_ids),
                    "blind_test": False,
                },
            }
        )
    overall_counts = [
        len(item["overall_evaluation"]["train_cases"])
        + len(item["overall_evaluation"]["dev_cases"])
        for item in rounds
    ]
    attempt_policy = build_attempt_policy(overall_case_count=110)
    attempt_policy["overall_case_counts_by_round"] = overall_counts
    attempt_policy["videos_total_max"] = 5 * 100 + sum(overall_counts)
    return {
        "protocol_version": "multi-five-rounds-v1",
        "dataset_fingerprint": dataset_fingerprint,
        "round_count": 5,
        "train_count": 50,
        "dev_count": 60,
        "test_count": 30,
        "attempts_per_round_max": 5,
        "attempt_policy": attempt_policy,
        "batch_case_count": 20,
        "overall_case_counts_by_round": overall_counts,
        "videos_per_attempt": 20,
        "videos_per_round_max": 100 + max(overall_counts),
        "videos_total_max": 5 * 100 + sum(overall_counts),
        "selection_policy": "five disjoint ten-case train families paired with five ten-case dev families; sixth dev family is overall-only",
        "patch_scope": "one Harness owner per accepted patch; dataset/evaluator/Blender are frozen",
        "rounds": rounds,
        "final_evaluation": {
            "scope": "all_train_and_all_dev_then_blind_test",
            "train_cases": list(train_ids),
            "dev_cases": list(dev_ids),
            "blind_test_cases": list(test_ids),
        },
    }


def build_active_protocol_manifest(
    train_ids: list[str], dev_ids: list[str], test_ids: list[str], *, dataset_fingerprint: str, dataset_id: str | None = None
) -> dict[str, Any]:
    if dataset_id == "trajectory-v4-multi" or len(train_ids) == 50 and len(dev_ids) == 60 and len(test_ids) == 30:
        return build_multi_five_round_manifest(train_ids, dev_ids, test_ids, dataset_fingerprint=dataset_fingerprint)
    return build_protocol_manifest(train_ids, dev_ids, test_ids, dataset_fingerprint=dataset_fingerprint)


def build_attempt_policy(max_attempts: int = 5, overall_case_count: int | None = None) -> dict[str, Any]:
    if max_attempts != 5:
        raise ValueError("this protocol fixes max_attempts at 5")
    # Keep the historical no-argument helper contract for old callers/tests;
    # the active phase passes its 100-case cumulative count explicitly.
    legacy_default = overall_case_count is None
    overall_case_count = 120 if legacy_default else overall_case_count
    return {
        "mode": "outer_loop_only",
        "max_attempts": 5,
        "inner_case_retries": 0,
        "render_retries_per_case": 2,
        "videos_per_attempt": 20,
        "overall_videos_per_round": overall_case_count,
        "videos_per_round_max": 5 * 20 + overall_case_count,
        "videos_total_max": 6 * (5 * 20 + overall_case_count) if legacy_default else 5 * (5 * 20 + overall_case_count),
    }


def validate_harness_patch_paths(paths: list[str]) -> None:
    """Reject changes outside Harness source; never patch Blender/data/evaluator outputs."""
    for raw_path in paths:
        path = raw_path.replace("\\", "/").lstrip("./")
        if not path.startswith("src/videoact/"):
            raise ValueError(
                "Harness-only patch scope violation: allowed files are under src/videoact/; "
                f"rejected {raw_path}"
            )
        if path.endswith(("trajectory.json", "camera_plan.json", "scene_contract.json")):
            raise ValueError(
                "Harness-only patch scope violation: generated plan/contract contents are immutable; "
                f"rejected {raw_path}"
            )


def anti_overfit_gate(
    train_before: float,
    train_after: float,
    paired_dev_before: float,
    paired_dev_after: float,
    overall_dev_before: float,
    overall_dev_after: float,
) -> dict[str, Any]:
    """Apply strict local and cumulative holdout gates before accepting a patch."""
    checks = {
        "train_strict_gain": train_after > train_before,
        "paired_dev_non_regression": paired_dev_after >= paired_dev_before,
        "overall_dev_non_regression": overall_dev_after >= overall_dev_before,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "reason": "accepted" if all(checks.values()) else "rejected_anti_overfit_gate",
    }


def write_training_memory_markdown(destination: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write one UTF-8 Markdown table containing the complete Harness training memory."""

    def cell(value: Any) -> str:
        if _is_missing(value):
            value = "unavailable"
        return str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    def first_available(row: dict[str, Any], *keys: str) -> Any:
        return _first_non_missing(*(row.get(key, _MISSING) for key in keys))

    lines = [
        "# T2Blendercodeharness 训练记忆表",
        "",
        "每一行保留真实 proxy 视频、独立评分通道、Harness 问题、修复和自然语言处理结论。",
        "",
        "| 轮数 | Attempt | Split | Case ID | Prompt | Proxy 视频地址 | Director plan 分 | Task score | Realism score | Review | 检测出的 Harness 问题 | Owner | 修复位置/方法 | 提升或下降 | 自然语言处理 |",
        "|---:|---:|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        fix_location = first_available(row, "fix_location")
        fix_method = first_available(row, "fix_method")
        if fix_location is _MISSING:
            fix_summary = fix_method
        elif fix_method is _MISSING:
            fix_summary = fix_location
        else:
            fix_summary = f"{fix_location}: {fix_method}"

        review = first_available(row, "review")
        if review is _MISSING:
            review_source = first_available(row, "review_source")
            review_confidence = first_available(row, "review_confidence")
            if review_source is not _MISSING and review_confidence is not _MISSING:
                review = f"{review_source} confidence={review_confidence}"
            elif review_source is not _MISSING:
                review = review_source

        values = (
            first_available(row, "round"),
            first_available(row, "attempt"),
            first_available(row, "split"),
            first_available(row, "case_id"),
            first_available(row, "prompt"),
            first_available(row, "proxy_video"),
            first_available(row, "director_plan_score"),
            first_available(row, "task_score", "video_score", "score"),
            first_available(row, "realism_score"),
            review,
            first_available(row, "detected_problem"),
            first_available(row, "owner"),
            fix_summary,
            first_available(row, "delta"),
            first_available(row, "handling"),
        )
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_real_scores(
    *, run_root: str | Path, deterministic_results: list[dict[str, Any]], vlm_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Merge shared evidence while keeping task and realism scores separate."""
    deterministic_by_id = {item["case_id"]: item for item in deterministic_results}
    vlm_by_id = {item["case_id"]: item for item in vlm_results}
    case_ids = sorted(set(deterministic_by_id) | set(vlm_by_id))
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        real = deterministic_by_id.get(case_id, {})
        vlm = vlm_by_id.get(case_id, {})
        aggregate = vlm.get("aggregate") or {}
        realism = (vlm.get("realism") if vlm.get("status") == "scored" else None) or real.get("realism") or {}
        video_path = Path(run_root) / case_id / "proxy.mp4"
        video_score = aggregate.get("final_score") if vlm.get("status") == "scored" else None
        cases.append(
            {
                "case_id": case_id,
                "proxy_video": str(video_path.resolve()),
                "video_exists": video_path.is_file() and video_path.stat().st_size > 0,
                "artifact_status": real.get("artifact_status"),
                "deterministic_status": real.get("status"),
                "deterministic_score": real.get("score"),
                "deterministic_findings": real.get("findings", []),
                "deterministic_finding_details": real.get("finding_details", []),
                "vlm_status": vlm.get("status"),
                "review_source": vlm.get("review_source"),
                "review_confidence": (
                    vlm.get("review_confidence")
                    or vlm.get("confidence")
                    or (vlm.get("vlm_response") or {}).get("confidence")
                ),
                "vlm_score": aggregate.get("vlm_score"),
                "video_score": video_score,
                "vlm_reason": vlm.get("reason"),
                "task_final_score": video_score,
                "realism_score": realism.get("score"),
                "realism_score_kind": realism.get("score_kind"),
                "realism_band": realism.get("band"),
                "realism_claim": realism.get("realism_claim"),
                "realism_requires_independent_review": realism.get("requires_independent_review"),
                "realism_evaluator_version": realism.get("evaluator_version"),
                "render_retry_count": _render_retry_count(Path(run_root) / case_id),
            }
        )
    scored = [float(item["video_score"]) for item in cases if item["video_score"] is not None]
    review_sources = {
        item.get("review_source")
        for item in cases
        if item.get("video_score") is not None and item.get("review_source")
    }
    if review_sources == {"assistant_local_review"}:
        scoring_mode = "real_blender_video_assistant_local_review"
    elif review_sources and review_sources != {"external_vlm"}:
        scoring_mode = "real_blender_video_mixed_review"
    else:
        scoring_mode = "real_blender_video_vlm"
    deterministic_scores = [
        float(item["deterministic_score"])
        for item in cases
        if item["deterministic_score"] is not None
    ]
    realism_scores = [
        float(item["realism_score"])
        for item in cases
        if item.get("realism_score") is not None and item.get("realism_score_kind")
    ]
    failure_counts: dict[str, int] = {}
    for item in cases:
        for failure_id in item["deterministic_findings"]:
            failure_counts[failure_id] = failure_counts.get(failure_id, 0) + 1
    return {
        "scoring_mode": scoring_mode,
        "run_root": str(Path(run_root).resolve()),
        "case_count": len(cases),
        "real_video_count": sum(item["video_exists"] for item in cases),
        "vlm_scored_count": len(scored),
        "aggregate": {
            "mean_task_final_score": round(sum(scored) / len(scored), 4) if scored else None,
            "mean_final_score": round(sum(scored) / len(scored), 4) if scored else None,
            "mean_deterministic_score": round(sum(deterministic_scores) / len(deterministic_scores), 4)
            if deterministic_scores
            else None,
            "mean_artifact_only_realism_score": round(sum(realism_scores) / len(realism_scores), 4)
            if realism_scores
            else None,
            "realism_scored_count": len(realism_scores),
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        "score_channels": {
            "task_final_score": "legacy deterministic/VLM task score; not added to realism",
            "artifact_only_realism_score": "v3 geometry/PNG evidence; not added to task score",
        },
        "cases": cases,
    }


def _render_retry_count(run_dir: Path) -> int:
    path = run_dir / "render_attempts.json"
    if not path.is_file():
        return 0
    try:
        attempts = json.loads(path.read_text(encoding="utf-8"))
        return max(0, len(attempts) - 1) if isinstance(attempts, list) else 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def _dataset_records(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    return {
        record["case_id"]: record
        for record in (
            json.loads(line)
            for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def write_unified_outputs(
    report: dict[str, Any], *, dataset_root: str | Path, report_root: str | Path, markdown_path: str | Path | None = None
) -> None:
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "real_unified_score.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = _dataset_records(dataset_root)
    lines = [
        "# Real Blender Video Evaluation",
        "",
        f"- scoring mode: `{report['scoring_mode']}`",
        f"- case count: `{report['case_count']}`",
        f"- real videos: `{report['real_video_count']}`",
        f"- video-scored cases: `{report['vlm_scored_count']}`",
        f"- mean final score: `{report['aggregate']['mean_final_score']}`",
        f"- mean task final score (separate channel): `{report['aggregate'].get('mean_task_final_score')}`",
        f"- mean artifact-only realism score (separate channel): `{report['aggregate'].get('mean_artifact_only_realism_score')}`",
        "",
        "分数只来自真实 Blender 生成的 `proxy.mp4` 的采样帧，并标注 external_vlm 或 assistant_local_review 来源；artifact 不完整、VLM unavailable 或本地复核未完成的 case 不进入 mean final score。",
        "",
        "| Case | Prompt | Proxy video | Deterministic | Video review | Task final score | Artifact-only realism | Artifact | Review status | Findings |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report["cases"]:
        prompt = records.get(item["case_id"], {}).get("prompt", "").replace("|", "\\|")
        video = Path(item["proxy_video"])
        video_link = f"[{video.name}]({video.as_posix()})" if item["video_exists"] else f"NOT_RENDERED: `{video}`"
        findings = ", ".join(item["deterministic_findings"]) or "none"
        lines.append(
            "| {case_id} | {prompt} | {video} | {deterministic} | {vlm} | {final} | {realism} | {artifact} | {vlm_status} | {findings} |".format(
                case_id=item["case_id"],
                prompt=prompt,
                video=video_link,
                deterministic=item["deterministic_score"],
                vlm=item["vlm_score"],
                final=item["video_score"],
                realism=item["realism_score"],
                artifact=item["artifact_status"],
                vlm_status=f"{item['vlm_status']} ({item.get('review_source') or 'none'})",
                findings=findings,
            )
        )
    destination = Path(markdown_path) if markdown_path else root / "real_unified_score.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_real_batch(
    run_root: str | Path,
    *,
    dataset_root: str | Path,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    os.environ["OPENAI_VLM_MODEL"] = vlm_model
    render_report = render_jobs(run_root, blender_bin=blender_bin, workers=workers, timeout_s=timeout_s)
    deterministic_results = evaluate_real_split(run_root, dataset_root=dataset_root, blender_bin=blender_bin)
    preliminary = merge_real_scores(run_root=run_root, deterministic_results=deterministic_results, vlm_results=[])
    preliminary["status"] = "awaiting_shared_vlm_review"
    write_unified_outputs(
        preliminary,
        dataset_root=dataset_root,
        report_root=run_root,
        markdown_path=markdown_path,
    )
    vlm_results = evaluate_split(run_root, dataset_root=dataset_root)
    report = merge_real_scores(
        run_root=run_root,
        deterministic_results=deterministic_results,
        vlm_results=vlm_results,
    )
    report["render"] = render_report
    report["vlm_model"] = canonical_vlm_name(vlm_model)
    report["evaluator_version"] = "real-v4-shared-evidence-separate-scores"
    report["vlm_call_policy"] = "one VLM call per eligible case; geometry/PNG realism is local and separate"
    write_unified_outputs(report, dataset_root=dataset_root, report_root=run_root, markdown_path=markdown_path)
    if report["vlm_scored_count"] != report["real_video_count"]:
        report["status"] = "incomplete_vlm_scoring"
    else:
        report["status"] = "complete"
    (Path(run_root) / "real_unified_score.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def prepare_full_split(
    output_root: str | Path,
    *,
    split: str,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
) -> Path:
    split_root = Path(output_root) / split
    prepare_jobs(
        split,
        split_root,
        dataset_root=dataset_root,
        harness_version=harness_version,
        evaluator_version=evaluator_version,
    )
    return split_root


def prepare_full_train(
    output_root: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
) -> Path:
    return prepare_full_split(
        output_root,
        split="train",
        dataset_root=dataset_root,
        harness_version=harness_version,
        evaluator_version=evaluator_version,
    )


def summarize_real_reports(reports: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    cases = [case for report in reports for case in report.get("cases", [])]
    scores = [float(case["video_score"]) for case in cases if case.get("video_score") is not None]
    realism_scores = [
        float(case["realism_score"])
        for case in cases
        if case.get("realism_score") is not None
    ]
    return {
        "scope": scope,
        "case_count": len(cases),
        "real_video_count": sum(bool(case.get("video_exists")) for case in cases),
        "vlm_scored_count": len(scores),
        "mean_final_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_task_final_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_artifact_only_realism_score": round(sum(realism_scores) / len(realism_scores), 4)
        if realism_scores
        else None,
        "realism_scored_count": len(realism_scores),
        "cases": cases,
    }


def _load_patch_metadata(round_root: Path) -> dict[str, Any]:
    path = round_root / "patch_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_harness_memory_jsonl(destination: str | Path, round_reports: list[dict[str, Any]]) -> None:
    """Persist round evidence for the next Harness self-evolution decision."""
    events: list[dict[str, Any]] = []
    for report in round_reports:
        round_number = report["round"]
        patch = report.get("patch", {})
        base = {
            "round": round_number,
            "parent_version": patch.get("parent_version", "not_recorded"),
            "candidate_version": patch.get("candidate_version", "not_recorded"),
            "owner": patch.get("owner", "not_recorded"),
            "files": patch.get("files", []),
        }
        events.append({**base, "event": "proposal", "status": "recorded"})
        for split in ("train", "dev"):
            aggregate = report["splits"].get(split, {}).get("aggregate", {})
            events.append(
                {
                    **base,
                    "event": f"{split}_evaluated",
                    "mean_final_score": aggregate.get("mean_final_score"),
                    "real_video_count": report["splits"].get(split, {}).get("real_video_count"),
                    "vlm_scored_count": report["splits"].get(split, {}).get("vlm_scored_count"),
                }
            )
        events.append(
            {
                **base,
                "event": "overall_evaluated",
                "train_mean_final_score": report["overall_evaluation"]["train"].get("mean_final_score"),
                "dev_mean_final_score": report["overall_evaluation"]["dev"].get("mean_final_score"),
            }
        )
        events.append({**base, "event": "decision", "decision": patch.get("decision", "pending_patch_manifest")})
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _protocol_round(dataset_root: str | Path, round_number: int) -> dict[str, Any]:
    dataset = Path(dataset_root)
    split_payload = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_active_protocol_manifest(
        split_payload["train"], split_payload["dev"], split_payload["test"], dataset_fingerprint=metadata["fingerprint"], dataset_id=metadata.get("dataset_id")
    )
    if not 1 <= round_number <= protocol["round_count"]:
        raise ValueError(f"round must be between 1 and {protocol['round_count']}")
    return protocol["rounds"][round_number - 1]


def _memory_rows_from_reports(output_root: str | Path, dataset_root: str | Path) -> list[dict[str, Any]]:
    records = _dataset_records(dataset_root)
    output = Path(output_root)
    rows: list[dict[str, Any]] = []
    for report_path in sorted(output.rglob("real_unified_score.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        relative_parts = report_path.relative_to(output).parts
        round_number = _first_non_missing(report.get("round", _MISSING), _numbered_path_part(relative_parts, "round-"))
        attempt_number = _first_non_missing(
            report.get("attempt", _MISSING), _numbered_path_part(relative_parts, "attempt-")
        )
        path_split = report_path.parent.name if report_path.parent.name in {"train", "dev", "test"} else _MISSING
        round_index = next(
            (index for index, part in enumerate(relative_parts) if part.startswith("round-")),
            None,
        )
        patch_root = output.joinpath(*relative_parts[: round_index + 1]) if round_index is not None else report_path.parent
        patch = _load_patch_metadata(patch_root)
        for case in report.get("cases", []):
            record = records.get(case["case_id"], {})
            task_score = _first_non_missing(
                case.get("task_final_score", _MISSING),
                case.get("video_score", _MISSING),
                case.get("score", _MISSING),
            )
            rows.append(
                {
                    "round": round_number,
                    "attempt": attempt_number,
                    "split": _first_non_missing(
                        case.get("split", _MISSING), report.get("split", _MISSING), path_split
                    ),
                    "case_id": case["case_id"],
                    "prompt": record.get("prompt", _MISSING),
                    "proxy_video": case.get("proxy_video", _MISSING),
                    "director_plan_score": _first_non_missing(
                        case.get("director_plan_score", _MISSING),
                        record.get("director_plan_score", _MISSING),
                        report.get("director_plan_score", _MISSING),
                    ),
                    "task_score": task_score,
                    "realism_score": case.get("realism_score", _MISSING),
                    "review": case.get("review", _MISSING),
                    "review_source": case.get("review_source", _MISSING),
                    "review_confidence": case.get("review_confidence", _MISSING),
                    "detected_problem": _first_non_missing(
                        ", ".join(case.get("deterministic_findings", [])),
                        patch.get("detected_problem", _MISSING),
                    ),
                    "owner": _first_non_missing(case.get("owner", _MISSING), patch.get("owner", _MISSING)),
                    "fix_location": patch.get("fix_location", _MISSING),
                    "fix_method": patch.get("fix_method", _MISSING),
                    "delta": patch.get("delta", _MISSING),
                    "handling": patch.get("handling", _MISSING),
                }
            )
    return rows


def update_training_memory_table(output_root: str | Path, dataset_root: str | Path, destination: str | Path) -> None:
    write_training_memory_markdown(destination, _memory_rows_from_reports(output_root, dataset_root))


def run_outer_attempt(
    output_root: str | Path,
    *,
    round_number: int,
    attempt_number: int,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path,
) -> dict[str, Any]:
    if not 1 <= attempt_number <= 5:
        raise ValueError("attempt must be between 1 and 5; this is an outer-loop attempt, not an inner repair retry")
    batch = _protocol_round(dataset_root, round_number)
    attempt_root = Path(output_root) / f"round-{round_number:02d}" / f"attempt-{attempt_number:02d}" / "real"
    reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_root = attempt_root / split
        prepare_jobs(
            split,
            split_root,
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            case_ids=batch[split],
        )
        reports[split] = run_real_batch(
            split_root,
            dataset_root=dataset_root,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
        )
    result = {"round": round_number, "attempt": attempt_number, "batch": batch, "splits": reports}
    attempt_root.parent.parent.mkdir(parents=True, exist_ok=True)
    (attempt_root.parent.parent / "attempt_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_training_memory_table(output_root, dataset_root, markdown_path)
    return result


def run_outer_overall(
    output_root: str | Path,
    *,
    round_number: int,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path,
) -> dict[str, Any]:
    overall_root = Path(output_root) / f"round-{round_number:02d}" / "overall" / "real"
    batch = _protocol_round(dataset_root, round_number)
    reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_root = overall_root / split
        prepare_jobs(
            split,
            split_root,
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            case_ids=batch["overall_evaluation"][f"{split}_cases"],
        )
        reports[split] = run_real_batch(
            split_root,
            dataset_root=dataset_root,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
        )
    result = {
        "round": round_number,
        "scope": "cumulative_train_and_dev",
        "batch": batch,
        "splits": reports,
    }
    overall_root.parent.parent.mkdir(parents=True, exist_ok=True)
    (overall_root.parent.parent / "overall_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_training_memory_table(output_root, dataset_root, markdown_path)
    return result


def run_six_round_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    split_payload = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_protocol_manifest(
        split_payload["train"], split_payload["dev"], split_payload["test"], dataset_fingerprint=metadata["fingerprint"]
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "six_round_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = _dataset_records(dataset_root)
    memory_rows: list[dict[str, Any]] = []
    cumulative = {"train": [], "dev": []}
    round_reports: list[dict[str, Any]] = []
    for batch in protocol["rounds"]:
        round_root = root / f"round-{batch['round']:02d}" / "real"
        per_split: dict[str, dict[str, Any]] = {}
        for split in ("train", "dev"):
            split_root = round_root / split
            prepare_jobs(
                split,
                split_root,
                dataset_root=dataset_root,
                harness_version=harness_version,
                evaluator_version=evaluator_version,
                case_ids=batch[split],
            )
            report = run_real_batch(
                split_root,
                dataset_root=dataset_root,
                blender_bin=blender_bin,
                workers=workers,
                timeout_s=timeout_s,
                vlm_model=vlm_model,
            )
            per_split[split] = report
            cumulative[split].append(report)
            patch = _load_patch_metadata(root / f"round-{batch['round']:02d}")
            for case in report["cases"]:
                record = records[case["case_id"]]
                task_score = _first_non_missing(
                    case.get("task_final_score", _MISSING),
                    case.get("video_score", _MISSING),
                    case.get("score", _MISSING),
                )
                memory_rows.append(
                    {
                        "round": batch["round"],
                        "split": split,
                        "case_id": case["case_id"],
                        "prompt": record["prompt"],
                        "proxy_video": case.get("proxy_video", _MISSING),
                        "director_plan_score": _first_non_missing(
                            case.get("director_plan_score", _MISSING),
                            record.get("director_plan_score", _MISSING),
                            report.get("director_plan_score", _MISSING),
                        ),
                        "task_score": task_score,
                        "realism_score": case.get("realism_score", _MISSING),
                        "review": case.get("review", _MISSING),
                        "review_source": case.get("review_source", _MISSING),
                        "review_confidence": case.get("review_confidence", _MISSING),
                        "detected_problem": _first_non_missing(
                            ", ".join(case.get("deterministic_findings", [])),
                            patch.get("detected_problem", _MISSING),
                        ),
                        "owner": _first_non_missing(case.get("owner", _MISSING), patch.get("owner", _MISSING)),
                        "fix_location": patch.get("fix_location", _MISSING),
                        "fix_method": patch.get("fix_method", _MISSING),
                        "delta": patch.get("delta", _MISSING),
                        "handling": patch.get("handling", _MISSING),
                    }
                )
            write_training_memory_markdown(root / "harness_training_memory.md", memory_rows)
        overall = {
            "train": summarize_real_reports(cumulative["train"], scope="cumulative_train_and_dev"),
            "dev": summarize_real_reports(cumulative["dev"], scope="cumulative_train_and_dev"),
        }
        round_report = {
            "round": batch["round"],
            "batch": batch,
            "patch": _load_patch_metadata(root / f"round-{batch['round']:02d}"),
            "splits": per_split,
            "overall_evaluation": overall,
        }
        round_reports.append(round_report)
        (root / f"round-{batch['round']:02d}" / "overall_evaluation.json").write_text(
            json.dumps(round_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    write_training_memory_markdown(root / "harness_training_memory.md", memory_rows)
    memory_path = root / "memory" / "harness_updates.jsonl"
    write_harness_memory_jsonl(memory_path, round_reports)
    result = {
        "protocol": protocol,
        "rounds": round_reports,
        "memory_table": str((root / "harness_training_memory.md").resolve()),
        "memory_jsonl": str(memory_path.resolve()),
    }
    (root / "six_round_training_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def run_multi_five_round_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path,
) -> dict[str, Any]:
    """Execute one real attempt plus one round-end overall evaluation per round."""
    dataset = Path(dataset_root)
    split_payload = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_multi_five_round_manifest(
        split_payload["train"],
        split_payload["dev"],
        split_payload["test"],
        dataset_fingerprint=metadata["fingerprint"],
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "multi_five_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    round_reports = []
    for batch in protocol["rounds"]:
        attempt = run_outer_attempt(
            root,
            round_number=batch["round"],
            attempt_number=1,
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
            markdown_path=markdown_path,
        )
        overall = run_outer_overall(
            root,
            round_number=batch["round"],
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
            markdown_path=markdown_path,
        )
        round_reports.append({"round": batch["round"], "attempt": attempt, "overall": overall})
    result = {"protocol": protocol, "rounds": round_reports, "memory_table": str(Path(markdown_path).resolve())}
    (root / "multi_five_training_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["protocol", "attempt", "overall", "multi-five-rounds", "six-rounds", "existing-rounds", "full-train", "all"],
        required=True,
    )
    parser.add_argument("--dataset-root", default="dataset/trajectory-v4-multi")
    parser.add_argument("--round-root", default="out/training/multi-five-rounds-v1")
    parser.add_argument("--full-train-root", default="out/training/full-evaluation-real-v6")
    parser.add_argument("--harness-version", default="h-t2-hard-v4")
    parser.add_argument("--evaluator-version", default="real-v4-shared-evidence-separate-scores")
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--vlm-model", choices=["gpt-5.6-luna", "gpt-5.6-terra"], default="gpt-5.6-luna")
    parser.add_argument("--markdown-path", default="docs/t2blendercodeharness-multi-training-memory-v1.md")
    parser.add_argument("--round", dest="round_number", type=int)
    parser.add_argument("--attempt", dest="attempt_number", type=int)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    all_reports: dict[str, Any] = {}
    split_payload = json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_active_protocol_manifest(
        split_payload["train"], split_payload["dev"], split_payload["test"], dataset_fingerprint=metadata["fingerprint"], dataset_id=metadata.get("dataset_id")
    )
    protocol_path = Path(args.round_root) / ("multi_five_protocol.json" if protocol["protocol_version"] == "multi-five-rounds-v1" else "six_round_protocol.json")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0
    if args.mode in {"attempt", "overall"}:
        if args.round_number is None:
            raise SystemExit("--round is required for --mode attempt/overall")
        if args.mode == "attempt":
            if args.attempt_number is None:
                raise SystemExit("--attempt is required for --mode attempt")
            result = run_outer_attempt(
                args.round_root,
                round_number=args.round_number,
                attempt_number=args.attempt_number,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                markdown_path=args.markdown_path,
            )
        else:
            result = run_outer_overall(
                args.round_root,
                round_number=args.round_number,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                markdown_path=args.markdown_path,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.mode in {"multi-five-rounds", "six-rounds", "all"}:
        if protocol["protocol_version"] == "multi-five-rounds-v1":
            all_reports["multi_five_rounds"] = run_multi_five_round_protocol(
                args.round_root,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                markdown_path=args.markdown_path,
            )
        else:
            all_reports["six_rounds"] = run_six_round_protocol(
                args.round_root,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
            )
    if args.mode == "existing-rounds":
        round_root = Path(args.round_root)
        for round_dir in sorted(round_root.glob("round-*")):
            round_reports = {}
            for split in ("train", "dev"):
                split_root = round_dir / "real" / split
                if not split_root.is_dir():
                    continue
                round_reports[split] = run_real_batch(
                    split_root,
                    dataset_root=dataset_root,
                    blender_bin=args.blender_bin,
                    workers=args.workers,
                    timeout_s=args.timeout_s,
                    vlm_model=args.vlm_model,
                )
            if round_reports:
                (round_dir / "real_round_report.json").write_text(
                    json.dumps(round_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                all_reports[round_dir.name] = round_reports
    if args.mode in {"full-train", "all"}:
        train_root = prepare_full_train(
            args.full_train_root,
            dataset_root=dataset_root,
            harness_version=args.harness_version,
            evaluator_version=args.evaluator_version,
        )
        all_reports["full_train"] = run_real_batch(
            train_root,
            dataset_root=dataset_root,
            blender_bin=args.blender_bin,
            workers=args.workers,
            timeout_s=args.timeout_s,
            vlm_model=args.vlm_model,
            markdown_path=args.markdown_path,
        )
        if args.mode == "all":
            dev_root = prepare_full_split(
                args.full_train_root,
                split="dev",
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
            )
            all_reports["full_dev"] = run_real_batch(
                dev_root,
                dataset_root=dataset_root,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
            )
    summary_path = Path(args.full_train_root).parent / "real_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(all_reports, indent=2, sort_keys=True))
    return 0 if all(item.get("status") == "complete" for item in _flatten_reports(all_reports)) else 2


def _flatten_reports(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, dict) and "status" in value:
            result.append(value)
        elif isinstance(value, dict):
            result.extend(_flatten_reports(value))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
