"""Run real test cases with generation and visual review overlapped.

Each worker owns one case end-to-end: GLM DirectorPlan, GLM BlenderCode,
real Blender execution, and local read-only Codex visual review.  Because the
workers are independent, a case in visual review does not block generation of
the next cases.  No template generation path is exposed by this runner.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


def build_case_command(
    *,
    python_executable: str | Path,
    dataset_root: str | Path,
    case_id: str,
    run_root: str | Path,
    blender_bin: str | Path,
    timeout_s: int,
) -> list[str]:
    """Build the fixed, non-template command used for one streaming case."""

    return [
        str(python_executable),
        "scripts/run_batch_eval.py",
        "--dataset-root",
        str(dataset_root),
        "--split",
        "test",
        "--case-ids",
        str(case_id),
        "--run-root",
        str(run_root),
        "--blender-bin",
        str(blender_bin),
        "--workers",
        "1",
        "--timeout-s",
        str(int(timeout_s)),
        "--provider-mode",
        "glm",
        "--visual-review-provider",
        "codex",
    ]


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def _case_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("cases")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict) and row.get("case_id")]


def _provider_summary(base: Path) -> dict[str, Any]:
    provider_kinds: Counter[str] = Counter()
    model_ids: Counter[str] = Counter()
    fallback_cases = 0
    manifest_count = 0
    for path in base.rglob("provider_manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        manifest_count += 1
        if manifest.get("fallback_used") is True:
            fallback_cases += 1
        stages = manifest.get("stages")
        if not isinstance(stages, dict):
            continue
        for stage in stages.values():
            if not isinstance(stage, dict):
                continue
            calls = stage.get("calls")
            if not isinstance(calls, list):
                calls = [stage]
            for call in calls:
                if not isinstance(call, dict):
                    continue
                kind = str(call.get("provider_kind") or "unknown")
                model = str(call.get("model_id") or "unknown")
                provider_kinds[kind] += 1
                model_ids[model] += 1
    return {
        "manifest_count": manifest_count,
        "fallback_case_count": fallback_cases,
        "provider_kind_call_counts": dict(sorted(provider_kinds.items())),
        "model_id_call_counts": dict(sorted(model_ids.items())),
    }


def summarize_case_reports(
    base: str | Path,
    *,
    expected_case_ids: list[str] | None = None,
    runner_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Merge independently written case reports without hiding missing cases."""

    root = Path(base).resolve()
    by_case: dict[str, dict[str, Any]] = {}
    report_statuses: Counter[str] = Counter()
    preparation_failed_count = 0
    artifact_failed_count = 0
    for report_path in sorted(root.rglob("real_unified_score.json")):
        # The final aggregate is written at ``root`` and must not be consumed
        # as one more case report when this function is called repeatedly.
        if report_path == root / "real_unified_score.json":
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        report_statuses[str(report.get("status") or "unknown")] += 1
        rows = _case_rows_from_report(report)
        row_has_prep_fields = False
        row_has_artifact_fields = False
        for row in rows:
            case_id = str(row["case_id"])
            by_case[case_id] = row
            if row.get("preparation_status") is not None or row.get("preparation_failure"):
                preparation_failed_count += 1
                row_has_prep_fields = True
            if row.get("artifact_status") not in {None, "complete"}:
                artifact_failed_count += 1
                row_has_artifact_fields = True
        if not row_has_prep_fields:
            preparation_failed_count += int(report.get("preparation_failed_count") or 0)
        if not row_has_artifact_fields:
            artifact_failed_count += int(report.get("artifact_failed_count") or 0)

    expected = [str(case_id) for case_id in (expected_case_ids or sorted(by_case))]
    for case_id in expected:
        if case_id not in by_case:
            by_case[case_id] = {
                "case_id": case_id,
                "video_exists": False,
                "vlm_status": "unavailable",
                "vlm_reason": "runner_missing_report",
                "overall_vlm_score": None,
                "task_final_score": None,
                "realism_score": None,
            }
    cases = [by_case[case_id] for case_id in sorted(by_case)]
    scored = [row for row in cases if row.get("overall_vlm_score") is not None]
    status_counts = Counter(str(row.get("vlm_status") or "missing") for row in cases)
    real_video_count = sum(bool(row.get("video_exists")) for row in cases)
    all_expected_reported = len(cases) == len(expected) and all(
        row.get("vlm_status") == "scored" for row in cases if row.get("case_id") in set(expected)
    )
    return {
        "status": "complete" if all_expected_reported else "incomplete_streaming_eval",
        "run_root": str(root),
        "case_count": len(cases),
        "expected_case_count": len(expected),
        "real_video_count": real_video_count,
        "vlm_scored_count": len(scored),
        "preparation_failed_count": preparation_failed_count,
        "artifact_failed_count": artifact_failed_count,
        "vlm_status_counts": dict(sorted(status_counts.items())),
        "case_report_status_counts": dict(sorted(report_statuses.items())),
        "runner_results": runner_results or [],
        "provider_summary": _provider_summary(root),
        "aggregate": {
            "mean_task_final_score": _mean(row.get("task_final_score") for row in scored),
            "mean_realism_score": _mean(row.get("realism_score") for row in scored),
            "mean_overall_vlm_score": _mean(row.get("overall_vlm_score") for row in scored),
        },
        "cases": cases,
    }


def _load_test_case_ids(dataset_root: Path) -> list[str]:
    return [
        str(payload["case_id"])
        for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if payload.get("split") == "test"
    ]


def _run_one_case(
    *,
    python_executable: str | Path,
    dataset_root: Path,
    case_id: str,
    job_root: Path,
    blender_bin: str | Path,
    timeout_s: int,
) -> dict[str, Any]:
    job_root.mkdir(parents=True, exist_ok=True)
    command = build_case_command(
        python_executable=python_executable,
        dataset_root=dataset_root,
        case_id=case_id,
        run_root=job_root,
        blender_bin=blender_bin,
        timeout_s=timeout_s,
    )
    log_path = job_root / "runner.log"
    started = time.monotonic()
    result: dict[str, Any] = {
        "case_id": case_id,
        "job_root": str(job_root.resolve()),
        "log_path": str(log_path.resolve()),
        "command": command,
    }
    try:
        env = os.environ.copy()
        env["T2BLENDER_CODEX_VISUAL_LOCK_PATH"] = str(
            (job_root.parent.parent / "codex_visual_review.lock").resolve()
        )
        env.setdefault("T2BLENDER_CODEX_VISUAL_LOCK_TIMEOUT_S", "1800")
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=max(1800, int(timeout_s) * 8),
                env=env,
                check=False,
            )
        result["return_code"] = int(completed.returncode)
    except subprocess.TimeoutExpired:
        result["return_code"] = -1
        result["error"] = "streaming_case_process_timeout"
    except OSError as exc:
        result["return_code"] = -1
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    report_path = job_root / "real_unified_score.json"
    result["report_path"] = str(report_path.resolve())
    result["report_exists"] = report_path.is_file()
    return result


def safe_case_future_result(
    future: concurrent.futures.Future[dict[str, Any]],
    *,
    case_id: str,
    job_root: Path,
) -> dict[str, Any]:
    """Turn an unexpected worker exception into an auditable case result."""

    try:
        return future.result()
    except Exception as exc:  # pragma: no cover - the test exercises this boundary
        report_path = job_root / "real_unified_score.json"
        return {
            "case_id": case_id,
            "job_root": str(job_root.resolve()),
            "log_path": str((job_root / "runner.log").resolve()),
            "return_code": -1,
            "error": f"worker_exception:{type(exc).__name__}:{exc}",
            "report_path": str(report_path.resolve()),
            "report_exists": report_path.is_file(),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-test-100-v1")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise SystemExit("--workers must be between 1 and 12")
    dataset_root = Path(args.dataset_root).resolve()
    run_root = Path(args.run_root).resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"run root already contains artifacts: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    case_ids = _load_test_case_ids(dataset_root)
    if not case_ids:
        raise SystemExit("test manifest contains no test cases")
    progress_path = run_root / "stream_progress.jsonl"
    runner_results: list[dict[str, Any]] = []
    started_payload = {
        "event": "stream_started",
        "case_count": len(case_ids),
        "workers": args.workers,
        "provider_mode": "glm",
        "visual_review_provider": "codex",
    }
    progress_path.write_text(json.dumps(started_payload, ensure_ascii=False) + "\n", encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_one_case,
                python_executable=sys.executable,
                dataset_root=dataset_root,
                case_id=case_id,
                job_root=run_root / "jobs" / case_id,
                blender_bin=args.blender_bin,
                timeout_s=args.timeout_s,
            ): case_id
            for case_id in case_ids
        }
        for future in concurrent.futures.as_completed(futures):
            case_id = futures[future]
            result = safe_case_future_result(
                future,
                case_id=case_id,
                job_root=run_root / "jobs" / case_id,
            )
            runner_results.append(result)
            with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps({"event": "case_finished", **result}, ensure_ascii=False) + "\n")
            print(json.dumps({"event": "case_finished", **result}, ensure_ascii=False), flush=True)
    runner_results.sort(key=lambda item: str(item.get("case_id")))
    summary = summarize_case_reports(
        run_root,
        expected_case_ids=case_ids,
        runner_results=runner_results,
    )
    (run_root / "real_unified_score.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "test_score_summary.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "case_count": summary["case_count"],
                "real_video_count": summary["real_video_count"],
                "vlm_scored_count": summary["vlm_scored_count"],
                "aggregate": summary["aggregate"],
                "vlm_status_counts": summary["vlm_status_counts"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "stream_finished",
                "status": summary["status"],
                "case_count": summary["case_count"],
                "real_video_count": summary["real_video_count"],
                "vlm_scored_count": summary["vlm_scored_count"],
                "aggregate": summary["aggregate"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary["case_count"] == len(case_ids) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_case_command", "summarize_case_reports"]
