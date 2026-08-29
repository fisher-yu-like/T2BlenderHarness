"""Render bounded case groups and evaluate the finished group in the background.

The scheduler deliberately has two levels of concurrency:

* one rendering group at a time, with a bounded number of Blender CLI workers
  (at most 12; the safe default is 4);
* one evaluator worker for the previously finished group while the next group
  renders.  The evaluator is serial because the local realism pass launches a
  Blender geometry-inspection process per case.

This command writes only real render/evaluator evidence.  It does not invoke a
template, manufacture a VLM score, or alter a frozen job source.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_real_runs import evaluate_real_run  # noqa: E402
from scripts.render_proxy_jobs_parallel import render_jobs  # noqa: E402


def group_case_ids(case_ids: Iterable[str], *, group_size: int = 12) -> list[list[str]]:
    """Split an ordered case list into bounded, non-empty groups."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    values = [str(case_id) for case_id in case_ids]
    if len(values) != len(set(values)):
        raise ValueError("case IDs must be unique within a run root")
    return [values[index : index + group_size] for index in range(0, len(values), group_size)]


def _records(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    manifest = Path(dataset_root) / "manifest.jsonl"
    return {
        str(record["case_id"]): record
        for record in (
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _evaluate_group(
    run_root: Path,
    case_ids: list[str],
    *,
    dataset_root: Path,
    blender_bin: str,
) -> dict[str, Any]:
    records = _records(dataset_root)
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        run_dir = run_root / case_id
        try:
            cases.append(
                evaluate_real_run(
                    run_dir,
                    record=records.get(case_id),
                    blender_bin=blender_bin,
                )
            )
        except Exception as exc:  # keep the group audit append-only and explicit
            cases.append(
                {
                    "case_id": case_id,
                    "status": "evaluation_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "proxy_video": str((run_dir / "proxy.mp4").resolve()),
                }
            )
    scored = [float(item["score"]) for item in cases if item.get("score") is not None]
    realism = [
        float((item.get("realism") or {}).get("score"))
        for item in cases
        if (item.get("realism") or {}).get("score") is not None
    ]
    return {
        "run_root": str(run_root.resolve()),
        "case_ids": case_ids,
        "case_count": len(case_ids),
        "evaluated_count": len(cases),
        "pass_count": sum(item.get("status") == "pass" for item in cases),
        "mean_deterministic_score": round(sum(scored) / len(scored), 4) if scored else None,
        "mean_artifact_only_realism_score": round(sum(realism) / len(realism), 4) if realism else None,
        "visual_review": "pending_human_or_external_vlm",
        "cases": cases,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_grouped_pipeline(
    run_roots: list[str | Path],
    *,
    dataset_root: str | Path,
    blender_bin: str,
    group_size: int = 12,
    workers: int = 4,
    timeout_s: int = 300,
    max_retries: int = 2,
    output: str | Path | None = None,
) -> dict[str, Any]:
    if not 1 <= workers <= 12:
        raise ValueError("workers must be between 1 and 12")
    roots = [Path(value).resolve() for value in run_roots]
    if not roots:
        raise ValueError("at least one run root is required")
    all_groups: list[tuple[Path, list[str], int]] = []
    for run_root in roots:
        case_ids = sorted(path.parent.name for path in run_root.glob("*/blender_job.py"))
        for case_group in group_case_ids(case_ids, group_size=group_size):
            all_groups.append((run_root, case_group, len(all_groups) + 1))

    destination = Path(output).resolve() if output else roots[0].parent / "grouped_pipeline_report.json"
    progress_path = destination.with_name("grouped_pipeline_progress.jsonl")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text("", encoding="utf-8")
    completed_groups: list[dict[str, Any]] = []
    pending: Future[dict[str, Any]] | None = None
    pending_meta: tuple[Path, list[str], int, dict[str, Any]] | None = None

    def collect_pending() -> None:
        nonlocal pending, pending_meta
        if pending is None or pending_meta is None:
            return
        run_root, case_ids, ordinal, render_report = pending_meta
        evaluation = pending.result()
        result = {
            "group": ordinal,
            "run_root": str(run_root),
            "case_ids": case_ids,
            "render": render_report,
            "evaluation": evaluation,
        }
        completed_groups.append(result)
        with progress_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        _write_json(destination, {"status": "in_progress", "groups_completed": completed_groups, "progress_path": str(progress_path.resolve())})
        pending = None
        pending_meta = None

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as evaluator_pool:
        for run_root, case_ids, ordinal in all_groups:
            render_report = render_jobs(
                run_root,
                blender_bin=blender_bin,
                workers=workers,
                timeout_s=timeout_s,
                max_retries=max_retries,
                case_ids=case_ids,
            )
            render_result = {
                "group": ordinal,
                "run_root": str(run_root),
                "case_ids": case_ids,
                "render": render_report,
            }
            _write_json(destination.with_name(f"group-{ordinal:03d}-render.json"), render_result)
            # The next render starts immediately after this call returns while
            # the previous group's evaluator future is still running.  Before
            # submitting a new future, collect the previous one so evaluator
            # concurrency remains exactly one.
            if pending is not None and pending_meta is not None:
                collect_pending()
            pending = evaluator_pool.submit(
                _evaluate_group,
                run_root,
                case_ids,
                dataset_root=Path(dataset_root).resolve(),
                blender_bin=blender_bin,
            )
            pending_meta = (run_root, case_ids, ordinal, render_report)
        collect_pending()

    final = {
        "status": "completed",
        "group_size": group_size,
        "render_workers": workers,
        "evaluator_workers": 1,
        "max_render_retries": max_retries,
        "elapsed_s": round(time.monotonic() - started, 3),
        "group_count": len(all_groups),
        "groups_completed": completed_groups,
        "progress_path": str(progress_path.resolve()),
    }
    _write_json(destination, final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True, help="prepared root containing case directories; repeat roots")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--max-render-retries", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = run_grouped_pipeline(
        args.run_root,
        dataset_root=args.dataset_root,
        blender_bin=args.blender_bin,
        group_size=args.group_size,
        workers=args.workers,
        timeout_s=args.timeout_s,
        max_retries=args.max_render_retries,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
