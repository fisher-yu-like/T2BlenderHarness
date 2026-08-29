"""Bounded real-case regeneration loop used by outer-loop training.

Each inner attempt owns a fresh plan/code/render candidate.  A failed
candidate is moved to an immutable ``inner_attempts`` archive before the next
candidate is prepared.  The module is intentionally callback based so the
training runner can keep provider, Blender, and evaluator policies at their
existing boundaries.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable


PrepareCallback = Callable[[list[str], int], dict[str, Any]]
RenderCallback = Callable[[list[str], int], dict[str, Any]]
EvaluateCallback = Callable[[str, int], dict[str, Any]]


def _archive_case(split_root: Path, case_id: str, attempt: int) -> str | None:
    source = split_root / case_id
    if not source.exists():
        return None
    destination = split_root / "inner_attempts" / case_id / f"attempt-{attempt:02d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"inner attempt archive already exists: {destination}")
    shutil.move(str(source), str(destination))
    return str(destination.resolve())


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _failure_entry(
    *,
    case_id: str,
    attempt: int,
    status: str,
    reason: str,
    archive_path: str | None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "attempt": attempt,
        "case_id": case_id,
        "status": status,
        "reason": reason,
        "archive_path": archive_path,
    }
    if detail:
        entry["detail"] = detail
    return entry


def run_real_inner_loop(
    case_ids: Iterable[str],
    split_root: str | Path,
    *,
    prepare: PrepareCallback,
    render: RenderCallback,
    evaluate: EvaluateCallback,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Regenerate and execute each failed case at most ``max_attempts`` times.

    ``prepare`` must generate a fresh DirectorPlan and Blender source for the
    supplied case IDs.  ``render`` must return per-case results with
    ``status=success`` only when real Blender artifacts are complete.  The
    evaluator is called only for successful renders; any non-pass evaluation
    is treated as a candidate failure and regenerated on the next attempt.
    """

    if not 1 <= max_attempts <= 3:
        raise ValueError("max_attempts must be between 1 and 3")
    values = [str(case_id) for case_id in case_ids]
    if not values:
        raise ValueError("at least one case ID is required")
    if len(values) != len(set(values)):
        raise ValueError("case IDs must be unique")

    root = Path(split_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "inner_loop_progress.jsonl"
    if progress_path.exists():
        raise FileExistsError(f"inner loop progress already exists: {progress_path}")

    pending = list(values)
    cases: dict[str, dict[str, Any]] = {
        case_id: {
            "case_id": case_id,
            "selected_attempt": None,
            "status": "pending",
            "attempts": [],
        }
        for case_id in values
    }

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        current = list(pending)
        attempt_entries: list[dict[str, Any]] = []
        next_pending: list[str] = []
        try:
            prepared_payload = prepare(current, attempt) or {}
        except Exception as exc:
            for case_id in current:
                archive_path = _archive_case(root, case_id, attempt)
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="prepare_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    archive_path=archive_path,
                )
                cases[case_id]["attempts"].append(entry)
                attempt_entries.append(entry)
                next_pending.append(case_id)
            pending = next_pending
            _write_progress(
                progress_path,
                {"attempt": attempt, "pending_case_ids": pending, "entries": attempt_entries},
            )
            continue

        prepared_ids = [str(case_id) for case_id in prepared_payload.get("prepared_ids", [])]
        prepared_set = set(prepared_ids)
        if not prepared_set.issubset(current):
            raise ValueError("prepare returned a case ID that was not pending")
        if len(prepared_ids) != len(prepared_set):
            raise ValueError("prepare returned duplicate prepared case IDs")

        failures = prepared_payload.get("failures", {}) or {}
        if not isinstance(failures, dict):
            raise ValueError("prepare failures must be a mapping")
        for case_id in current:
            if case_id in prepared_set:
                continue
            raw_failure = failures.get(case_id) or {}
            if not isinstance(raw_failure, dict):
                raw_failure = {"reason": str(raw_failure)}
            reason = str(raw_failure.get("reason") or raw_failure.get("status") or "plan_not_prepared")
            archive_path = _archive_case(root, case_id, attempt)
            entry = _failure_entry(
                case_id=case_id,
                attempt=attempt,
                status="plan_failed",
                reason=reason,
                archive_path=archive_path,
                detail=raw_failure,
            )
            cases[case_id]["attempts"].append(entry)
            attempt_entries.append(entry)
            next_pending.append(case_id)

        try:
            render_payload = render(prepared_ids, attempt) if prepared_ids else {}
            render_results = (render_payload or {}).get("results", {}) or {}
            if not isinstance(render_results, dict):
                raise ValueError("render results must be a mapping")
        except Exception as exc:
            for case_id in prepared_ids:
                archive_path = _archive_case(root, case_id, attempt)
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="render_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    archive_path=archive_path,
                )
                cases[case_id]["attempts"].append(entry)
                attempt_entries.append(entry)
                next_pending.append(case_id)
            pending = list(dict.fromkeys(next_pending))
            _write_progress(
                progress_path,
                {"attempt": attempt, "pending_case_ids": pending, "entries": attempt_entries},
            )
            continue
        for case_id in prepared_ids:
            raw_render = render_results.get(case_id)
            render_result = raw_render if isinstance(raw_render, dict) else {}
            if render_result.get("status") != "success":
                reason = str(render_result.get("reason") or render_result.get("status") or "render_failed")
                archive_path = _archive_case(root, case_id, attempt)
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="render_failed",
                    reason=reason,
                    archive_path=archive_path,
                    detail=render_result,
                )
                cases[case_id]["attempts"].append(entry)
                attempt_entries.append(entry)
                next_pending.append(case_id)
                continue

            try:
                evaluation = evaluate(case_id, attempt) or {}
            except Exception as exc:  # an evaluator failure is a failed candidate
                evaluation = {
                    "status": "evaluation_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            if evaluation.get("status") == "pass":
                entry = {
                    "attempt": attempt,
                    "case_id": case_id,
                    "status": "success",
                    "evaluation": evaluation,
                }
                cases[case_id]["attempts"].append(entry)
                cases[case_id]["status"] = "success"
                cases[case_id]["selected_attempt"] = attempt
                cases[case_id]["evaluation"] = evaluation
                cases[case_id]["proxy_video"] = evaluation.get("proxy_video")
                attempt_entries.append(entry)
            else:
                reason = str(evaluation.get("reason") or "plan_or_runtime_evaluation_failed")
                archive_path = _archive_case(root, case_id, attempt)
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="evaluation_failed",
                    reason=reason,
                    archive_path=archive_path,
                    detail=evaluation,
                )
                cases[case_id]["attempts"].append(entry)
                attempt_entries.append(entry)
                next_pending.append(case_id)

        pending = list(dict.fromkeys(next_pending))
        _write_progress(
            progress_path,
            {
                "attempt": attempt,
                "pending_case_ids": pending,
                "entries": attempt_entries,
            },
        )

    for case_id in pending:
        cases[case_id]["status"] = "exhausted"
        cases[case_id]["reason"] = "max_inner_attempts_exhausted"

    result = {
        "status": "completed" if not pending else "exhausted",
        "max_attempts": max_attempts,
        "case_count": len(values),
        "completed_count": sum(item["status"] == "success" for item in cases.values()),
        "exhausted_count": sum(item["status"] == "exhausted" for item in cases.values()),
        "pending_case_ids": pending,
        "progress_path": str(progress_path.resolve()),
        "cases": cases,
    }
    (root / "inner_loop_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["run_real_inner_loop"]
