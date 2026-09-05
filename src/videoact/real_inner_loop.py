"""Bounded real-case regeneration loop used by outer-loop training.

Each inner attempt owns a fresh plan/code/render candidate.  A failed
candidate is moved to an immutable ``inner_attempts`` archive before the next
candidate is prepared.  The module is intentionally callback based so the
training runner can keep provider, Blender, and evaluator policies at their
existing boundaries.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import inspect
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


PrepareCallback = Callable[[list[str], int], dict[str, Any]]
RenderCallback = Callable[[list[str], int], dict[str, Any]]
EvaluateCallback = Callable[[str, int], dict[str, Any]]
StageCallback = Callable[..., dict[str, Any]]


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


def _tree_hashes(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    hashes: dict[str, str] = {}
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        hashes[str(item.relative_to(path)).replace("\\", "/")] = digest
    return hashes


def _tree_digest(hashes: dict[str, str]) -> str | None:
    if not hashes:
        return None
    payload = json.dumps(sorted(hashes.items()), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    retry_stage: str,
    input_hashes: dict[str, Any] | None = None,
    output_hashes: dict[str, Any] | None = None,
    parent_attempt_id: str | None = None,
) -> dict[str, Any]:
    entry = {
        "attempt": attempt,
        "case_id": case_id,
        "status": status,
        "reason": reason,
        "archive_path": archive_path,
        "retry_stage": retry_stage,
        "input_hashes": input_hashes or {},
        "output_hashes": output_hashes or {},
        "parent_attempt_id": parent_attempt_id or f"{case_id}:attempt-{max(0, attempt - 1):02d}",
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
    evaluator is called only for successful renders.  A valid video with a
    semantic failure is terminal evidence for the outer loop; only execution
    or evaluation-recovery failures are regenerated on the next attempt.
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
                input_hashes = _tree_hashes(root / case_id)
                archive_path = _archive_case(root, case_id, attempt)
                archive_hashes = _tree_hashes(Path(archive_path)) if archive_path else {}
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="prepare_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    archive_path=archive_path,
                    retry_stage="director",
                    input_hashes={"case": _tree_digest(input_hashes), "files": input_hashes},
                    output_hashes={"archive": _tree_digest(archive_hashes), "files": archive_hashes},
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
            input_hashes = _tree_hashes(root / case_id)
            archive_path = _archive_case(root, case_id, attempt)
            output_hashes = _tree_hashes(Path(archive_path)) if archive_path else {}
            entry = _failure_entry(
                case_id=case_id,
                attempt=attempt,
                status="plan_failed",
                reason=reason,
                archive_path=archive_path,
                detail=raw_failure,
                retry_stage="director" if "director" in reason.lower() or "plan" in reason.lower() or "coverage" in reason.lower() else "codegen",
                input_hashes={"case": _tree_digest(input_hashes), "files": input_hashes},
                output_hashes={"archive": _tree_digest(output_hashes), "files": output_hashes},
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
                input_hashes = _tree_hashes(root / case_id)
                archive_path = _archive_case(root, case_id, attempt)
                output_hashes = _tree_hashes(Path(archive_path)) if archive_path else {}
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="render_failed",
                    reason=f"{type(exc).__name__}: {exc}",
                    archive_path=archive_path,
                    retry_stage="executor",
                    input_hashes={"case": _tree_digest(input_hashes), "files": input_hashes},
                    output_hashes={"archive": _tree_digest(output_hashes), "files": output_hashes},
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
                input_hashes = _tree_hashes(root / case_id)
                archive_path = _archive_case(root, case_id, attempt)
                output_hashes = _tree_hashes(Path(archive_path)) if archive_path else {}
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="render_failed",
                    reason=reason,
                    archive_path=archive_path,
                    detail=render_result,
                    retry_stage="collector" if "artifact" in reason.lower() else "executor",
                    input_hashes={"case": _tree_digest(input_hashes), "files": input_hashes},
                    output_hashes={"archive": _tree_digest(output_hashes), "files": output_hashes},
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
            elif (
                evaluation.get("execution_status") == "valid"
                and evaluation.get("semantic_status") in {"failed_required_event", "uncertain"}
            ) or evaluation.get("retryable") is False:
                terminal_status = "semantic_failed" if evaluation.get("semantic_status") == "failed_required_event" else "semantic_uncertain"
                entry = {
                    "attempt": attempt,
                    "case_id": case_id,
                    "status": terminal_status,
                    "reason": str(evaluation.get("reason") or terminal_status),
                    "retry_stage": "outer_loop",
                    "parent_attempt_id": f"{case_id}:attempt-{max(0, attempt - 1):02d}",
                    "evaluation": evaluation,
                }
                cases[case_id]["attempts"].append(entry)
                cases[case_id]["status"] = terminal_status
                cases[case_id]["selected_attempt"] = attempt
                cases[case_id]["evaluation"] = evaluation
                cases[case_id]["proxy_video"] = evaluation.get("proxy_video")
                attempt_entries.append(entry)
            else:
                reason = str(evaluation.get("reason") or "plan_or_runtime_evaluation_failed")
                input_hashes = _tree_hashes(root / case_id)
                archive_path = _archive_case(root, case_id, attempt)
                output_hashes = _tree_hashes(Path(archive_path)) if archive_path else {}
                entry = _failure_entry(
                    case_id=case_id,
                    attempt=attempt,
                    status="evaluation_failed",
                    reason=reason,
                    archive_path=archive_path,
                    detail=evaluation,
                    retry_stage="evaluator",
                    input_hashes={"case": _tree_digest(input_hashes), "files": input_hashes},
                    output_hashes={"archive": _tree_digest(output_hashes), "files": output_hashes},
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


def _invoke_stage_callback(callback: StageCallback, case_id: str, state: dict[str, Any], attempt: int) -> dict[str, Any]:
    """Invoke a stage hook once without signature-based hidden retries."""

    try:
        signature = inspect.signature(callback)
        positional = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        if any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()) or len(positional) >= 3:
            value = callback(case_id, dict(state), attempt)
        elif len(positional) == 2:
            value = callback(case_id, dict(state))
        elif len(positional) == 1:
            value = callback(case_id)
        else:
            value = callback()
    except (TypeError, ValueError):
        # Opaque callables are still called exactly once with the documented
        # three-argument contract.
        value = callback(case_id, dict(state), attempt)
    if not isinstance(value, dict):
        raise ValueError("stage callback must return an object")
    return value


def _stage_success(value: Mapping[str, Any]) -> bool:
    return str(value.get("status") or "").casefold() in {"success", "pass", "passed", "complete", "completed", "valid"}


def _stage_hashes(value: Mapping[str, Any]) -> dict[str, Any]:
    hashes = value.get("hashes") or value.get("output_hashes") or value.get("artifact_hashes")
    if isinstance(hashes, Mapping):
        return {str(key): str(item) for key, item in hashes.items()}
    # A stage may return explicit plan/source/blend hashes without wrapping
    # them; retain them as lineage rather than inventing file contents.
    return {
        str(key): str(value[key])
        for key in ("plan_hash", "source_hash", "blend_hash", "telemetry_hash", "mp4_hash")
        if value.get(key) is not None
    }


def run_stage_aware_inner_loop(
    case_ids: Iterable[str],
    split_root: str | Path,
    *,
    director: StageCallback,
    codegen: StageCallback,
    executor: StageCallback,
    evaluator: StageCallback,
    observer: StageCallback | None = None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Run retries at the first failed stage while preserving valid upstreams.

    The callbacks are intentionally stage-specific.  A Director failure
    retries Director; a CodeAgent failure keeps the valid plan; executor and
    observer failures keep plan/source/blend respectively; evaluator
    transport failures keep every prior artifact.  A valid video with a
    semantic failure is terminal evidence and is never regenerated here.
    """

    if not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be between 1 and 5")
    values = [str(case_id) for case_id in case_ids]
    if not values or len(values) != len(set(values)):
        raise ValueError("case IDs must be non-empty and unique")
    root = Path(split_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    progress = root / "stage_retry_progress.jsonl"
    if progress.exists():
        raise FileExistsError(f"stage retry progress already exists: {progress}")
    cases: dict[str, dict[str, Any]] = {
        case_id: {
            "case_id": case_id,
            "status": "pending",
            "selected_attempt": None,
            "attempts": [],
            "stage_call_counts": {},
        }
        for case_id in values
    }

    for case_id in values:
        state: dict[str, Any] = {"case_id": case_id, "stage": "director"}
        stage = "director"
        total_attempts = 0
        retry_count = 0
        terminal = False
        # ``max_attempts`` is the shared retry budget.  Successful progression
        # through the five stages does not consume a retry; a failed stage
        # consumes one and is the only stage invoked again.
        while not terminal and retry_count <= max_attempts:
            total_attempts += 1
            callback = {
                "director": director,
                "codegen": codegen,
                "executor": executor,
                "observer": observer,
                "evaluator": evaluator,
            }.get(stage)
            if callback is None:
                raise ValueError(f"stage callback is not configured: {stage}")
            input_hashes = _stage_hashes(state)
            try:
                output = _invoke_stage_callback(callback, case_id, state, total_attempts)
            except Exception as exc:
                output = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "retryable": True,
                }
            cases[case_id]["stage_call_counts"][stage] = cases[case_id]["stage_call_counts"].get(stage, 0) + 1
            output_hashes = _stage_hashes(output)
            if _stage_success(output):
                state.update({key: value for key, value in output.items() if key not in {"status", "reason"}})
                state["stage"] = stage
                if stage == "director":
                    stage = "codegen"
                elif stage == "codegen":
                    stage = "executor"
                elif stage == "executor":
                    stage = "observer" if observer is not None else "evaluator"
                elif stage == "observer":
                    stage = "evaluator"
                elif stage == "evaluator":
                    cases[case_id]["status"] = "success"
                    cases[case_id]["selected_attempt"] = total_attempts
                    cases[case_id]["evaluation"] = output
                    terminal = True
                continue

            semantic = (
                stage == "evaluator"
                and str(output.get("execution_status") or "").casefold() == "valid"
                and str(output.get("semantic_status") or "").casefold()
                in {"failed_required_event", "semantic_failed", "uncertain", "semantic_uncertain"}
            )
            terminal_failure = bool(output.get("retryable") is False) and not semantic
            retry_stage = stage
            if semantic:
                status = "semantic_failed" if "failed" in str(output.get("semantic_status") or "").casefold() else "semantic_uncertain"
                cases[case_id]["status"] = status
                cases[case_id]["selected_attempt"] = total_attempts
                cases[case_id]["evaluation"] = output
                entry = {
                    "attempt": total_attempts,
                    "case_id": case_id,
                    "status": status,
                    "retry_stage": "outer_loop",
                    "parent_attempt_id": f"{case_id}:attempt-{max(0, total_attempts - 1):02d}",
                    "reason": str(output.get("reason") or status),
                    "input_hashes": input_hashes,
                    "output_hashes": output_hashes,
                    "detail": output,
                }
                cases[case_id]["attempts"].append(entry)
                _write_progress(progress, entry)
                terminal = True
                continue

            if terminal_failure:
                cases[case_id]["status"] = "terminal_failed"
                cases[case_id]["selected_attempt"] = total_attempts
                cases[case_id]["evaluation"] = output
                entry = {
                    "attempt": total_attempts,
                    "case_id": case_id,
                    "status": "terminal_failed",
                    "retry_stage": "outer_loop",
                    "parent_attempt_id": f"{case_id}:attempt-{max(0, total_attempts - 1):02d}",
                    "reason": str(output.get("reason") or f"{stage}_marked_non_retryable"),
                    "input_hashes": input_hashes,
                    "output_hashes": output_hashes,
                    "detail": output,
                }
                cases[case_id]["attempts"].append(entry)
                _write_progress(progress, entry)
                terminal = True
                continue

            entry = {
                "attempt": total_attempts,
                "case_id": case_id,
                "status": "retryable_failure",
                "retry_stage": retry_stage,
                "parent_attempt_id": f"{case_id}:attempt-{max(0, total_attempts - 1):02d}",
                "reason": str(output.get("reason") or output.get("status") or f"{stage}_failed"),
                "input_hashes": input_hashes,
                "output_hashes": output_hashes,
                "detail": output,
            }
            cases[case_id]["attempts"].append(entry)
            _write_progress(progress, entry)
            # Keep successful upstream state and retry only this stage.  The
            # callback may return replacement hashes on its next success.
            state["stage"] = stage
            retry_count += 1
        if not terminal:
            cases[case_id]["status"] = "exhausted"
            cases[case_id]["reason"] = "max_stage_retry_attempts_exhausted"

    result = {
        "status": "completed" if all(item["status"] in {"success", "semantic_failed", "semantic_uncertain"} for item in cases.values()) else "exhausted",
        "max_attempts": max_attempts,
        "case_count": len(values),
        "completed_count": sum(item["status"] == "success" for item in cases.values()),
        "exhausted_count": sum(item["status"] == "exhausted" for item in cases.values()),
        "progress_path": str(progress.resolve()),
        "cases": cases,
        "retry_policy": {
            "director_failure": "director",
            "codegen_failure": "codegen",
            "executor_failure": "executor",
            "observer_failure": "observer",
            "evaluator_transport_failure": "evaluator",
            "semantic_failure": "outer_loop_terminal",
        },
    }
    (root / "stage_retry_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = ["run_real_inner_loop", "run_stage_aware_inner_loop"]
