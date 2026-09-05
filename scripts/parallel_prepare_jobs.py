"""Prepare independent dynamic-agent case chunks in parallel.

The GLM Director and Blender-code calls are independent across cases, but a
single ``prepare_jobs`` invocation owns mutable provider and cache state.  This
module runs isolated child invocations and merges only their immutable case
artifacts.  It is deliberately limited to the model-backed preparation path;
the explicit template baseline never enters this helper.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_real_jobs.py"


def _chunk(values: list[str], count: int) -> list[list[str]]:
    if not values:
        return []
    size = max(1, math.ceil(len(values) / max(1, count)))
    return [values[start : start + size] for start in range(0, len(values), size)]


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _rewrite_job_paths(job: dict[str, Any], *, chunk_root: Path, output_root: Path) -> dict[str, Any]:
    """Rebind absolute child paths to the merged case root."""

    result = dict(job)
    for field in ("run_dir", "job_path", "provider_manifest_path", "failure_path"):
        raw = result.get(field)
        if not raw:
            continue
        old = Path(str(raw))
        if not old.is_absolute():
            old = (chunk_root / old).resolve()
        else:
            old = old.resolve()
        if not _path_inside(old, chunk_root):
            # Keep an auditable failure rather than silently pointing at a
            # location outside the isolated preparation workspace.
            raise ValueError(f"child job path escapes chunk root: {field}={old}")
        relative = old.relative_to(chunk_root.resolve())
        result[field] = str((output_root / relative).resolve())
    return result


def _worker_command(
    *,
    split: str,
    chunk_root: Path,
    dataset_root: Path,
    harness_version: str,
    evaluator_version: str,
    case_ids: list[str],
    code_cache_dir: Path,
    codegen_examples_root: str | Path | None,
    codex_command: str,
    provider_timeout_s: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PREPARE_SCRIPT),
        "--split",
        split,
        "--out-dir",
        str(chunk_root),
        "--dataset-root",
        str(dataset_root),
        "--harness-version",
        harness_version,
        "--evaluator-version",
        evaluator_version,
        "--generation-mode",
        "agent",
        "--provider-mode",
        "glm",
        "--code-cache-dir",
        str(code_cache_dir),
        "--codex-command",
        codex_command,
        "--timeout-s",
        str(provider_timeout_s),
    ]
    if codegen_examples_root is not None:
        command.extend(["--codegen-examples-root", str(Path(codegen_examples_root).resolve())])
    for case_id in case_ids:
        command.extend(["--case-id", str(case_id)])
    return command


def _run_chunk(
    *,
    command: list[str],
    chunk_root: Path,
    case_ids: list[str],
    timeout_s: int,
) -> dict[str, Any]:
    chunk_root.mkdir(parents=True, exist_ok=True)
    stdout_path = chunk_root / "prepare.stdout.log"
    stderr_path = chunk_root / "prepare.stderr.log"
    status = "completed"
    reason = None
    return_code: int | None = None
    try:
        with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as stderr:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_s,
            )
        return_code = int(completed.returncode)
        if return_code != 0:
            status = "worker_failed"
            reason = f"prepare worker exited with code {return_code}"
    except subprocess.TimeoutExpired:
        status = "worker_timeout"
        reason = f"prepare worker exceeded {timeout_s}s"
    except Exception as exc:  # pragma: no cover - defensive process boundary
        status = "worker_failed"
        reason = f"{type(exc).__name__}: {exc}"

    index_path = chunk_root / "job_index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            index = None
            status = "worker_failed"
            reason = f"child job index unreadable: {type(exc).__name__}: {exc}"
    else:
        index = None
    if not isinstance(index, dict):
        jobs = [
            {
                "case_id": case_id,
                "run_dir": str((chunk_root / case_id).resolve()),
                "generation_mode": "agent",
                "provider_mode": "glm",
                "status": status,
                "failure_path": None,
                "reason": reason or "child job index missing",
            }
            for case_id in case_ids
        ]
        index = {
            "split": None,
            "generation_mode": "agent",
            "provider_mode": "glm",
            "case_count": len(jobs),
            "prepared_count": 0,
            "failed_count": len(jobs),
            "jobs": jobs,
        }
    return {
        "status": status,
        "reason": reason,
        "return_code": return_code,
        "case_ids": case_ids,
        "root": str(chunk_root.resolve()),
        "index": index,
    }


def prepare_jobs_parallel(
    split: str,
    out_dir: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    case_ids: list[str],
    code_cache_dir: str | Path,
    codegen_examples_root: str | Path | None = None,
    codex_command: str = "codex",
    provider_timeout_s: int = 180,
    workers: int = 4,
    attempt: int = 1,
) -> dict[str, Any]:
    """Prepare a GLM agent batch using isolated child processes.

    Every child receives explicit case IDs and has its own cache/provider
    instances.  The returned index has the same shape as ``prepare_jobs`` so
    the existing provenance and inner-loop gates remain authoritative.
    """

    if split not in {"calibration", "train", "dev", "test"}:
        raise ValueError("split must be calibration, train, dev, or test")
    if not case_ids:
        raise ValueError("case_ids must not be empty")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_ids must be unique")
    if not isinstance(workers, int) or isinstance(workers, bool) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between 1 and 8")
    if not isinstance(provider_timeout_s, int) or provider_timeout_s <= 0:
        raise ValueError("provider_timeout_s must be a positive integer")

    output_root = Path(out_dir).resolve()
    dataset = Path(dataset_root).resolve()
    cache_root = Path(code_cache_dir).resolve()
    parallel_root = output_root / ".parallel_prepare" / f"attempt-{int(attempt):02d}"
    if parallel_root.exists() and any(parallel_root.iterdir()):
        raise FileExistsError(f"parallel preparation workspace already contains artifacts: {parallel_root}")
    parallel_root.mkdir(parents=True, exist_ok=True)

    chunks = _chunk([str(case_id) for case_id in case_ids], min(workers, len(case_ids)))
    jobs_by_id: dict[str, dict[str, Any]] = {}
    child_results: list[dict[str, Any]] = []
    child_timeout_s = max(1800, provider_timeout_s * max(12, len(case_ids) * 4))
    with ThreadPoolExecutor(max_workers=len(chunks), thread_name_prefix="glm-prepare") as pool:
        futures = []
        for index, chunk_ids in enumerate(chunks, start=1):
            chunk_root = parallel_root / f"chunk-{index:02d}"
            command = _worker_command(
                split=split,
                chunk_root=chunk_root,
                dataset_root=dataset,
                harness_version=harness_version,
                evaluator_version=evaluator_version,
                case_ids=chunk_ids,
                code_cache_dir=cache_root / f"attempt-{int(attempt):02d}" / f"chunk-{index:02d}",
                codegen_examples_root=codegen_examples_root,
                codex_command=codex_command,
                provider_timeout_s=provider_timeout_s,
            )
            futures.append(
                pool.submit(
                    _run_chunk,
                    command=command,
                    chunk_root=chunk_root,
                    case_ids=chunk_ids,
                    timeout_s=child_timeout_s,
                )
            )
        for future in as_completed(futures):
            child_results.append(future.result())

    child_results.sort(key=lambda item: str(item.get("root")))
    for child in child_results:
        chunk_root = Path(str(child["root"])).resolve()
        index = child.get("index") or {}
        raw_jobs = index.get("jobs") if isinstance(index, dict) else None
        raw_jobs = raw_jobs if isinstance(raw_jobs, list) else []
        jobs_in_child = {str(job.get("case_id")): job for job in raw_jobs if isinstance(job, dict)}
        for case_id in child["case_ids"]:
            job = jobs_in_child.get(str(case_id))
            if job is None:
                job = {
                    "case_id": str(case_id),
                    "run_dir": str((chunk_root / str(case_id)).resolve()),
                    "generation_mode": "agent",
                    "provider_mode": "glm",
                    "status": child.get("status") or "worker_failed",
                    "failure_path": None,
                    "reason": child.get("reason") or "child job omitted case",
                }
            old_case_root = Path(str(job.get("run_dir") or (chunk_root / str(case_id))) ).resolve()
            if old_case_root.exists():
                if not _path_inside(old_case_root, chunk_root):
                    raise ValueError(f"child case directory escapes chunk root: {old_case_root}")
                destination = output_root / str(case_id)
                if destination.exists():
                    raise FileExistsError(f"merged run directory already exists: {destination}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_case_root), str(destination))
            jobs_by_id[str(case_id)] = _rewrite_job_paths(job, chunk_root=chunk_root, output_root=output_root)

    missing = [str(case_id) for case_id in case_ids if str(case_id) not in jobs_by_id]
    if missing:
        raise RuntimeError(f"parallel preparation lost case records: {missing}")
    jobs = [jobs_by_id[str(case_id)] for case_id in case_ids]
    index = {
        "split": split,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "generation_mode": "agent",
        "provider_mode": "glm",
        "case_count": len(jobs),
        "prepared_count": sum(job.get("status") == "prepared" for job in jobs),
        "failed_count": sum(job.get("status") != "prepared" for job in jobs),
        "jobs": jobs,
        "parallel_preparation": {
            "status": "completed",
            "workers": len(chunks),
            "attempt": int(attempt),
            "workspace": str(parallel_root.resolve()),
            "children": child_results,
            "template_backed": False,
            "llm_generated": True,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "job_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


__all__ = ["prepare_jobs_parallel"]
