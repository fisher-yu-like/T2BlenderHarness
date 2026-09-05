"""Render immutable Blender proxy jobs in parallel with Blender CLI.

Each case has an isolated working directory.  The script never merges cases,
so one failed render cannot overwrite another case's artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.real_pipeline import RealRunStateMachine  # noqa: E402
from videoact.real_artifacts import probe_mp4  # noqa: E402
from videoact.real_video import assemble_mp4_from_pngs  # noqa: E402
from videoact.observer_contract import sha256_file, write_observer_request  # noqa: E402


def build_blender_command(job_dir: str | Path, blender_bin: str) -> list[str]:
    root = Path(job_dir).resolve()
    # Blender's embedded Python on Windows does not honor the parent process'
    # PYTHONPATH.  Inject the verified project root before loading the frozen
    # job; this is executor wiring, not a scene/codegen fallback.
    project_root_literal = repr(str(ROOT))
    path_bootstrap = f"import sys; sys.path.insert(0, {project_root_literal})"
    return [
        blender_bin,
        "-b",
        "--python-expr",
        path_bootstrap,
        "--python",
        str((root / "blender_job.py").resolve()),
    ]


def classify_render_status(return_code: int, render_artifacts_ready: bool) -> str:
    return "success" if return_code == 0 and render_artifacts_ready else "failed"


def _extract_blender_version(*outputs: str | None) -> str | None:
    """Extract Blender's semantic version from noisy CLI output."""

    pattern = re.compile(r"\bBlender\s+(\d+(?:\.\d+){1,3})\b", re.IGNORECASE)
    for output in outputs:
        if not isinstance(output, str):
            continue
        for line in output.splitlines():
            match = pattern.search(line)
            if match:
                return match.group(1)
    return None


def mark_render_state(job_dir: str | Path, *, return_code: int, blender_version: str | None = None) -> None:
    """Persist the CLI outcome in the same state machine used by real evaluation."""
    root = Path(job_dir)
    machine = RealRunStateMachine(root, case_id=root.name)
    if machine.state in {"evaluated", "failed"}:
        machine.transition("executing", {"backend": "blender-cli", "explicit_rerender": True})
    if machine.state == "prepared":
        machine.transition("executing", {"backend": "blender-cli"})
    response = {
        "status": "success" if return_code == 0 else "failed",
        "return_code": return_code,
        "backend": "blender-cli",
    }
    if blender_version:
        response["blender_version"] = blender_version
        manifest_path = root / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(manifest_payload, dict):
                    manifest_payload["blender_version"] = str(blender_version).strip()
                    manifest_path.write_text(
                        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                # The state record still preserves the CLI evidence; the
                # final ExperimentFingerprint gate will fail closed if the
                # manifest cannot be updated.
                pass
    machine.record_mcp_response(response)


def _clear_render_outputs(job_dir: Path) -> None:
    """Clear only generated render outputs before a retry; keep immutable inputs."""
    for relative in (
        "proxy.blend",
        "candidate.blend",
        "proxy.mp4",
        "telemetry.json",
        "telemetry_manifest.json",
        "observer_request.json",
        "untrusted_candidate_telemetry.json",
    ):
        path = job_dir / relative
        if path.is_file():
            path.unlink()
    frames_dir = job_dir / "frames"
    if frames_dir.is_dir():
        shutil.rmtree(frames_dir)


def _render_artifacts_ready(job_dir: Path, *, trusted_observer_required: bool = False) -> bool:
    required = [
        job_dir / "proxy.blend",
        job_dir / "telemetry.json",
        job_dir / "frames" / "index.json",
    ]
    if trusted_observer_required:
        required.extend(
            [
                job_dir / "candidate.blend",
                job_dir / "observer_request.json",
                job_dir / "telemetry_manifest.json",
            ]
        )
    animation_frames = sorted((job_dir / "frames" / "animation").glob("frame_*.png"))
    return all(path.is_file() and path.stat().st_size > 0 for path in required) and bool(animation_frames)


def _observer_command(job_dir: Path, blender_bin: str, request_path: Path, observer_source_hash: str) -> list[str]:
    return [
        blender_bin,
        "-b",
        str((job_dir / "candidate.blend").resolve()),
        "--python",
        str(OBSERVER_SOURCE.resolve()),
        "--",
        "--run-dir",
        str(job_dir.resolve()),
        "--request",
        str(request_path.resolve()),
        "--observer-source-sha256",
        observer_source_hash,
    ]


OBSERVER_SOURCE = ROOT / "blender" / "trusted_observer.py"


def _mesh_entity_ids_for_observer(job_dir: Path) -> list[str]:
    """Request mesh narrow-phase data only for physical scene entities."""

    contract_path = job_dir / "scene_contract.json"
    if not contract_path.is_file():
        return []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    entity_ids: list[str] = []
    for entity in contract.get("entities", []) if isinstance(contract, dict) else []:
        if not isinstance(entity, dict):
            continue
        kind = str(entity.get("kind") or "").lower()
        entity_id = str(entity.get("id") or "").strip()
        if entity_id and kind in {"actor", "character", "prop", "support", "environment"}:
            entity_ids.append(entity_id)
    return list(dict.fromkeys(entity_ids))


def _run_trusted_observer(
    job_dir: Path,
    *,
    blender_bin: str,
    timeout_s: int,
    manifest_payload: dict[str, Any],
) -> dict[str, Any]:
    """Run the fixed observer after quarantining all generated telemetry."""

    candidate = job_dir / "candidate.blend"
    proxy = job_dir / "proxy.blend"
    if not candidate.is_file():
        return {"status": "failed", "error": "candidate_blend_missing"}
    if proxy.is_file() and sha256_file(proxy) != sha256_file(candidate):
        return {"status": "failed", "error": "candidate_proxy_blend_hash_mismatch"}
    if not proxy.is_file():
        shutil.copy2(candidate, proxy)
    observer_hash = sha256_file(OBSERVER_SOURCE)
    declared_hash = str(manifest_payload.get("observer_source_hash") or "")
    if declared_hash and declared_hash != observer_hash:
        return {"status": "failed", "error": "observer_source_hash_not_allowlisted"}
    for relative in ("proxy.mp4", "telemetry_manifest.json", "observer_request.json"):
        path = job_dir / relative
        if path.is_file():
            path.unlink()
    generated_telemetry = job_dir / "telemetry.json"
    if generated_telemetry.is_file():
        generated_telemetry.replace(job_dir / "untrusted_candidate_telemetry.json")
    frames_dir = job_dir / "frames"
    if frames_dir.is_dir():
        shutil.rmtree(frames_dir)
    request_path = job_dir / "observer_request.json"
    request = write_observer_request(
        request_path,
        candidate_blend_hash=sha256_file(candidate),
        observer_source_hash=observer_hash,
        mesh_entity_ids=_mesh_entity_ids_for_observer(job_dir),
        obligation_ids=[
            str(item)
            for item in manifest_payload.get("obligation_ids", [])
            if isinstance(item, str) and item.strip()
        ],
    )
    command = _observer_command(job_dir, blender_bin, request_path, observer_hash)
    inherited_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath = os.pathsep.join(item for item in (str(ROOT), inherited_pythonpath) if item)
    try:
        completed = subprocess.run(
            command,
            cwd=str(job_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
            env={**os.environ, "PYTHONPATH": pythonpath},
        )
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "error": f"observer_timeout:{exc}", "command": command, "request": request}
    except OSError as exc:
        return {"status": "failed", "error": f"observer_oserror:{exc}", "command": command, "request": request}
    return {
        "status": "success" if completed.returncode == 0 else "failed",
        "return_code": completed.returncode,
        "command": command,
        "request": request,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
        "error": None if completed.returncode == 0 else "trusted_observer_nonzero_exit",
    }


def _assemble_and_probe_video(job_dir: Path) -> dict[str, Any]:
    animation_frames = sorted((job_dir / "frames" / "animation").glob("frame_*.png"))
    if animation_frames and not (job_dir / "proxy.mp4").is_file():
        try:
            fps = 24
            manifest_path = job_dir / "run_manifest.json"
            if manifest_path.is_file():
                try:
                    fps = int(json.loads(manifest_path.read_text(encoding="utf-8")).get("fps") or fps)
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            assemble_mp4_from_pngs(animation_frames, job_dir / "proxy.mp4", fps=fps)
        except (OSError, RuntimeError, ValueError) as exc:
            return {"playable": False, "error": f"assembly:{type(exc).__name__}:{exc}"}
    return probe_mp4(job_dir / "proxy.mp4", minimum_frames=3)


def _run_one(
    job_dir: Path,
    blender_bin: str,
    timeout_s: int,
    max_retries: int = 2,
    rollout_seed: int | None = None,
) -> dict[str, Any]:
    job_dir = job_dir.resolve()
    job_path = job_dir / "blender_job.py"
    if not job_path.is_file():
        return {"case_id": job_dir.name, "status": "missing_job", "job_dir": str(job_dir.resolve())}
    existing_state = RealRunStateMachine(job_dir, case_id=job_dir.name).state
    if existing_state in {"evaluated", "failed"}:
        _clear_render_outputs(job_dir)
    command = build_blender_command(job_dir, blender_bin)
    frozen_source_hash = hashlib.sha256(job_path.read_bytes()).hexdigest()
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    expected_manifest_code_hash = None
    manifest_payload: dict[str, Any] = {}
    manifest_path = job_dir / "run_manifest.json"
    if manifest_path.is_file():
        try:
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest_payload, dict):
                manifest_payload = {}
            expected_manifest_code_hash = manifest_payload.get("code_hash")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            expected_manifest_code_hash = None
            manifest_payload = {}
    trusted_observer_required = bool(manifest_payload.get("trusted_observer_required"))
    if expected_manifest_code_hash and expected_manifest_code_hash != frozen_source_hash:
        return {
            "case_id": job_dir.name,
            "status": "failed",
            "job_dir": str(job_dir.resolve()),
            "return_code": None,
            "render_artifacts_ready": False,
            "video_probe": {"playable": False, "error": "manifest_code_hash_mismatch_before_render"},
            "retry_count": 0,
            "attempts": [],
            "proxy_video": None,
            "job_source_hash": frozen_source_hash,
        }
    attempts: list[dict[str, Any]] = []
    for attempt_number in range(1, max_retries + 2):
        started = time.monotonic()
        if attempt_number > 1:
            _clear_render_outputs(job_dir)
        completed = None
        error = None
        try:
            # Agent-generated jobs import the verified ``blender.lib`` package
            # from the project.  Blender starts with the case directory on
            # sys.path, so the executor must expose the project root
            # explicitly; this is runtime wiring, not generated scene code.
            inherited_pythonpath = os.environ.get("PYTHONPATH", "")
            pythonpath = os.pathsep.join(
                item for item in (str(ROOT), inherited_pythonpath) if item
            )
            environment = {**os.environ, "PYTHONPATH": pythonpath}
            if rollout_seed is not None:
                environment["T2BLENDER_ROLLOUT_SEED"] = str(int(rollout_seed))
            completed = subprocess.run(
                command,
                cwd=str(job_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"timeout:{exc}"
        except OSError as exc:
            error = f"oserror:{exc}"
        observer_result: dict[str, Any] | None = None
        if trusted_observer_required and completed is not None and completed.returncode == 0:
            observer_result = _run_trusted_observer(
                job_dir,
                blender_bin=blender_bin,
                timeout_s=timeout_s,
                manifest_payload=manifest_payload,
            )
            if observer_result.get("status") != "success":
                error = (error + ";" if error else "") + str(observer_result.get("error") or "trusted_observer_failed")
        render_artifacts_ready = _render_artifacts_ready(
            job_dir,
            trusted_observer_required=trusted_observer_required,
        )
        current_source_hash = hashlib.sha256(job_path.read_bytes()).hexdigest() if job_path.is_file() else None
        source_unchanged = current_source_hash == frozen_source_hash
        if not source_unchanged:
            error = (error + ";" if error else "") + "job_source_changed_during_render"
        manifest_code_hash = None
        if manifest_path.is_file():
            try:
                manifest_code_hash = json.loads(
                    manifest_path.read_text(encoding="utf-8")
                ).get("code_hash")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                manifest_code_hash = None
        if expected_manifest_code_hash and manifest_code_hash != frozen_source_hash:
            error = (error + ";" if error else "") + "manifest_code_hash_mismatch"
        video_probe = _assemble_and_probe_video(job_dir) if render_artifacts_ready else {
            "playable": False,
            "error": "render_artifacts_incomplete",
        }
        return_code = completed.returncode if completed is not None else None
        attempt_result = {
            "attempt": attempt_number,
            "return_code": return_code,
            "render_artifacts_ready": render_artifacts_ready,
            "video_probe": video_probe,
            "duration_s": time.monotonic() - started,
            "stdout_tail": (completed.stdout or "")[-2000:] if completed is not None else "",
            "stderr_tail": (completed.stderr or "")[-2000:] if completed is not None else "",
            "error": error,
            "source_hash": current_source_hash,
            "source_unchanged": source_unchanged,
            "manifest_code_hash": manifest_code_hash,
            "trusted_observer_required": trusted_observer_required,
            "trusted_observer": observer_result,
        }
        attempts.append(attempt_result)
        if (
            source_unchanged
            and return_code == 0
            and render_artifacts_ready
            and video_probe.get("playable")
            and (not trusted_observer_required or (observer_result or {}).get("status") == "success")
            and (not expected_manifest_code_hash or manifest_code_hash == frozen_source_hash)
        ):
            result = {
                "case_id": job_dir.name,
                "status": "success",
                "job_dir": str(job_dir.resolve()),
                "command": command,
                "return_code": return_code,
                "render_artifacts_ready": True,
                "video_probe": video_probe,
                "retry_count": attempt_number - 1,
                "attempts": attempts,
                "proxy_video": str((job_dir / "proxy.mp4").resolve()),
                "job_source_hash": frozen_source_hash,
            }
            blender_version = _extract_blender_version(
                completed.stdout,
                completed.stderr,
                (observer_result or {}).get("stdout_tail"),
                (observer_result or {}).get("stderr_tail"),
            )
            mark_render_state(job_dir, return_code=0, blender_version=blender_version)
            (job_dir / "render_attempts.json").write_text(json.dumps(attempts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return result
        if not source_unchanged:
            break
    last = attempts[-1]
    result = {
        "case_id": job_dir.name,
        "status": "timeout" if str(last.get("error", "")).startswith("timeout:") else "failed",
        "job_dir": str(job_dir.resolve()),
        "command": command,
        "return_code": last.get("return_code"),
        "render_artifacts_ready": last.get("render_artifacts_ready", False),
        "video_probe": last.get("video_probe", {}),
        "retry_count": len(attempts) - 1,
        "attempts": attempts,
        "proxy_video": str((job_dir / "proxy.mp4").resolve()) if (job_dir / "proxy.mp4").exists() else None,
        "job_source_hash": frozen_source_hash,
    }
    mark_render_state(job_dir, return_code=1, blender_version="")
    (job_dir / "render_attempts.json").write_text(json.dumps(attempts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def render_jobs(
    run_root: str | Path,
    *,
    blender_bin: str = "blender",
    workers: int | None = None,
    timeout_s: int = 900,
    max_retries: int = 2,
    rollout_seed: int | None = None,
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    root = Path(run_root)
    all_job_dirs = sorted(path.parent for path in root.glob("*/blender_job.py"))
    if case_ids is None:
        job_dirs = all_job_dirs
    else:
        requested = [str(case_id) for case_id in case_ids]
        if len(requested) != len(set(requested)):
            raise ValueError("case_ids must be unique")
        by_id = {path.name: path for path in all_job_dirs}
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"requested case IDs have no frozen Blender job: {unknown}")
        job_dirs = [by_id[case_id] for case_id in requested]
    workers = min(12, max(1, workers or min(4, os.cpu_count() or 1)))
    resolved_blender = shutil.which(blender_bin) or (blender_bin if Path(blender_bin).is_file() else None)
    if resolved_blender is None:
        report = {
            "status": "unavailable",
            "reason": f"Blender CLI not found: {blender_bin}",
            "run_root": str(root.resolve()),
            "job_count": len(job_dirs),
            "workers": workers,
            "rollout_seed": rollout_seed,
            "selected_case_ids": [path.name for path in job_dirs],
            "results": [
                {"case_id": job_dir.name, "status": "not_started", "job_dir": str(job_dir.resolve())}
                for job_dir in job_dirs
            ],
        }
    else:
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, job_dir, resolved_blender, timeout_s, max_retries, rollout_seed): job_dir.name
                for job_dir in job_dirs
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["case_id"])
        report = {
            "status": "completed",
            "blender_bin": resolved_blender,
            "run_root": str(root.resolve()),
            "job_count": len(job_dirs),
            "workers": workers,
            "max_render_retries": max_retries,
            "rollout_seed": rollout_seed,
            "selected_case_ids": [path.name for path in job_dirs],
            "success_count": sum(item["status"] == "success" for item in results),
            "results": results,
        }
    (root / "cli_render_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--blender-bin", default="blender")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--max-render-retries", type=int, default=2)
    parser.add_argument("--rollout-seed", type=int)
    parser.add_argument("--case-id", action="append", default=None, help="render only this frozen case; repeat for a serial group")
    args = parser.parse_args()
    report = render_jobs(
        args.run_root,
        blender_bin=args.blender_bin,
        workers=args.workers,
        timeout_s=args.timeout_s,
        max_retries=args.max_render_retries,
        rollout_seed=args.rollout_seed,
        case_ids=args.case_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"completed", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
