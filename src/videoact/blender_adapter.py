"""Controlled execution boundary for Blender CLI and MCP backends."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .contracts import ExecutionResult


class BlenderCliBackend:
    def __init__(self, blender_bin: str = "blender", runner: Callable[..., Any] | None = None):
        self.blender_bin = blender_bin
        self.runner = runner or subprocess.run

    def build_command(self, script_path: str | Path) -> list[str]:
        return [self.blender_bin, "-b", "--python", str(script_path)]

    def run(
        self,
        script_path: str | Path,
        run_dir: str | Path,
        timeout_s: float = 300,
    ) -> ExecutionResult:
        command = self.build_command(script_path)
        started = time.monotonic()
        try:
            completed = self.runner(
                command,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
                check=False,
            )
        except (subprocess.TimeoutExpired, TimeoutError) as exc:
            return ExecutionResult(
                status="timeout",
                backend="cli",
                command=command,
                duration_s=time.monotonic() - started,
                error=str(exc),
            )
        except OSError as exc:
            return ExecutionResult(
                status="failed",
                backend="cli",
                command=command,
                duration_s=time.monotonic() - started,
                error=str(exc),
            )

        return ExecutionResult(
            status="success" if completed.returncode == 0 else "failed",
            backend="cli",
            command=command,
            return_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_s=time.monotonic() - started,
            error=None if completed.returncode == 0 else "Blender CLI returned a non-zero exit code",
        )


class BlenderMcpBackend:
    def __init__(self, transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None):
        self.transport = transport

    def run(
        self,
        script_path: str | Path,
        run_dir: str | Path,
        timeout_s: float = 300,
    ) -> ExecutionResult:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        request = {"method": "execute_script", "script_path": str(script_path)}
        with (run_dir / "mcp_calls.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(request, sort_keys=True) + "\n")

        if self.transport is None:
            return ExecutionResult(
                status="failed",
                backend="mcp",
                request=request,
                error="MCP transport is not configured",
            )

        started = time.monotonic()
        try:
            response = self.transport(request)
        except TimeoutError as exc:
            return ExecutionResult(
                status="timeout",
                backend="mcp",
                request=request,
                duration_s=time.monotonic() - started,
                error=str(exc),
            )
        except Exception as exc:  # transport boundary must convert failures to data
            return ExecutionResult(
                status="failed",
                backend="mcp",
                request=request,
                duration_s=time.monotonic() - started,
                error=str(exc),
            )

        status = response.get("status", "failed")
        return ExecutionResult(
            status=status if status in {"success", "failed", "timeout"} else "failed",
            backend="mcp",
            request=request,
            artifact_paths=response.get("artifact_paths", {}),
            stdout=response.get("stdout", ""),
            stderr=response.get("stderr", ""),
            duration_s=time.monotonic() - started,
            error=response.get("error"),
        )


class BlenderAdapter:
    def __init__(self, *, cli: Any | None = None, mcp: Any | None = None):
        self.cli = cli or BlenderCliBackend()
        self.mcp = mcp or BlenderMcpBackend()

    def run(
        self,
        script_path: str | Path,
        run_dir: str | Path,
        *,
        prefer: str = "mcp",
        timeout_s: float = 300,
    ) -> ExecutionResult:
        if prefer not in {"mcp", "cli"}:
            raise ValueError("prefer must be 'mcp' or 'cli'")
        if prefer == "cli":
            return self.cli.run(script_path, run_dir, timeout_s=timeout_s)

        mcp_result = self.mcp.run(script_path, run_dir, timeout_s=timeout_s)
        if mcp_result.status == "success":
            return mcp_result
        cli_result = self.cli.run(script_path, run_dir, timeout_s=timeout_s)
        return cli_result.model_copy(update={"fallback_used": True})
