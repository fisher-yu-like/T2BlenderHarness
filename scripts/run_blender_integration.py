"""Run Blender-backed integration tests with an explicitly selected binary."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASETEMP = ROOT / ".pytest-blender-integration"


def _with_local_basetemp(pytest_args: Sequence[str]) -> list[str]:
    args = list(pytest_args)
    if "--basetemp" not in args and not any(arg.startswith("--basetemp=") for arg in args):
        args[0:0] = ["--basetemp", str(DEFAULT_BASETEMP)]
    return args


def build_command(blender_bin: str, pytest_args: Sequence[str] = ()) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    environment["VIDEOACT_BLENDER_BIN"] = blender_bin
    return (
        [
            sys.executable,
            "-m",
            "pytest",
            "-m",
            "blender_integration",
            *_with_local_basetemp(pytest_args),
        ],
        environment,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        binary_index = arguments.index("--blender-bin")
        blender_bin = arguments.pop(binary_index + 1)
        arguments.pop(binary_index)
    except (IndexError, ValueError) as exc:
        raise SystemExit("--blender-bin <path> is required") from exc
    command, environment = build_command(blender_bin, arguments)
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
