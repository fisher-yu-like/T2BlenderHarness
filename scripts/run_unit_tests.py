"""Run the cross-platform unit suite with the active Python interpreter."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASETEMP = ROOT / ".pytest-unit"


def _with_local_basetemp(pytest_args: Sequence[str]) -> list[str]:
    args = list(pytest_args)
    if "--basetemp" not in args and not any(arg.startswith("--basetemp=") for arg in args):
        args[0:0] = ["--basetemp", str(DEFAULT_BASETEMP)]
    return args


def build_command(pytest_args: Sequence[str] = ()) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not blender_integration",
        *_with_local_basetemp(pytest_args),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    pytest_args = sys.argv[1:] if argv is None else list(argv)
    return subprocess.run(build_command(pytest_args), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
