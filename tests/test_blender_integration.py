from __future__ import annotations

import os
import subprocess

import pytest


@pytest.mark.blender_integration
def test_blender_binary_starts_in_factory_background_mode() -> None:
    blender_bin = os.environ["VIDEOACT_BLENDER_BIN"]
    completed = subprocess.run(
        [
            blender_bin,
            "--background",
            "--factory-startup",
            "--python-expr",
            "import bpy; print(bpy.app.version_string)",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
