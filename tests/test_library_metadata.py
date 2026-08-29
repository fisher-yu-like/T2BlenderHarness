from __future__ import annotations

import json
import subprocess
import sys

from blender.lib.__meta__ import collect_library_signatures
from scripts.export_library_signatures import export_library_signatures


def test_library_signatures_cover_all_public_categories() -> None:
    payload = collect_library_signatures()

    assert len(payload["geometry"]) >= 8
    assert len(payload["rigging"]) >= 3
    assert len(payload["constraints"]) >= 2
    assert len(payload["camera"]) >= 4
    assert len(payload["layout"]) >= 3
    for category in payload.values():
        for item in category:
            assert item["name"]
            assert item["signature"]
            assert item["docstring"]
            assert item["tags"]
            assert item["cost_estimate"]
            assert item["example_usage"]
            assert item["usage_count"] == 0


def test_exported_signatures_are_valid_json(tmp_path) -> None:
    path = export_library_signatures(tmp_path / "signatures.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["geometry"]
    assert path.is_file()


def test_cli_exports_signatures_in_a_fresh_process(tmp_path) -> None:
    destination = tmp_path / "fresh-signatures.json"
    completed = subprocess.run(
        [sys.executable, "scripts/export_library_signatures.py", "--output", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(destination.read_text(encoding="utf-8"))["camera"]
