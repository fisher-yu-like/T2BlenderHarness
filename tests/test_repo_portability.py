from __future__ import annotations

import json
import sys
from pathlib import Path


def test_portability_checker_detects_json_escaped_windows_absolute_path(tmp_path: Path, monkeypatch) -> None:
    import scripts.check_repo_portability as checker

    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps({"source_path": "C:" + r"\Users\author\Desktop\T2BlenderCode\source.json"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "SCANNED_ROOTS", (tmp_path,))
    monkeypatch.setattr(checker, "SCANNED_FILES", ())

    assert checker._text_failures() == ["absolute_windows_path:metadata.json"]


def test_active_benchmark_metadata_paths_are_repo_relative_posix() -> None:
    root = Path(__file__).resolve().parents[1]
    metadata = json.loads(
        (root / "dataset" / "vbench2-agent-training-index-v1" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    paths = [metadata["source_path"], *metadata["reference_roots"]]
    assert all(not Path(path).is_absolute() and "\\" not in path for path in paths)
    assert all((root / path).exists() for path in paths)


def test_portability_checker_ignores_historical_experiment_evidence() -> None:
    import scripts.check_repo_portability as checker

    assert checker._text_failures() == []


def test_unit_test_entry_uses_running_python_and_excludes_blender_integration() -> None:
    from scripts.run_unit_tests import build_command

    command = build_command(["-q"])
    assert command[:5] == [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "not blender_integration",
    ]
    assert command[-1] == "-q"
    assert command[5] == "--basetemp"


def test_unit_test_entry_respects_explicit_basetemp() -> None:
    from scripts.run_unit_tests import build_command

    command = build_command(["--basetemp", "custom-temp", "-q"])
    assert command.count("--basetemp") == 1
    assert command[command.index("--basetemp") + 1] == "custom-temp"


def test_unit_test_entry_forwards_leading_pytest_flags(monkeypatch) -> None:
    import scripts.run_unit_tests as unit_tests

    observed = {}
    monkeypatch.setattr(
        unit_tests.subprocess,
        "run",
        lambda command, **kwargs: observed.update(command=command, **kwargs)
        or type("Completed", (), {"returncode": 0})(),
    )

    assert unit_tests.main(["-q"]) == 0
    assert observed["command"][-1] == "-q"


def test_blender_integration_entry_uses_running_python_and_explicit_binary() -> None:
    from scripts.run_blender_integration import build_command

    command, environment = build_command("C:/Blender/blender.exe", ["-q"])

    assert command[:5] == [
        sys.executable,
        "-m",
        "pytest",
        "-m",
        "blender_integration",
    ]
    assert command[5] == "--basetemp"
    assert command[-1] == "-q"
    assert environment["VIDEOACT_BLENDER_BIN"] == "C:/Blender/blender.exe"
