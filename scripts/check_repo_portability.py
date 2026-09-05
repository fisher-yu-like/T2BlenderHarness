"""Fail-closed portability checks for the checked-in benchmark contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "dataset" / "vbench2-agent-training-index-v1" / "metadata.json"
SCANNED_ROOTS = (ROOT / "dataset" / "vbench2-agent-training-index-v1",)
SCANNED_FILES = (ROOT / "pyproject.toml", ROOT / ".github" / "workflows" / "ci.yml")
ABSOLUTE_WINDOWS = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:(?:/|\\{1,2})")


def _metadata_failures() -> list[str]:
    failures: list[str] = []
    try:
        payload = json.loads(METADATA.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return [f"metadata_unreadable:{type(exc).__name__}"]
    source_path = payload.get("source_path")
    if not isinstance(source_path, str) or Path(source_path).is_absolute() or "\\" in source_path:
        failures.append("metadata_source_path_not_repo_relative_posix")
    elif not (ROOT / source_path).is_file():
        failures.append(f"metadata_source_missing:{source_path}")
    reference_roots = payload.get("reference_roots")
    if not isinstance(reference_roots, list) or not reference_roots:
        failures.append("metadata_reference_roots_missing")
    else:
        for reference_root in reference_roots:
            if not isinstance(reference_root, str) or Path(reference_root).is_absolute() or "\\" in reference_root:
                failures.append(f"metadata_reference_root_not_repo_relative_posix:{reference_root}")
            elif not (ROOT / reference_root).is_dir():
                failures.append(f"metadata_reference_root_missing:{reference_root}")
    return failures


def _text_failures() -> list[str]:
    failures: list[str] = []
    paths: list[Path] = list(SCANNED_FILES)
    for scanned_root in SCANNED_ROOTS:
        if not scanned_root.is_dir():
            failures.append(f"missing_scanned_root:{scanned_root.relative_to(ROOT)}")
            continue
        paths.extend(path for path in scanned_root.rglob("*") if path.is_file())
    for path in paths:
        if path.suffix.lower() not in {".py", ".json", ".jsonl", ".toml", ".md", ".yml", ".yaml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if ABSOLUTE_WINDOWS.search(text):
            failures.append(f"absolute_windows_path:{path.relative_to(ROOT)}")
    return failures


def main() -> int:
    failures = _metadata_failures() + _text_failures()
    report = {"status": "pass" if not failures else "fail", "failures": failures}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
