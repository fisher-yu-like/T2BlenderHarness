"""Assemble and verify the immutable G0--G3 formal-training release gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from videoact.release_gates import (
    build_formal_release_report,
    validate_formal_release_report,
)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--g4", "--generalization", dest="g4", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = build_formal_release_report(
        _read(args.g0),
        _read(args.g1),
        _read(args.pilot),
        _read(args.shadow),
        _read(args.g4) if args.g4 is not None else None,
    )
    verification = validate_formal_release_report(result)
    # Do not embed the outer report hash inside the report that is about to be
    # resealed: doing so would leave a stale self-reference.  Preserve the
    # hash that was actually verified under an explicitly named field.
    verification_for_report = dict(verification)
    verified_input_hash = verification_for_report.pop("report_hash", None)
    if verified_input_hash:
        verification_for_report["verified_input_report_hash"] = verified_input_hash
    result["verification"] = verification_for_report
    from videoact.release_gates import seal_report

    result = seal_report(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if verification.get("training_allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
