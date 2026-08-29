"""Run an explicit labeled audit of semantic source-reuse detection."""

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

from videoact.source_fingerprints import evaluate_fingerprint_pairs


def builtin_pairs() -> list[dict[str, Any]]:
    """Return 100 transparent, hand-authored detector audit pairs.

    The first half varies only literals/identifiers and is labeled as reuse;
    the second half changes control flow and is labeled as genuinely
    different.  This is a regression fixture, not training data.
    """

    pairs: list[dict[str, Any]] = []
    for index in range(50):
        pairs.append(
            {
                "pair_id": f"reuse-{index:03d}",
                "first": "import bpy\nvalue = 1\nbpy.ops.mesh.primitive_cube_add()\n",
                "second": f"import bpy\nvalue = {index + 2}\nbpy.ops.mesh.primitive_cube_add()\n",
                "expected_template_reuse": True,
            }
        )
    for index in range(50):
        pairs.append(
            {
                "pair_id": f"different-{index:03d}",
                "first": "import bpy\nfor _ in range(2):\n    bpy.ops.mesh.primitive_cube_add()\n",
                "second": "import bpy\nif True:\n    bpy.ops.mesh.primitive_cube_add()\n",
                "expected_template_reuse": False,
            }
        )
    return pairs


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        pairs = value
    elif isinstance(value, dict) and isinstance(value.get("pairs"), list):
        pairs = value["pairs"]
    else:
        raise ValueError("pair input must be a JSON list or an object with a pairs list")
    if not all(isinstance(pair, dict) for pair in pairs):
        raise ValueError("every labeled pair must be an object")
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pairs", type=Path, help="JSON file containing labeled source pairs")
    source.add_argument("--builtin-fixture", action="store_true", help="run the 100-pair regression fixture")
    parser.add_argument("--minimum-pairs", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        pairs = builtin_pairs() if args.builtin_fixture else _load_pairs(args.pairs)
        report = evaluate_fingerprint_pairs(pairs, minimum_pairs=args.minimum_pairs)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        report = {
            "version": "source-fingerprint-v1",
            "status": "fail",
            "reason": f"audit_input_error:{type(exc).__name__}:{exc}",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
