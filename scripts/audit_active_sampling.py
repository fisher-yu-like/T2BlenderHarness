"""Audit budgeted active sampling against historical train/dev replay data."""

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

from videoact.active_sampling import audit_sampling_replay


def _load(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        batches = value
    elif isinstance(value, dict) and isinstance(value.get("batches"), list):
        batches = value["batches"]
    else:
        raise ValueError("replay input must be a JSON list or an object with a batches list")
    if not all(isinstance(batch, dict) for batch in batches):
        raise ValueError("every replay batch must be an object")
    return batches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True, help="historical train/dev replay JSON")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--target-lower-bound", type=float, required=True)
    parser.add_argument("--min-cases", type=int, default=10)
    parser.add_argument("--min-reduction", type=float, default=0.30)
    parser.add_argument("--min-agreement", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = audit_sampling_replay(
            _load(args.replay),
            budget=args.budget,
            target_lower_bound=args.target_lower_bound,
            min_cases=args.min_cases,
            min_reduction=args.min_reduction,
            min_agreement=args.min_agreement,
            seed=args.seed,
            iterations=args.iterations,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        report = {
            "version": "active-sampling-v2-replay",
            "status": "fail",
            "reason": f"audit_input_error:{type(exc).__name__}:{exc}",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
