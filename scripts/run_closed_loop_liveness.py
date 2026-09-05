"""Run the train-only fault-injection liveness suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.liveness import (  # noqa: E402
    FaultInjection,
    default_fault_injections,
    run_liveness_suite,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.input:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        faults = [FaultInjection.model_validate(item) for item in (payload if isinstance(payload, list) else payload.get("faults", []))]
    else:
        faults = default_fault_injections()
    report = run_liveness_suite(faults)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.training_allowed else 1


if __name__ == "__main__":
    raise SystemExit(main())
