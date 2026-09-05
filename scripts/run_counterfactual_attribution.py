"""Run bounded train-only counterfactual attribution from JSON inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.failure_attribution import CounterfactualAttributor  # noqa: E402


def _load(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return [payload]


def run(input_path: str | Path, output_path: str | Path, *, max_runs: int = 5) -> list[dict[str, Any]]:
    records = _load(Path(input_path))
    attributor = CounterfactualAttributor(max_runs=max_runs)
    results = []
    for record in records:
        failure = record.get("failure", record)
        counterfactuals = record.get("counterfactuals", [])
        result = attributor.attribute(failure, counterfactuals=counterfactuals)
        results.append(result.model_dump(mode="json"))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-runs", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.out, max_runs=args.max_runs), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
