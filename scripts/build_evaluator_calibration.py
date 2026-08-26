"""Build a transparent evaluator calibration status report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_calibration_report(
    labels: list[dict[str, Any]],
    *,
    minimum_labeled_cases: int = 10,
) -> dict[str, Any]:
    labeled = [
        label
        for label in labels
        if label.get("pass_fail") in {"pass", "fail"}
        and label.get("primary_failure_owner") not in {None, "unreviewed"}
    ]
    predicted = [label for label in labeled if label.get("predicted_owner")]
    owner_accuracy = None
    if predicted:
        owner_accuracy = sum(
            label["predicted_owner"] == label["primary_failure_owner"] for label in predicted
        ) / len(predicted)
    status = "ready" if len(labeled) >= minimum_labeled_cases else "not_ready"
    return {
        "calibration_version": "calibration-v1",
        "status": status,
        "minimum_labeled_cases": minimum_labeled_cases,
        "labeled_cases": len(labeled),
        "owner_accuracy": owner_accuracy,
        "note": "Independent human labels are required before threshold changes or preference export.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="dataset/labels.jsonl")
    parser.add_argument("--out", default="out/calibration.json")
    args = parser.parse_args()
    labels = [
        json.loads(line)
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = build_calibration_report(labels)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
