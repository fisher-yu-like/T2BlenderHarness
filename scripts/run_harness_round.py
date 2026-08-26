"""Run one fixed 10-train/10-dev Harness round and prepare matching jobs."""

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

from scripts.prepare_real_jobs import prepare_jobs
from scripts.run_benchmark import run_benchmark


def _split_ids(dataset_root: Path, split: str) -> list[str]:
    ids = json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))[split]
    if len(ids) < 10:
        raise ValueError(f"{split} split has fewer than 10 cases")
    return ids[:10]


def run_round(
    round_number: int,
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    harness_version: str,
    evaluator_version: str = "deterministic-v2-independent-oracle",
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected = {split: _split_ids(dataset, split) for split in ("train", "dev")}
    reports: dict[str, dict[str, Any]] = {}
    jobs: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        reports[split] = run_benchmark(
            split,
            "fake",
            output / "benchmarks" / split,
            dataset_root=dataset,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            seed="t2blendercodeharness-five-round-seed-1",
            case_ids=selected[split],
        )
        jobs[split] = prepare_jobs(
            split,
            output / "real" / split,
            dataset_root=dataset,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            case_ids=selected[split],
        )
    report = {
        "round": round_number,
        "dataset_id": json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))["dataset_id"],
        "dataset_fingerprint": json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))["fingerprint"],
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "selected_case_ids": selected,
        "benchmarks": reports,
        "jobs": {split: {"case_count": jobs[split]["case_count"], "job_index": str((output / "real" / split / "job_index.json").resolve())} for split in jobs},
    }
    (output / "round_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--harness-version", required=True)
    args = parser.parse_args()
    print(json.dumps(run_round(args.round, args.dataset_root, args.out_root, harness_version=args.harness_version), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
