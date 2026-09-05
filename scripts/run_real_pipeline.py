"""Drive the real pipeline stages that are safe to run from the host process."""

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

from scripts.evaluate_real_runs import evaluate_real_split
from scripts.prepare_real_jobs import prepare_jobs
from videoact.meta_harness import MetaHarnessOptimizer


def run_pipeline(
    mode: str,
    *,
    split: str,
    out_dir: str | Path,
    dataset_root: str | Path = "dataset/trajectory-v5-agent-codegen",
    harness_version: str = "t2blendercodeharness-v5-executable-director",
    train_records: list[dict[str, Any]] | None = None,
    forbidden_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    if mode == "dry-run":
        return prepare_jobs(split, out_dir, dataset_root=dataset_root, harness_version=harness_version)
    if mode == "evaluate":
        results = evaluate_real_split(out_dir, dataset_root=dataset_root)
        summary = {
            "mode": mode,
            "split": split,
            "case_count": len(results),
            "pass_count": sum(result["status"] == "pass" for result in results),
            "results": results,
        }
        Path(out_dir, "real_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary
    if mode == "optimize":
        if split != "train":
            raise ValueError("optimize mode can only consume the train split")
        if train_records is None:
            raise ValueError("train_records are required for optimize mode")
        optimizer = MetaHarnessOptimizer(output_dir=out_dir)
        proposal = optimizer.propose(train_records, forbidden_case_ids=forbidden_case_ids)
        return {"status": "proposal_ready", "proposal": proposal.model_dump(mode="json")}
    raise ValueError("mode must be dry-run, evaluate, or optimize; MCP execution is performed by the connected host tool")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "evaluate", "optimize"], required=True)
    parser.add_argument("--split", choices=["calibration", "train", "dev", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--harness-version", default="t2blendercodeharness-v5-executable-director")
    parser.add_argument("--train-records")
    args = parser.parse_args()
    records = None
    if args.train_records:
        records = json.loads(Path(args.train_records).read_text(encoding="utf-8"))
    result = run_pipeline(
        args.mode,
        split=args.split,
        out_dir=args.out_dir,
        dataset_root=args.dataset_root,
        harness_version=args.harness_version,
        train_records=records,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
