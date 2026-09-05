"""Aggregate real train/dev reports and apply the MetaHarness proposal gate."""

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

from videoact.meta_harness import MetaHarnessOptimizer  # noqa: E402
from videoact.failure_extractor import FailureExtractor  # noqa: E402


def collect_records(
    run_root: str | Path,
    split: str,
    *,
    forbidden_case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    root = Path(run_root)
    if not root.is_dir():
        return records
    run_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "run_manifest.json").is_file()
    )
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest_split = str(manifest.get("split") or "")
        if manifest_split and manifest_split != split:
            raise ValueError(
                f"run manifest split mismatch for {manifest.get('case_id')}: "
                f"expected {split}, got {manifest_split}"
            )
        report = json.loads((run_dir / "deterministic_report.json").read_text(encoding="utf-8"))
        normalized_failures: list[dict[str, Any]] = []
        abstentions: list[dict[str, Any]] = []
        if split == "train":
            # Only train evidence may reach proposal aggregation.  Dev is
            # summarized below for acceptance and is intentionally not fed to
            # the extractor/controller.
            extracted = FailureExtractor().extract_run(
                run_dir,
                expected_split="train",
                forbidden_case_ids=forbidden_case_ids,
            )
            normalized_failures = [item.to_finding() for item in extracted if item.actionable]
            abstentions = [item.model_dump(mode="json") for item in extracted if item.abstain]
        else:
            # Dev remains available for the paired acceptance summary only;
            # it cannot become a proposal input.
            normalized_failures = list(report.get("findings", []))
        records.append(
            {
                "case_id": manifest["case_id"],
                "split": split,
                "status": report["terminal_status"],
                "score": report["score"],
                "hard_gate_failed": report["hard_gate_failed"],
                "findings": normalized_failures,
                "abstentions": abstentions,
                "raw_findings": report.get("findings", []),
            }
        )
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(record["score"]) for record in records]
    return {
        "case_count": len(records),
        "pass_count": sum(record["status"] == "pass" for record in records),
        "hard_gate_count": sum(bool(record["hard_gate_failed"]) for record in records),
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


def run_outer_loop(
    train_root: str | Path,
    dev_root: str | Path,
    output: str | Path,
    *,
    forbidden_case_ids: set[str] | None = None,
) -> dict[str, Any]:
    train_records = collect_records(
        train_root,
        "train",
        forbidden_case_ids=forbidden_case_ids,
    )
    dev_records = collect_records(dev_root, "dev")
    optimizer = MetaHarnessOptimizer(output_dir=Path(output).parent)
    train_summary = summarize(train_records)
    dev_summary = summarize(dev_records)
    try:
        proposal = optimizer.propose(train_records, forbidden_case_ids=forbidden_case_ids)
    except ValueError as exc:
        result = {
            "status": "no_patch",
            "reason": str(exc),
            "scoring_mode": "deterministic_only",
            "train": train_summary,
            "dev": dev_summary,
            "train_records": train_records,
            "dev_records": dev_records,
        }
    else:
        result = {
            "status": "proposal_ready",
            "reason": "repeated train failure requires a one-owner Harness patch",
            "proposal": proposal.model_dump(mode="json"),
            "scoring_mode": "deterministic_only",
            "train": train_summary,
            "dev": dev_summary,
            "train_records": train_records,
            "dev_records": dev_records,
        }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True)
    parser.add_argument("--dev-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(run_outer_loop(args.train_root, args.dev_root, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
