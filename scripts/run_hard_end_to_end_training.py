"""Run strict train/dev acceptance and frozen-test verification for hard data.

The baseline report is treated as pre-patch evidence.  The candidate reruns
train and dev after the one-owner code change; test is executed only after the
acceptance gate and is never used to choose the patch.
"""

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

from scripts.run_benchmark import run_benchmark
from training.harness_memory import HarnessMemoryStore
from videoact.outer_loop import evaluate_candidate
from videoact.meta_harness import MetaHarnessOptimizer


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_end_to_end(
    dataset_root: str | Path,
    baseline_root: str | Path,
    output_root: str | Path,
    *,
    candidate_harness_version: str = "h-t2-hard-v1",
    evaluator_version: str = "deterministic-v2-independent-oracle",
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    baseline = Path(baseline_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    baseline_report = _load(baseline / "training_report.json")
    test_ids = set(baseline_report["test_case_ids"])
    proposal = MetaHarnessOptimizer(output_dir=output / "proposal").propose(
        _records(baseline / "records" / "train.jsonl"), forbidden_case_ids=test_ids
    )
    proposal_case_ids = sorted(
        set(proposal.affected_files) and {
            record["case_id"]
            for record in _records(baseline / "records" / "train.jsonl")
            if any(finding.get("owner") == proposal.owner for finding in record.get("findings", []))
        }
    )

    candidate_reports: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev"):
        candidate_reports[split] = run_benchmark(
            split,
            "fake",
            output / "benchmarks" / split,
            dataset_root=dataset,
            harness_version=candidate_harness_version,
            evaluator_version=evaluator_version,
            seed="trajectory-v3-hard-seed-1",
        )
        (output / "records").mkdir(parents=True, exist_ok=True)
        (output / "records" / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "case_id": case["case_id"],
                        "split": split,
                        "status": "pass" if case["status"] == "success" else "fail",
                        "score": case["score"],
                        "findings": [],
                    },
                    sort_keys=True,
                )
                + "\n"
                for case in candidate_reports[split]["cases"]
            ),
            encoding="utf-8",
        )

    before = {
        "train_score": float(baseline_report["benchmarks"]["train"]["mean_score"]),
        "dev_score": float(baseline_report["benchmarks"]["dev"]["mean_score"]),
    }
    after = {
        "train_score": float(candidate_reports["train"]["aggregate"]["mean_score"]),
        "dev_score": float(candidate_reports["dev"]["aggregate"]["mean_score"]),
    }
    dev_gate = {"hard_regression": after["dev_score"] < before["dev_score"]}
    decision = evaluate_candidate(before, after, {"hard_regression": False}, dev_gate)

    dataset_fingerprint = _load(dataset / "metadata.json")["fingerprint"]
    memory = HarnessMemoryStore(output / "memory")
    memory_id = memory.begin_update(
        parent_version=baseline_report["harness_version"],
        candidate_version=candidate_harness_version,
        owner=proposal.owner,
        dataset_fingerprint=dataset_fingerprint,
        evaluator_fingerprint=evaluator_version,
        affected_case_ids=proposal_case_ids,
        forbidden_case_ids=test_ids,
    )
    memory.append_event(
        memory_id,
        "patch_applied",
        files=proposal.affected_files,
        patch_summary="expanded action/entity normalization for hard compositional prompts",
    )
    memory.append_event(
        memory_id,
        "train_evaluated",
        train_before=before["train_score"],
        train_after=after["train_score"],
        evidence=["baseline/training_report.json", "benchmarks/train/benchmark_report.json"],
    )
    memory.append_event(
        memory_id,
        "dev_evaluated",
        dev_before=before["dev_score"],
        dev_after=after["dev_score"],
        hard_regression=dev_gate["hard_regression"],
        evidence=["baseline/training_report.json", "benchmarks/dev/benchmark_report.json"],
    )
    if decision.accepted:
        memory.append_event(
            memory_id,
            "accepted",
            train_before=before["train_score"],
            train_after=after["train_score"],
            dev_before=before["dev_score"],
            dev_after=after["dev_score"],
            hard_regression=dev_gate["hard_regression"],
            reason=decision.reason,
        )
        candidate_reports["test"] = run_benchmark(
            "test",
            "fake",
            output / "benchmarks" / "test",
            dataset_root=dataset,
            harness_version=candidate_harness_version,
            evaluator_version=evaluator_version,
            seed="trajectory-v3-hard-seed-1",
        )
        memory.append_event(
            memory_id,
            "test_evaluated",
            score=candidate_reports["test"]["aggregate"]["mean_score"],
            pass_rate=candidate_reports["test"]["aggregate"]["pass_rate"],
            evidence=["benchmarks/test/benchmark_report.json"],
            frozen=True,
        )
    else:
        memory.append_event(memory_id, "rejected", reason=decision.reason)

    report = {
        "dataset_id": _load(dataset / "metadata.json")["dataset_id"],
        "dataset_fingerprint": dataset_fingerprint,
        "evaluator_version": evaluator_version,
        "baseline": baseline_report["benchmarks"],
        "candidate": {
            split: {
                "case_count": candidate_reports[split]["case_count"],
                "mean_score": candidate_reports[split]["aggregate"]["mean_score"],
                "pass_rate": candidate_reports[split]["aggregate"]["pass_rate"],
                "unique_plan_count": candidate_reports[split]["aggregate"]["unique_plan_count"],
                "plan_collision_rate": candidate_reports[split]["aggregate"]["plan_collision_rate"],
            }
            for split in candidate_reports
        },
        "outer_loop": {
            "status": "accepted" if decision.accepted else "rejected",
            "owner": proposal.owner,
            "affected_case_count": len(proposal_case_ids),
            "memory_id": memory_id,
            "acceptance": decision.model_dump(mode="json"),
            "test_policy": "frozen_final_only",
        },
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--candidate-harness-version", default="h-t2-hard-v1")
    args = parser.parse_args()
    print(json.dumps(run_end_to_end(args.dataset_root, args.baseline_root, args.out_root, candidate_harness_version=args.candidate_harness_version), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
