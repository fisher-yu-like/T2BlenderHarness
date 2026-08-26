"""Run the Harness train/dev/test benchmark and persist update memory."""

from __future__ import annotations

import argparse
import hashlib
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
from videoact.meta_harness import MetaHarnessOptimizer


OWNER_BY_FAILURE = {
    "camera_event_uncovered": "camera_planner",
    "telemetry_inactive_camera": "camera_planner",
    "telemetry_entity_kind_mismatch": "scene_parser",
    "incomplete_proxy": "blender_executor",
    "incomplete_real_artifacts": "proxy_renderer",
    "oracle_event_order_mismatch": "scene_parser",
    "oracle_constraint_missing": "scene_parser",
    "oracle_proxy_entity_mismatch": "scene_parser",
    "oracle_camera_intent_missing": "camera_planner",
    "oracle_motion_primitive_missing": "trajectory_planner",
    "oracle_attachment_lifecycle_mismatch": "trajectory_planner",
}


def _dataset_fingerprint(dataset_root: Path) -> str:
    metadata = dataset_root / "metadata.json"
    if metadata.exists():
        return json.loads(metadata.read_text(encoding="utf-8"))["fingerprint"]
    return hashlib.sha256((dataset_root / "manifest.jsonl").read_bytes()).hexdigest()


def _finding(failure_id: str) -> dict[str, Any]:
    owner = OWNER_BY_FAILURE.get(failure_id, "evaluator")
    route = {
        "camera_planner": "camera_repair",
        "scene_parser": "scene_contract_repair",
        "blender_executor": "runtime_repair",
        "proxy_renderer": "runtime_repair",
        "evaluator": "candidate_recovery",
    }.get(owner, "candidate_recovery")
    return {
        "failure_id": failure_id,
        "owner": owner,
        "category": "runtime" if owner in {"blender_executor", "proxy_renderer"} else "harness",
        "severity": "hard",
        "message": f"{failure_id} reported by benchmark",
        "evidence": ["benchmark_report.json"],
        "repair_route": route,
    }


def _records(report: dict[str, Any], split: str) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["case_id"],
            "split": split,
            "status": "pass" if case["status"] == "success" else "fail",
            "score": case["score"],
            "findings": [_finding(failure_id) for failure_id in case.get("failure_ids", [])],
        }
        for case in report.get("cases", [])
    ]


def run_training(
    dataset_root: str | Path = "dataset/trajectory-v2",
    output_root: str | Path = "out/training/trajectory-v2",
    *,
    harness_version: str = "h-trajectory-v2",
    evaluator_version: str = "deterministic-v1",
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    benchmark_reports: dict[str, dict[str, Any]] = {}
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        report = run_benchmark(
            split,
            "fake",
            output / "benchmarks" / split,
            dataset_root=dataset,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            seed="trajectory-v2-seed-1",
        )
        benchmark_reports[split] = report
        records_by_split[split] = _records(report, split)
        (output / "records").mkdir(parents=True, exist_ok=True)
        (output / "records" / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records_by_split[split]),
            encoding="utf-8",
        )

    train_records = records_by_split["train"]
    test_case_ids = {record["case_id"] for record in records_by_split["test"]}
    optimizer = MetaHarnessOptimizer(output_dir=output / "memory")
    memory = HarnessMemoryStore(output / "memory")
    train_score = benchmark_reports["train"]["aggregate"]["mean_score"]
    dev_score = benchmark_reports["dev"]["aggregate"]["mean_score"]
    dataset_fingerprint = _dataset_fingerprint(dataset)
    try:
        proposal = optimizer.propose(train_records, forbidden_case_ids=test_case_ids)
    except ValueError as exc:
        memory_id = memory.begin_update(
            parent_version=harness_version,
            candidate_version=harness_version,
            owner="none",
            dataset_fingerprint=dataset_fingerprint,
            evaluator_fingerprint=evaluator_version,
            affected_case_ids=[],
        )
        memory.append_event(memory_id, "train_evaluated", score=train_score, evidence=["benchmarks/train/benchmark_report.json"])
        memory.append_event(memory_id, "dev_evaluated", score=dev_score, evidence=["benchmarks/dev/benchmark_report.json"], hard_regression=False)
        memory.append_event(memory_id, "test_evaluated", score=benchmark_reports["test"]["aggregate"]["mean_score"], evidence=["benchmarks/test/benchmark_report.json"], frozen=True)
        memory.append_event(memory_id, "no_patch", reason=str(exc))
        outer_loop = {
            "status": "no_patch",
            "reason": str(exc),
            "memory_id": memory_id,
            "proposal_case_ids": [],
            "train_score": train_score,
            "dev_score": dev_score,
        }
    else:
        proposal_case_ids = sorted(
            {
                record["case_id"]
                for record in train_records
                if any(
                    finding["owner"] == proposal.owner
                    for finding in record.get("findings", [])
                )
            }
        )
        memory_id = memory.begin_update(
            parent_version=harness_version,
            candidate_version=f"{harness_version}+proposal",
            owner=proposal.owner,
            dataset_fingerprint=dataset_fingerprint,
            evaluator_fingerprint=evaluator_version,
            affected_case_ids=proposal_case_ids,
            forbidden_case_ids=test_case_ids,
        )
        memory.append_event(memory_id, "train_evaluated", score=train_score, evidence=["benchmarks/train/benchmark_report.json"])
        memory.append_event(memory_id, "dev_evaluated", score=dev_score, evidence=["benchmarks/dev/benchmark_report.json"], hard_regression=False)
        memory.append_event(memory_id, "test_evaluated", score=benchmark_reports["test"]["aggregate"]["mean_score"], evidence=["benchmarks/test/benchmark_report.json"], frozen=True)
        memory.append_event(memory_id, "rejected", reason="proposal recorded; no code patch applied by benchmark runner")
        outer_loop = {
            "status": "proposal_only",
            "reason": "a coding agent must apply and rerun a one-owner patch before acceptance",
            "memory_id": memory_id,
            "proposal_case_ids": proposal_case_ids,
            "proposal": proposal.model_dump(mode="json"),
            "train_score": train_score,
            "dev_score": dev_score,
        }

    metadata = dataset / "metadata.json"
    dataset_id = json.loads(metadata.read_text(encoding="utf-8")).get("dataset_id", dataset.name) if metadata.exists() else dataset.name
    report = {
        "dataset_id": dataset_id,
        "dataset_fingerprint": dataset_fingerprint,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "training_mode": "code-level-harness-evolution",
        "test_policy": "frozen_final_only",
        "test_case_ids": sorted(test_case_ids),
        "benchmarks": {
            split: {
                "case_count": benchmark_reports[split]["case_count"],
                "pass_rate": benchmark_reports[split]["aggregate"]["pass_rate"],
                "mean_score": benchmark_reports[split]["aggregate"]["mean_score"],
            }
            for split in ("train", "dev", "test")
        },
        "outer_loop": outer_loop,
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/trajectory-v2")
    parser.add_argument("--out-root", default="out/training/trajectory-v2")
    parser.add_argument("--harness-version", default="h-trajectory-v2")
    parser.add_argument("--evaluator-version", default="deterministic-v1")
    args = parser.parse_args()
    print(json.dumps(run_training(args.dataset_root, args.out_root, harness_version=args.harness_version, evaluator_version=args.evaluator_version), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
