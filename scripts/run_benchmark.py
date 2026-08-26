"""Run a reproducible fake-backend benchmark over a frozen split."""

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

from videoact.contracts import ExecutionResult  # noqa: E402
from videoact.inner_loop import run_inner_loop  # noqa: E402
from evaluator.independent_oracle import evaluate_independent_oracle  # noqa: E402
from videoact.scene_contract import SceneContractBuilder  # noqa: E402
from videoact.trajectory import TrajectoryPlanner  # noqa: E402
from videoact.run_manifest import hash_payload  # noqa: E402


def cache_key(
    prompt: str,
    harness_version: str,
    evaluator_version: str,
    backend: str,
    seed: str,
    render_settings: str = "white-proxy-v1",
) -> str:
    payload = {
        "prompt": prompt,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "backend": backend,
        "seed": seed,
        "render_settings": render_settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FakeBenchmarkAdapter:
    def run(self, script_path, run_dir, *, prefer="mcp", timeout_s=300):
        return ExecutionResult(status="success", backend="fake")


def run_benchmark(
    split: str,
    backend: str,
    out_dir: str | Path,
    *,
    dataset_root: str | Path = "dataset",
    harness_version: str = "h1",
    evaluator_version: str = "deterministic-v1",
    seed: str = "seed-1",
    case_ids: list[str] | None = None,
) -> dict[str, Any]:
    if backend != "fake":
        raise ValueError("the local reproducibility benchmark only supports --backend fake")
    dataset_root = Path(dataset_root)
    manifest = [
        json.loads(line)
        for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_ids = case_ids or json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))[split]
    allowed_ids = set(json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))[split])
    if not set(split_ids) <= allowed_ids:
        raise ValueError(f"case IDs are not all in {split} split")
    by_id = {record["case_id"]: record for record in manifest}
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    case_results = []
    for case_id in split_ids:
        record = by_id[case_id]
        result = run_inner_loop(
            {
                "case_id": case_id,
                "prompt": record["prompt"],
                "duration_s": record["duration_s"],
                "fps": record["fps"],
            },
            {"version": harness_version, "evaluator_version": evaluator_version},
            output / case_id,
            adapter=FakeBenchmarkAdapter(),
            max_attempts=1,
        )
        contract = SceneContractBuilder().build(record["prompt"], duration_s=record["duration_s"], fps=record["fps"])
        plan = TrajectoryPlanner().plan(contract)
        plan_hash = hash_payload(plan.model_dump(mode="json"))
        independent_findings = evaluate_independent_oracle(record, contract, plan)
        all_findings = list(result.findings) + independent_findings
        findings = sorted({finding.failure_id for finding in all_findings})
        score = max(0.0, float(result.final_score or 0.0) - 15.0 * len(independent_findings))
        status = "fail" if any(finding.severity == "hard" for finding in all_findings) else result.status
        case_results.append(
            {
                "case_id": case_id,
                "status": status,
                "score": score,
                "cache_key": cache_key(record["prompt"], harness_version, evaluator_version, backend, seed),
                "failure_ids": findings,
                "independent_finding_count": len(independent_findings),
                "plan_hash": plan_hash,
                "proxy_scene_id": record.get("proxy_scene", {}).get("scene_id"),
            }
        )
    scores = [case["score"] for case in case_results]
    passed = sum(case["status"] == "success" for case in case_results)
    report = {
        "protocol_version": "benchmark-v1",
        "split": split,
        "backend": backend,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "seed": seed,
        "case_count": len(case_results),
        "cases": case_results,
        "aggregate": {
            "pass_rate": passed / len(case_results) if case_results else 0.0,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "failure_counts": {
                failure_id: sum(failure_id in case["failure_ids"] for case in case_results)
                for failure_id in sorted({item for case in case_results for item in case["failure_ids"]})
            },
            "unique_plan_count": len({case["plan_hash"] for case in case_results}),
            "plan_collision_rate": (
                1.0 - len({case["plan_hash"] for case in case_results}) / len(case_results)
                if case_results
                else 0.0
            ),
        },
    }
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--backend", choices=["fake"], default="fake")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-id", action="append", default=None)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.split, args.backend, args.out_dir, case_ids=args.case_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
