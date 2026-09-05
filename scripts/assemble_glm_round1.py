"""Seal the recovered round-01 train/dev/test evidence into one report."""

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

from scripts.train_real_harness import (  # noqa: E402
    build_glm_six_round_manifest,
    require_benchmark_training_dataset,
    require_vbench_test100_dataset,
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _case_summary(*reports: dict[str, Any]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for report in reports:
        cases.extend(item for item in report.get("cases", []) if isinstance(item, dict))
    scores = [
        float(item["overall_vlm_score"])
        for item in cases
        if item.get("overall_vlm_score") is not None
    ]
    real_count = sum(1 for item in cases if item.get("artifact_status") == "complete")
    scored_count = sum(1 for item in cases if item.get("vlm_status") == "scored")
    return {
        "status": "complete" if real_count == len(cases) and scored_count == len(cases) else "incomplete_local_visual_review",
        "case_count": len(cases),
        "real_video_count": real_count,
        "vlm_scored_count": scored_count,
        "vlm_unavailable_count": sum(1 for item in cases if item.get("vlm_status") == "unavailable"),
        "needs_human_review_count": sum(1 for item in cases if item.get("vlm_status") == "needs_human_review"),
        "mean_vlm_score": round(sum(scores) / len(scores), 4) if scores else None,
        "source_reports": [str(Path(report["_path"]).resolve()) for report in reports if report.get("_path")],
    }


def assemble(root: str | Path, dataset_root: str | Path, test_dataset_root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    dataset = Path(dataset_root)
    test_dataset = Path(test_dataset_root)
    require_benchmark_training_dataset(dataset)
    test_validation = require_vbench_test100_dataset(test_dataset)
    training_metadata = _load(dataset / "metadata.json")
    test_metadata = _load(test_dataset / "metadata.json")
    train_splits = _load(dataset / "splits.json")
    test_splits = _load(test_dataset / "splits.json")
    protocol = build_glm_six_round_manifest(
        train_splits["train"],
        train_splits["dev"],
        test_splits["test"],
        dataset_fingerprint=training_metadata["fingerprint"],
        test_dataset_id=test_metadata["dataset_id"],
        test_dataset_fingerprint=test_validation["fingerprint"],
    )
    protocol["provider_mode"] = "glm"
    (root_path / "six_round_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    attempt1_root = root_path / "round-01" / "attempt-01" / "real"
    attempt2_root = root_path / "round-01" / "attempt-02" / "real"
    train_report = _load(attempt1_root / "train" / "real_unified_score.json")
    dev_report = _load(attempt1_root / "dev" / "real_unified_score.json")
    case20_report = _load(attempt2_root / "dev" / "real_unified_score.json")
    train_report["_path"] = str(attempt1_root / "train" / "real_unified_score.json")
    dev_report["_path"] = str(attempt1_root / "dev" / "real_unified_score.json")
    case20_report["_path"] = str(attempt2_root / "dev" / "real_unified_score.json")
    test_report = _load(root_path / "round-01" / "test" / "test_report.json")
    patch_path = root_path / "round-01" / "harness_proposal.json"
    patch_audit = _load(patch_path) if patch_path.is_file() else {"status": "missing"}
    batch = protocol["rounds"][0]
    attempt1 = {
        "round": 1,
        "attempt": 1,
        "batch": {"train": batch["train"], "dev": batch["dev"]},
        "splits": {"train": train_report, "dev": dev_report},
        "status": "completed_with_one_dev_inner_loop_exhaustion",
    }
    attempt2 = {
        "round": 1,
        "attempt": 2,
        "batch": {"train": [], "dev": ["vbench2-dev-01-20"]},
        "splits": {"dev": case20_report},
        "status": "recovered_case20_after_director_boundary_fix",
    }
    round_report = {
        "round": 1,
        "execution_mode": "unattended_evidence_training_recovered",
        "batch": batch,
        "attempt": attempt1,
        "repair_attempts": [attempt2],
        "combined_splits": {
            "train": _case_summary(train_report),
            "dev": _case_summary(dev_report, case20_report),
        },
        "outer_loop": {
            "status": "round_completed_without_harness_patch",
            "attempt_count": 2,
            "max_attempts": 5,
            "inner_recovery_attempts": 3,
            "patch_audit": str(patch_path.resolve()),
            "test_excluded": True,
        },
        "patch_audit": patch_audit,
        "test": test_report,
    }
    round_dir = root_path / "round-01"
    (round_dir / "round_report.json").write_text(
        json.dumps(round_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return round_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--test-dataset-root", default="dataset/vbench2-agent-test-100-v1")
    args = parser.parse_args()
    report = assemble(args.root, args.dataset_root, args.test_dataset_root)
    print(json.dumps({"round": report["round"], "status": report["outer_loop"]["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
