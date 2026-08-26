"""Finalize a round-wide report after local visual fallback reviews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.train_real_harness import update_training_memory_table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--markdown-path", required=True)
    args = parser.parse_args()
    root = Path(args.round_root)
    split_reports = {}
    cases = []
    for split in ("train", "dev"):
        path = root / "overall" / "real" / split / "real_unified_score.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        split_reports[split] = report
        cases.extend({"split": split, **case} for case in report.get("cases", []))
    scored = [float(case["video_score"]) for case in cases if case.get("video_score") is not None]
    deterministic = [float(case["deterministic_score"]) for case in cases if case.get("deterministic_score") is not None]
    result = {
        "round": 1,
        "scope": "all_train_and_all_dev",
        "status": "complete_artifact_audit_with_partial_visual_scoring",
        "real_video_count": sum(bool(case.get("video_exists")) for case in cases),
        "case_count": len(cases),
        "vlm_scored_count": len(scored),
        "deterministic_fail_count": sum(case.get("deterministic_status") != "pass" for case in cases),
        "mean_deterministic_score": round(sum(deterministic) / len(deterministic), 4) if deterministic else None,
        "mean_final_score_scored_only": round(sum(scored) / len(scored), 4) if scored else None,
        "visual_score_policy": "failed deterministic cases are explicitly unscored; unavailable VLM is never converted to zero",
        "splits": split_reports,
        "cases": cases,
    }
    out = root / "overall_report.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_training_memory_table(root.parent, args.dataset_root, args.markdown_path)
    print(json.dumps({k: result[k] for k in ("case_count", "real_video_count", "vlm_scored_count", "deterministic_fail_count", "mean_deterministic_score", "mean_final_score_scored_only")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
