"""Compose a full round audit while reusing hash-identical immutable videos."""

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

from scripts.train_real_harness import update_training_memory_table  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-round-root", required=True)
    parser.add_argument("--candidate-round-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--markdown-path", required=True)
    args = parser.parse_args()
    previous = Path(args.previous_round_root)
    candidate = Path(args.candidate_round_root)
    combined_cases = []
    split_reports = {}
    replace_ids = {
        "train": {f"hard-02-{i:02d}" for i in range(1, 11)},
        "dev": {f"hard-08-{i:02d}" for i in range(1, 11)},
    }
    for split in ("train", "dev"):
        old_report = _load(previous / "overall" / "real" / split / "real_unified_score.json")
        new_report = _load(candidate / "attempt-02" / "real" / split / "real_unified_score.json")
        new_by_id = {case["case_id"]: case for case in new_report["cases"]}
        cases = []
        for old_case in old_report["cases"]:
            case_id = old_case["case_id"]
            if case_id in replace_ids[split]:
                case = dict(new_by_id[case_id])
                case["overall_source"] = "round-02-attempt-02-new-real-video"
            else:
                case = dict(old_case)
                case["overall_source"] = "round-01-overall-reused-hash-identical-plan"
            cases.append(case)
        split_report = dict(old_report)
        split_report["run_root"] = str((candidate / "real" / split).resolve())
        split_report["cases"] = cases
        split_report["real_video_count"] = sum(bool(case.get("video_exists")) for case in cases)
        split_report["vlm_scored_count"] = sum(case.get("video_score") is not None for case in cases)
        deterministic = [float(case["deterministic_score"]) for case in cases if case.get("deterministic_score") is not None]
        scored = [float(case["video_score"]) for case in cases if case.get("video_score") is not None]
        split_report["aggregate"] = {
            "mean_deterministic_score": round(sum(deterministic) / len(deterministic), 4),
            "mean_final_score": round(sum(scored) / len(scored), 4) if scored else None,
            "failure_counts": {},
        }
        for case in cases:
            for failure in case.get("deterministic_findings", []):
                split_report["aggregate"]["failure_counts"][failure] = split_report["aggregate"]["failure_counts"].get(failure, 0) + 1
        split_reports[split] = split_report
        combined_cases.extend({"split": split, **case} for case in cases)

    scored = [float(case["video_score"]) for case in combined_cases if case.get("video_score") is not None]
    deterministic = [float(case["deterministic_score"]) for case in combined_cases if case.get("deterministic_score") is not None]
    result = {
        "round": 2,
        "scope": "all_train_and_all_dev",
        "status": "complete_artifact_audit_with_partial_visual_scoring",
        "reuse_policy": "reuse only when scene_contract, trajectory, and camera_plan hashes are identical; replace changed cases with new real videos",
        "real_video_count": sum(bool(case.get("video_exists")) for case in combined_cases),
        "case_count": len(combined_cases),
        "vlm_scored_count": len(scored),
        "deterministic_fail_count": sum(case.get("deterministic_status") != "pass" for case in combined_cases),
        "mean_deterministic_score": round(sum(deterministic) / len(deterministic), 4),
        "mean_final_score_scored_only": round(sum(scored) / len(scored), 4) if scored else None,
        "visual_score_policy": "failed deterministic cases are explicitly unscored; unavailable VLM is never converted to zero",
        "splits": split_reports,
        "cases": combined_cases,
    }
    out = candidate / "overall_report.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_training_memory_table(candidate.parent, args.dataset_root, args.markdown_path)
    print(json.dumps({k: result[k] for k in ("case_count", "real_video_count", "vlm_scored_count", "deterministic_fail_count", "mean_deterministic_score", "mean_final_score_scored_only")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
