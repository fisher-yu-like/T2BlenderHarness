"""Run one real evaluation batch (test/train/dev) through the assistant arm.

Prepares, renders, and deterministically evaluates the requested cases with
the assistant-session providers, leaving the split awaiting local visual
review.  After the driving session authors reviews, complete them with
scripts/complete_assistant_local_reviews.py and summarize with --summarize.
"""

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

from evaluator.visual_primary import SCORING_V7_VERSION  # noqa: E402
from scripts.train_real_harness import (  # noqa: E402
    build_dynamic_codex_agents,
    run_real_batch_with_inner_loop,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--case-ids", help="comma-separated case ids; omit for every case in the split")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--summarize", action="store_true",
                        help="write test_score_summary.json from the merged split report")
    args = parser.parse_args()

    if args.summarize:
        report = json.loads(
            (Path(args.run_root) / "real_unified_score.json").read_text(encoding="utf-8")
        )
        scored = [
            {
                "case_id": c.get("case_id"),
                "task": c.get("task_final_score"),
                "realism": c.get("realism_score"),
                "overall": c.get("overall_vlm_score"),
                "confidence": c.get("review_confidence"),
            }
            for c in report.get("cases", [])
            if c.get("overall_vlm_score") is not None
        ]
        means = {
            key: round(sum(item[key] for item in scored) / len(scored), 4) if scored else None
            for key in ("task", "realism", "overall")
        }
        summary = {
            "run_root": str(Path(args.run_root).resolve()),
            "status": report.get("status"),
            "scored_count": len(scored),
            "real_video_count": report.get("real_video_count"),
            "means": means,
            "cases": scored,
        }
        target = Path(args.run_root) / "test_score_summary.json"
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False)[:600])
        return 0

    if args.case_ids:
        case_ids = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    else:
        case_ids = [
            json.loads(line)["case_id"]
            for line in (Path(args.dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("split") == args.split
        ]
    director, code_agent = build_dynamic_codex_agents(provider_mode="assistant")
    report = run_real_batch_with_inner_loop(
        args.run_root,
        split=args.split,
        case_ids=case_ids,
        dataset_root=args.dataset_root,
        harness_version="t2blendercodeharness-v5-assistant-session",
        evaluator_version=SCORING_V7_VERSION,
        blender_bin=args.blender_bin,
        workers=args.workers,
        timeout_s=args.timeout_s,
        vlm_model="gpt-5.6-luna",
        markdown_path=None,
        director_agent=director,
        code_agent=code_agent,
        provider_mode="assistant",
        code_cache_dir=Path(args.run_root) / "code_cache",
        max_inner_attempts=2,
    )
    print(json.dumps({"status": report.get("status"), "run_root": args.run_root}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
