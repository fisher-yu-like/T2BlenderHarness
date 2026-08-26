"""Complete a real run using Codex-authored local frame reviews, without rendering again."""

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

from scripts.evaluate_real_runs import evaluate_real_split  # noqa: E402
from scripts.evaluate_real_videos import evaluate_split  # noqa: E402
from scripts.train_real_harness import merge_real_scores, write_unified_outputs  # noqa: E402


def complete_reviews(
    run_root: str | Path,
    *,
    dataset_root: str | Path,
    assistant_review_dir: str | Path,
    blender_bin: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict:
    deterministic = evaluate_real_split(run_root, dataset_root=dataset_root, blender_bin=blender_bin)
    reviewed = evaluate_split(
        run_root,
        dataset_root=dataset_root,
        assistant_local=True,
        assistant_review_dir=assistant_review_dir,
    )
    report = merge_real_scores(
        run_root=run_root,
        deterministic_results=deterministic,
        vlm_results=reviewed,
    )
    report["review_source"] = "assistant_local_review"
    report["status"] = "complete" if report["vlm_scored_count"] == report["real_video_count"] else "incomplete_local_review"
    write_unified_outputs(
        report,
        dataset_root=dataset_root,
        report_root=run_root,
        markdown_path=markdown_path,
    )
    (Path(run_root) / "real_unified_score.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--assistant-review-dir", required=True)
    parser.add_argument("--blender-bin")
    parser.add_argument("--markdown-path")
    args = parser.parse_args()
    print(
        json.dumps(
            complete_reviews(
                args.run_root,
                dataset_root=args.dataset_root,
                assistant_review_dir=args.assistant_review_dir,
                blender_bin=args.blender_bin,
                markdown_path=args.markdown_path,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
