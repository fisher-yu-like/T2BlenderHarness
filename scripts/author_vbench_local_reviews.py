"""Materialize low-level observations for the historical VBench review bundle.

This script intentionally does not synthesize semantic or realism scores from
case IDs, variants, categories, or expected outcomes.  It is retained only to
make the old bundle auditable; those scores are superseded by an independent
VLM or validated human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.visual_evidence import score_sample_frames  # noqa: E402


def author_reviews(benchmark_root: str | Path, output_root: str | Path, dataset_root: str | Path) -> dict[str, int]:
    benchmark = Path(benchmark_root)
    output = Path(output_root)
    records = [
        json.loads(line)
        for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {record["case_id"]: record for record in records}
    counts: dict[str, int] = {}
    for variant in ("pretrain", "current"):
        for split in ("train", "dev"):
            run_root = benchmark / variant / "real" / split
            destination = output / variant / split
            count = 0
            if not run_root.is_dir():
                counts[f"{variant}/{split}"] = 0
                continue
            for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
                request_path = run_dir / "assistant_review_request.json"
                if not request_path.is_file():
                    continue
                request = json.loads(request_path.read_text(encoding="utf-8"))
                case_id = request.get("case_id") or run_dir.name
                if case_id not in by_id:
                    raise ValueError(f"review request case is absent from dataset: {case_id}")
                review = score_sample_frames(list(request.get("sampled_frames") or []))
                payload = {
                    **request,
                    "review_source": "frame_statistics",
                    "method": "frame_statistics_only-v1",
                    "reviewer": "frame-statistics",
                    "scores": review["scores"],
                    "score": None,
                    "artifact_health": review["artifact_health"],
                    "frame_metrics": review["frame_metrics"],
                    "visible_evidence": ["Only low-level properties of the exact sampled frames were measured."],
                    "weaknesses": ["Semantic and realism claims require an independent review."],
                    "confidence": 0.0,
                }
                destination.mkdir(parents=True, exist_ok=True)
                (destination / f"{case_id}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                count += 1
            counts[f"{variant}/{split}"] = count
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    print(json.dumps(author_reviews(args.benchmark_root, args.output_root, args.dataset_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

