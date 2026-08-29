"""Materialize low-level frame observations for legacy anonymous review requests.

This entry point is retained for compatibility with old review folders.  It
does not author semantic scores.  Use a real VLM or a human review payload for
prompt, event, physical, camera, and trajectory dimensions.
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


METHOD = "frame_statistics_only-v1"
REVIEW_SOURCE = "frame_statistics"
BLIND_REVIEW_VERSION = "frame-statistics-v1"


def build_reviews(blind_root: str | Path) -> dict[str, object]:
    root = Path(blind_root)
    requests_dir = root / "requests"
    reviews_dir = root / "reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    complete_samples = 0
    for request_path in sorted(requests_dir.glob("*.json")):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        samples: dict[str, dict[str, object]] = {}
        for label, sample in sorted((request.get("samples") or {}).items()):
            frame_paths = list(sample.get("sampled_frames") or [])
            review = score_sample_frames(frame_paths)
            review.update(
                {
                    "case_id": request["case_id"],
                    "sample_label": label,
                    "sampled_frames": frame_paths,
                    "reviewer": "frame-statistics",
                }
            )
            samples[label] = review
            complete_samples += int(review.get("status") == "complete")
        payload = {
            "review_version": BLIND_REVIEW_VERSION,
            "review_source": REVIEW_SOURCE,
            "reviewer": "frame-statistics",
            "method": METHOD,
            "case_id": request["case_id"],
            "samples": samples,
            "blind_policy": "only exact PNG paths were read; semantic dimensions remain unobserved",
        }
        (reviews_dir / request_path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        count += 1
    return {
        "case_count": count,
        "sample_count": complete_samples,
        "review_root": str(root.resolve()),
        "method": METHOD,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-root", required=True)
    args = parser.parse_args()
    print(json.dumps(build_reviews(args.blind_root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

