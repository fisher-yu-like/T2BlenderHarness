"""Write low-level frame observations for the legacy shared review bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.visual_evidence import score_sample_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--improved", action="store_true", help="deprecated compatibility flag; ignored")
    args = parser.parse_args()

    run_root = Path(args.run_root) / "real" / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = sorted(run_root.glob("*/assistant_review_request.json"))
    if len(requests) != 10:
        raise SystemExit(f"expected 10 review requests in {run_root}, found {len(requests)}")
    for request_path in requests:
        request = json.loads(request_path.read_text(encoding="utf-8"))
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
            "weaknesses": ["Semantic, physical, event, camera, and trajectory claims require an independent review."],
            "confidence": 0.0,
        }
        (output_dir / f"{request['case_id']}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"split": args.split, "count": len(requests), "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
