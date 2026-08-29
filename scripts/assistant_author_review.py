"""Assemble an assistant-local visual review payload from authored scores.

The driving agent inspects the sampled real frames of a case and authors the
numeric review in a side file; this tool binds that review to the evaluator's
exact review request (version, reviewer identity, sampled frame paths) so the
validated review passes the assistant-local contract.
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

from evaluator.assistant_local import ASSISTANT_REVIEW_VERSION  # noqa: E402

REVIEWER = "codex-assistant"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--scores-file", required=True, help="authored VLMJudgeResponse-shaped JSON")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    request = json.loads((run_dir / "assistant_review_request.json").read_text(encoding="utf-8"))
    authored = json.loads(Path(args.scores_file).read_text(encoding="utf-8"))
    if not isinstance(authored.get("scores"), dict):
        raise SystemExit("scores file must contain a 'scores' object (VLMJudgeResponse fields)")
    case_id = request.get("case_id")
    if not case_id:
        raise SystemExit("assistant review request is missing case_id")
    payload = {
        "review_version": ASSISTANT_REVIEW_VERSION,
        "review_source": "assistant_local_review",
        "reviewer": REVIEWER,
        "case_id": case_id,
        "sampled_frames": request.get("sampled_frames", []),
        "scores": authored["scores"],
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{case_id}.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"written": str(target), "case_id": case_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
