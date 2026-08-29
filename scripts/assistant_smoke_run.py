"""Single-case end-to-end smoke through the assistant-session provider arm.

Runs the real chain for one or a few cases: prepare (Director + Blender code
authored by the driving agent session) -> real Blender render -> deterministic
and runtime evaluation -> assistant-local visual review request.
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
    parser.add_argument("--case-ids", required=True)
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    args = parser.parse_args()

    director, code_agent = build_dynamic_codex_agents(provider_mode="assistant")
    report = run_real_batch_with_inner_loop(
        args.run_root,
        split=args.split,
        case_ids=[item.strip() for item in args.case_ids.split(",") if item.strip()],
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
    )
    print(json.dumps({"status": report.get("status"), "run_root": args.run_root}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
