"""Run the independent frozen 100-case test for one completed round."""

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

from evaluator.codex_visual import CodexVisualReviewProvider  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION  # noqa: E402
from scripts.train_real_harness import (  # noqa: E402
    build_dynamic_codex_agents,
    require_benchmark_training_dataset,
    require_vbench_test100_dataset,
    run_round_test,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--test-dataset-root", default="dataset/vbench2-agent-test-100-v1")
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-timeout-s", type=int, default=900)
    parser.add_argument("--provider-timeout-s", type=int, default=180)
    parser.add_argument("--visual-timeout-s", type=int, default=600)
    parser.add_argument("--visual-fallback-timeout-s", type=int, default=60)
    parser.add_argument("--visual-frame-budget", type=int, default=6)
    parser.add_argument("--visual-model", default="gpt-5.6-terra")
    parser.add_argument("--markdown-path", required=True)
    args = parser.parse_args()

    dataset = Path(args.dataset_root)
    test_dataset = Path(args.test_dataset_root)
    require_benchmark_training_dataset(dataset)
    validation = require_vbench_test100_dataset(test_dataset)
    test_ids = json.loads((test_dataset / "splits.json").read_text(encoding="utf-8"))["test"]
    director, code_agent = build_dynamic_codex_agents(
        codex_command="codex",
        timeout_s=args.provider_timeout_s,
        provider_mode="glm",
    )
    visual = CodexVisualReviewProvider(
        command="codex",
        model=args.visual_model,
        timeout_s=args.visual_timeout_s,
        fallback_timeout_s=args.visual_fallback_timeout_s,
        visual_frame_budget=args.visual_frame_budget,
    )
    report = run_round_test(
        args.root,
        round_number=args.round,
        test_case_ids=list(test_ids),
        dataset_root=dataset,
        test_dataset_root=test_dataset,
        harness_version="t2blendercodeharness-v5-executable-director",
        evaluator_version=VISUAL_PRIMARY_VERSION,
        blender_bin=args.blender_bin,
        workers=args.workers,
        timeout_s=args.render_timeout_s,
        provider_timeout_s=args.provider_timeout_s,
        vlm_model=args.visual_model,
        markdown_path=args.markdown_path,
        director_agent=director,
        code_agent=code_agent,
        provider_mode="glm",
        codex_command="codex",
        code_cache_dir=Path(args.root) / "code_cache",
        visual_provider=visual,
    )
    print(
        json.dumps(
            {
                "status": report["report"].get("status"),
                "case_count": report["report"].get("case_count"),
                "real_video_count": report["report"].get("real_video_count"),
                "vlm_scored_count": report["report"].get("vlm_scored_count"),
                "test_dataset_fingerprint": validation["fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
