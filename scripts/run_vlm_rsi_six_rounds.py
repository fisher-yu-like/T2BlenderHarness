"""Launch the explicit VLM-backed Harness-RSI six-round research run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.codex_visual import CodexVisualReviewProvider  # noqa: E402
from scripts.train_real_harness import (  # noqa: E402
    build_dynamic_codex_agents,
    run_six_round_protocol,
)
from videoact.vlm_rsi import CodexHarnessPatchAgent, VlmRsiTransitionController  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=r"D:\harness-rsi-training\vlm-rsi-six-rounds-20260905",
    )
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--test-dataset-root", default="dataset/vbench2-agent-test-100-v1")
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--provider-mode", default="model", choices=["model"])
    parser.add_argument("--vlm-model", default="gpt-5.6-luna")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--provider-timeout-s", type=int, default=1800)
    parser.add_argument("--visual-timeout-s", type=int, default=1800)
    parser.add_argument("--visual-frame-budget", type=int, default=8)
    parser.add_argument("--test-schedule", default="baseline_final_only", choices=["baseline_final_only"])
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--dry-protocol", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = Path(args.output_root).resolve()
    markdown_path = Path(args.markdown_path).resolve() if args.markdown_path else output_root / "harness_training_memory.md"
    os.environ["OPENAI_VLM_MODEL"] = args.vlm_model
    os.environ["CODEX_VLM_FALLBACK_MODEL"] = ""

    visual_provider = CodexVisualReviewProvider(
        command=args.codex_command,
        timeout_s=args.visual_timeout_s,
        model=args.vlm_model,
        fallback_model="",
        visual_frame_budget=args.visual_frame_budget,
    )
    director_agent, code_agent = build_dynamic_codex_agents(
        codex_command=args.codex_command,
        timeout_s=args.provider_timeout_s,
        provider_mode=args.provider_mode,
    )
    patch_agent = CodexHarnessPatchAgent(
        repo_root=ROOT,
        command=args.codex_command,
        model=args.vlm_model,
        timeout_s=args.provider_timeout_s,
    )
    controller = VlmRsiTransitionController(
        output_dir=output_root,
        repo_root=ROOT,
        coding_agent=patch_agent,
    )
    if args.dry_protocol:
        from scripts.train_real_harness import build_glm_six_round_manifest, prepare_vlm_rsi_protocol

        train = [f"train-{index:02d}" for index in range(60)]
        dev = [f"dev-{index:02d}" for index in range(60)]
        test = [f"test-{index:03d}" for index in range(100)]
        protocol = build_glm_six_round_manifest(
            train,
            dev,
            test,
            dataset_fingerprint="dry-run",
            test_dataset_id="dry-test",
            test_dataset_fingerprint="dry-test-fingerprint",
        )
        print(json.dumps(prepare_vlm_rsi_protocol(protocol, visual_provider=visual_provider), indent=2))
        return 0

    result = run_six_round_protocol(
        output_root,
        dataset_root=args.dataset_root,
        test_dataset_root=args.test_dataset_root,
        harness_version="harness-rsi-vlm-rsi-20260905",
        evaluator_version="visual-primary-v7",
        blender_bin=args.blender_bin,
        workers=args.workers,
        timeout_s=args.timeout_s,
        provider_timeout_s=args.provider_timeout_s,
        vlm_model=args.vlm_model,
        codex_command=args.codex_command,
        markdown_path=markdown_path,
        director_agent=director_agent,
        code_agent=code_agent,
        provider_mode=args.provider_mode,
        outer_transition=controller,
        diagnostic_only=False,
        visual_provider=visual_provider,
        test_schedule=args.test_schedule,
        ai_only_vlm_rsi=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

