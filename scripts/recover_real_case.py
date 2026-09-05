"""Regenerate selected real cases through the existing bounded inner loop."""

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

from scripts.train_real_harness import build_dynamic_codex_agents, run_real_batch_with_inner_loop  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--render-timeout-s", type=int, default=900)
    parser.add_argument("--provider-timeout-s", type=int, default=180)
    parser.add_argument("--harness-version", default="t2blendercodeharness-v5-executable-director")
    parser.add_argument("--evaluator-version", default="visual-primary-v7")
    parser.add_argument("--markdown-path", required=True)
    args = parser.parse_args()

    output = Path(args.output_root)
    dataset = Path(args.dataset_root)
    director, code_agent = build_dynamic_codex_agents(
        codex_command="codex",
        timeout_s=args.provider_timeout_s,
        provider_mode="glm",
    )
    report = run_real_batch_with_inner_loop(
        output,
        split=args.split,
        case_ids=list(args.case_id),
        dataset_root=dataset,
        harness_version=args.harness_version,
        evaluator_version=args.evaluator_version,
        blender_bin=args.blender_bin,
        workers=args.workers,
        timeout_s=args.render_timeout_s,
        vlm_model="gpt-5.6-terra",
        markdown_path=args.markdown_path,
        director_agent=director,
        code_agent=code_agent,
        provider_mode="glm",
        code_cache_dir=output.parent.parent.parent / "code_cache" / "case-recovery",
        max_inner_attempts=3,
        visual_provider=None,
    )
    payload = {
        "split": args.split,
        "case_ids": list(args.case_id),
        "report": report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "case_recovery_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "real_video_count": report.get("real_video_count"),
                "inner_status": (report.get("inner_loop") or {}).get("status"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
