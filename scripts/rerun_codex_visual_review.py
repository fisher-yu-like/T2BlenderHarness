"""Re-run local Codex visual review for already rendered cases.

This recovery tool is for a visual-review interruption only.  It never
regenerates a DirectorPlan or Blender source and never changes deterministic
evidence.  It uses independent Codex sessions per case and a bounded fallback
model, then merges the new visual results with the preserved batch report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator.codex_visual import CodexVisualReviewProvider  # noqa: E402
from scripts.evaluate_real_videos import evaluate_split  # noqa: E402
from scripts.train_real_harness import (  # noqa: E402
    _final_batch_status,
    merge_real_scores,
    write_unified_outputs,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"batch report is missing or unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"batch report must be an object: {path}")
    return payload


def _deterministic_from_batch_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("preserved batch report has no deterministic case rows")
    results = []
    for case in cases:
        if not isinstance(case, dict) or not str(case.get("case_id") or "").strip():
            continue
        results.append(
            {
                "case_id": case["case_id"],
                "status": case.get("deterministic_status") or "not_run",
                "score": case.get("deterministic_score"),
                "artifact_status": case.get("artifact_status"),
                "findings": case.get("deterministic_findings", []),
                "finding_details": case.get("deterministic_finding_details", []),
                "director_plan_score": case.get("director_plan_score"),
                "director_findings": case.get("director_findings", []),
                "interaction_findings": case.get("interaction_findings", []),
            }
        )
    if not results:
        raise ValueError("preserved batch report has no valid deterministic case rows")
    return results


def _preserved_visual_results(root: Path, case_ids: set[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case_id in sorted(case_ids):
        path = root / case_id / "vlm_report.json"
        payload = _load(path)
        results.append({"case_id": case_id, **payload})
    return results


def rerun_visual_review(
    batch_root: str | Path,
    *,
    dataset_root: str | Path,
    codex_command: str = "codex",
    visual_model: str = "gpt-5.6-terra",
    visual_timeout_s: int = 300,
    workers: int = 4,
    case_ids: list[str] | None = None,
    visual_frame_budget: int = 8,
    fallback_timeout_s: int | None = None,
) -> dict[str, Any]:
    root = Path(batch_root)
    preserved_path = root / "real_unified_score.json"
    preserved = _load(preserved_path)
    deterministic = _deterministic_from_batch_report(preserved)
    provider = CodexVisualReviewProvider(
        command=codex_command,
        timeout_s=visual_timeout_s,
        model=visual_model,
        visual_frame_budget=visual_frame_budget,
        fallback_timeout_s=fallback_timeout_s,
    )
    discovered_case_ids = {
        str(item.get("case_id"))
        for item in (preserved.get("cases") or [])
        if isinstance(item, dict) and str(item.get("case_id") or "").strip()
    }
    # A preliminary batch intentionally writes no vlm_report.json before the
    # visual pass starts.  Recovery must therefore key availability on the
    # rendered run manifest, not on the existence of a prior VLM report.
    rendered_case_ids = {
        str(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["case_id"])
        for run_dir in root.iterdir()
        if run_dir.is_dir() and (run_dir / "run_manifest.json").is_file()
    }
    available_case_ids = discovered_case_ids & rendered_case_ids
    selected_case_ids = set(case_ids) if case_ids is not None else available_case_ids
    if not selected_case_ids.issubset(available_case_ids):
        raise ValueError(
            "visual recovery case IDs need an existing rendered run and preserved report: "
            f"{sorted(selected_case_ids - available_case_ids)}"
        )
    visual_results = evaluate_split(
        root,
        dataset_root=dataset_root,
        provider=provider,
        max_workers=workers,
        case_ids=sorted(selected_case_ids),
        visual_frame_budget=visual_frame_budget,
    )
    preserved_case_ids = {
        case_id
        for case_id in (available_case_ids - selected_case_ids)
        if (root / case_id / "vlm_report.json").is_file()
    }
    visual_results.extend(_preserved_visual_results(root, preserved_case_ids))
    report = merge_real_scores(
        run_root=root,
        deterministic_results=deterministic,
        vlm_results=visual_results,
    )
    inner = preserved.get("inner_loop") or {}
    report.update(
        {
            "status": _final_batch_status(
                inner=inner,
                vlm_scored_count=report["vlm_scored_count"],
                real_video_count=report["real_video_count"],
            ),
            "render": preserved.get("render"),
            "agent_provenance": preserved.get("agent_provenance", []),
            "inner_loop": inner,
            "vlm_model": "codex_local_visual_review",
            "vlm_call_policy": "local_codex_visual_review_only; independent_case_sessions_with_bounded_fallback",
            "evaluator_version": preserved.get("evaluator_version"),
            "visual_review_recovery": {
                "source_report": str(preserved_path.resolve()),
                "provider": "codex_local_visual_review",
                "visual_model": visual_model,
                "visual_timeout_s": visual_timeout_s,
                "max_workers": workers,
                "visual_frame_budget": visual_frame_budget,
                "fallback_timeout_s": fallback_timeout_s,
                "selected_case_ids": sorted(selected_case_ids),
                "preserved_case_ids": sorted(preserved_case_ids),
                "rendered_case_ids": sorted(available_case_ids),
                "deterministic_evidence_reused": True,
                "generation_reused": True,
            },
        }
    )
    preserved_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_unified_outputs(report, dataset_root=dataset_root, report_root=root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--visual-model", default="gpt-5.6-terra")
    parser.add_argument("--visual-timeout-s", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--visual-frame-budget", type=int, default=8)
    parser.add_argument("--fallback-timeout-s", type=int, default=60)
    parser.add_argument("--case-id", action="append", default=None)
    args = parser.parse_args()
    report = rerun_visual_review(
        args.batch_root,
        dataset_root=args.dataset_root,
        codex_command=args.codex_command,
        visual_model=args.visual_model,
        visual_timeout_s=args.visual_timeout_s,
        workers=args.workers,
        case_ids=args.case_id,
        visual_frame_budget=args.visual_frame_budget,
        fallback_timeout_s=args.fallback_timeout_s,
    )
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "case_count": report.get("case_count"),
                "real_video_count": report.get("real_video_count"),
                "vlm_scored_count": report.get("vlm_scored_count"),
                "aggregate": report.get("aggregate", {}),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
