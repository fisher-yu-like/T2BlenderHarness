"""Run the bounded six-round GLM Harness evidence sequence.

Each round is executed in order: one 10-train/10-dev outer attempt, a
train-only Harness finding/proposal audit, then the independent frozen
100-case VBench test.  The test report is never passed to the proposal or
patch-selection boundary.  Checkpoints and the final Markdown document are
written after every completed round so a long experiment remains resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator.codex_visual import CodexVisualReviewProvider  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION  # noqa: E402
from scripts.evaluate_real_runs import discover_run_dirs, evaluate_real_split  # noqa: E402
from scripts.evaluate_real_videos import evaluate_split  # noqa: E402
from scripts.run_real_outer_loop import run_outer_loop  # noqa: E402
from scripts.train_real_harness import (  # noqa: E402
    _final_batch_status,
    build_dynamic_codex_agents,
    build_glm_six_round_manifest,
    merge_real_scores,
    require_benchmark_training_dataset,
    require_vbench_test100_dataset,
    run_bounded_outer_attempts,
    run_outer_attempt,
    run_round_test,
    write_unified_outputs,
    write_harness_memory_jsonl,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _mean_score(report: dict[str, Any]) -> float | None:
    cases = report.get("cases")
    values: list[float] = []
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict):
                continue
            value = case.get("overall_vlm_score")
            if value is None:
                value = case.get("task_final_score")
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
    return round(sum(values) / len(values), 4) if values else None


def _split_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report.get("status"),
        "case_count": report.get("case_count"),
        "real_video_count": report.get("real_video_count"),
        "vlm_scored_count": report.get("vlm_scored_count"),
        "vlm_unavailable_count": report.get("vlm_unavailable_count"),
        "needs_human_review_count": report.get("needs_human_review_count"),
        "mean_vlm_score": _mean_score(report),
        "report_path": str((Path(report.get("report_path")) if report.get("report_path") else Path()).resolve())
        if report.get("report_path")
        else None,
    }


def _round_summary(round_report: dict[str, Any]) -> dict[str, Any]:
    attempt = round_report.get("attempt") or {}
    splits = attempt.get("splits") if isinstance(attempt, dict) else {}
    combined = round_report.get("combined_splits") or {}
    test = round_report.get("test") or {}
    test_report = test.get("report") if isinstance(test, dict) else {}
    return {
        "round": round_report.get("round"),
        "attempt_count": round_report.get("outer_loop", {}).get("attempt_count", 1),
        "train": combined.get("train") or _split_summary(splits.get("train", {}) if isinstance(splits, dict) else {}),
        "dev": combined.get("dev") or _split_summary(splits.get("dev", {}) if isinstance(splits, dict) else {}),
        "test100": _split_summary(test_report if isinstance(test_report, dict) else {}),
        "patch_audit": round_report.get("patch_audit", {}),
    }


def _rendered_case_ids(run_root: Path) -> list[str]:
    """Return only cases with a real rendered run in the active batch."""

    return [
        str(json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))["case_id"])
        for run_dir in discover_run_dirs(run_root)
        if (run_dir / "proxy.mp4").is_file()
    ]


def _wait_for_assistant_reviews(
    review_dir: Path,
    *,
    case_ids: list[str],
    timeout_s: int,
) -> None:
    """Pause at the explicit Codex-assistant VLM boundary until reviews exist."""

    if not case_ids:
        return
    if timeout_s <= 0:
        raise ValueError("assistant review timeout must be positive")
    review_dir.mkdir(parents=True, exist_ok=True)
    status_path = review_dir / "review_wait_status.json"
    started = time.monotonic()
    while True:
        missing = [case_id for case_id in case_ids if not (review_dir / f"{case_id}.json").is_file()]
        status_path.write_text(
            json.dumps(
                {
                    "status": "waiting_for_codex_assistant_vlm" if missing else "reviews_ready",
                    "case_count": len(case_ids),
                    "completed_count": len(case_ids) - len(missing),
                    "missing_case_ids": missing,
                    "review_dir": str(review_dir.resolve()),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not missing:
            return
        if time.monotonic() - started >= timeout_s:
            raise TimeoutError(
                f"Codex assistant VLM reviews were not authored before timeout: {missing}"
            )
        time.sleep(5)


def _complete_assistant_batch(
    run_root: Path,
    *,
    dataset_root: Path,
    review_dir: Path,
    blender_bin: str,
    markdown_path: Path,
) -> dict[str, Any]:
    """Merge explicit Codex-assistant visual reviews into a real batch report."""

    preserved_path = run_root / "real_unified_score.json"
    preserved = _load_json(preserved_path)
    deterministic = evaluate_real_split(
        run_root,
        dataset_root=dataset_root,
        blender_bin=blender_bin,
    )
    reviewed = evaluate_split(
        run_root,
        dataset_root=dataset_root,
        assistant_local=True,
        assistant_review_dir=review_dir,
        scoring_policy=VISUAL_PRIMARY_VERSION,
        max_workers=1,
    )
    report = merge_real_scores(
        run_root=run_root,
        deterministic_results=deterministic,
        vlm_results=reviewed,
    )
    report.update(
        {
            "status": _final_batch_status(
                inner=preserved.get("inner_loop", {}),
                vlm_scored_count=report["vlm_scored_count"],
                real_video_count=report["real_video_count"],
            ),
            "render": preserved.get("render"),
            "agent_provenance": preserved.get("agent_provenance", []),
            "inner_loop": preserved.get("inner_loop", {}),
            "vlm_model": "codex-assistant",
            "vlm_call_policy": "codex_assistant_visual_review_of_exact_sampled_frames",
            "evaluator_version": VISUAL_PRIMARY_VERSION,
            "assistant_visual_review": {
                "provider": "codex-assistant",
                "review_dir": str(review_dir.resolve()),
                "generation_reused": True,
                "deterministic_evidence_reused": True,
            },
        }
    )
    write_unified_outputs(
        report,
        dataset_root=dataset_root,
        report_root=run_root,
        markdown_path=markdown_path,
    )
    preserved_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _write_training_document(
    path: Path,
    *,
    root: Path,
    protocol: dict[str, Any],
    round_reports: list[dict[str, Any]],
    status: str,
) -> None:
    lines = [
        "# GLM-5.3-flash Harness RSI training record",
        "",
        f"Status: `{status}`",
        "",
        "This record is generated from real benchmark prompts, GLM structured generation, real Blender rendering, and Codex-assistant visual review of exact sampled frames. No fixed-template generation or synthetic score is used.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{protocol.get('dataset_id')}`; train={protocol.get('train_count')}, dev={protocol.get('dev_count')}",
        f"- Frozen test: `{protocol.get('test_dataset_id')}`; {protocol.get('test_count')} cases after every round",
        f"- Rounds: {protocol.get('round_count')}; each round uses 10 train + 10 dev",
        f"- Outer attempts: at most {protocol.get('attempts_per_round_max')}; inner case recovery: at most {protocol.get('attempt_policy', {}).get('inner_case_attempts_max')}",
        "- Test100 is evaluation-only and is excluded from issue localization, owner choice, and patch selection.",
        "- Harness patches, if any, must be sourced from repeated train evidence and pass the dev non-regression gate.",
        "",
        "## Round results",
        "",
        "| Round | Train | Dev | Frozen test100 | Patch audit |",
        "|---:|---|---|---|---|",
    ]
    for report in round_reports:
        summary = _round_summary(report)
        def cell(name: str) -> str:
            item = summary[name]
            return f"{item.get('status')} / VLM {item.get('vlm_scored_count')}/{item.get('real_video_count')} / mean {item.get('mean_vlm_score')}"

        audit = summary.get("patch_audit") or {}
        lines.append(
            f"| {summary.get('round')} | {cell('train')} | {cell('dev')} | {cell('test100')} | {audit.get('status')} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Experiment root: `{root.resolve()}`",
            f"- Protocol: `{(root / 'six_round_protocol.json').resolve()}`",
            f"- Checkpoint: `{(root / 'six_round_training_report.json').resolve()}`",
            f"- Memory JSONL: `{(root / 'memory' / 'harness_updates.jsonl').resolve()}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_checkpoint(
    root: Path,
    *,
    protocol: dict[str, Any],
    round_reports: list[dict[str, Any]],
    markdown_path: Path,
    status: str,
) -> None:
    payload = {
        "status": status,
        "protocol": protocol,
        "provider_mode": "glm",
        "generation": {
            "director_model": "glm-5.3-flash",
            "blender_code_model": "glm-5.3-flash",
            "template_backed": False,
            "llm_generated": True,
        },
        "visual_review": {
            "provider": "codex-assistant",
            "evaluation_only": False,
        },
        "rounds": round_reports,
        "completed_round_count": len(round_reports),
        "training_document": str(markdown_path.resolve()),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "six_round_training_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    memory_rows = [
        {
            "round": report.get("round"),
            "patch": report.get("patch_audit", {}),
            "splits": (report.get("attempt") or {}).get("splits", {}),
            "test_evaluation": report.get("test", {}),
        }
        for report in round_reports
    ]
    write_harness_memory_jsonl(root / "memory" / "harness_updates.jsonl", memory_rows)
    _write_training_document(
        markdown_path,
        root=root,
        protocol=protocol,
        round_reports=round_reports,
        status=status,
    )


def run_sequence(
    *,
    root: str | Path,
    dataset_root: str | Path,
    test_dataset_root: str | Path,
    blender_bin: str,
    workers: int,
    render_timeout_s: int,
    provider_timeout_s: int,
    visual_timeout_s: int,
    visual_fallback_timeout_s: int,
    visual_frame_budget: int,
    visual_model: str,
    codex_command: str,
    prepare_workers: int,
    markdown_path: str | Path,
    vlm_provider: str = "assistant",
    assistant_review_timeout_s: int = 7200,
    start_round: int = 1,
    prior_round_report: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    root_path = Path(root)
    dataset = Path(dataset_root)
    test_dataset = Path(test_dataset_root)
    training_validation = require_benchmark_training_dataset(dataset)
    test_validation = require_vbench_test100_dataset(test_dataset)
    train_splits = _load_json(dataset / "splits.json")
    test_splits = _load_json(test_dataset / "splits.json")
    training_metadata = _load_json(dataset / "metadata.json")
    test_metadata = _load_json(test_dataset / "metadata.json")
    protocol = build_glm_six_round_manifest(
        train_splits["train"],
        train_splits["dev"],
        test_splits["test"],
        dataset_fingerprint=training_metadata["fingerprint"],
        test_dataset_id=test_metadata["dataset_id"],
        test_dataset_fingerprint=test_validation["fingerprint"],
    )
    protocol["provider_mode"] = "glm"
    protocol["training_validation"] = training_validation
    protocol["test_validation"] = test_validation
    root_path.mkdir(parents=True, exist_ok=True)
    (root_path / "six_round_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    round_reports: list[dict[str, Any]] = []
    if prior_round_report is not None:
        prior = _load_json(Path(prior_round_report))
        if isinstance(prior.get("round"), int):
            round_reports.append(prior)
        elif isinstance(prior.get("rounds"), list):
            round_reports.extend(item for item in prior["rounds"] if isinstance(item, dict))

    director_agent, code_agent = build_dynamic_codex_agents(
        codex_command=codex_command,
        timeout_s=provider_timeout_s,
        provider_mode="glm",
    )
    if vlm_provider not in {"assistant", "codex-cli"}:
        raise ValueError("vlm_provider must be assistant or codex-cli")
    visual_provider = (
        CodexVisualReviewProvider(
            command=codex_command,
            model=visual_model,
            timeout_s=visual_timeout_s,
            fallback_timeout_s=visual_fallback_timeout_s,
            visual_frame_budget=visual_frame_budget,
        )
        if vlm_provider == "codex-cli"
        else None
    )
    markdown = Path(markdown_path)
    for batch in protocol["rounds"]:
        round_number = int(batch["round"])
        if round_number < int(start_round):
            continue
        round_root = root_path / f"round-{round_number:02d}"
        existing_path = round_root / "round_report.json"
        if resume and existing_path.is_file():
            existing = _load_json(existing_path)
            if int(existing.get("round", -1)) == round_number:
                round_reports = [item for item in round_reports if int(item.get("round", -1)) != round_number]
                round_reports.append(existing)
                round_reports.sort(key=lambda item: int(item.get("round", 0)))
                _write_checkpoint(root_path, protocol=protocol, round_reports=round_reports, markdown_path=markdown, status="running")
                continue

        def run_attempt(attempt_number: int) -> dict[str, Any]:
            review_root = round_root / f"attempt-{attempt_number:02d}" / "assistant_reviews"
            attempt_result = run_outer_attempt(
                root_path,
                round_number=round_number,
                attempt_number=attempt_number,
                dataset_root=dataset,
                harness_version="t2blendercodeharness-v5-executable-director",
                evaluator_version=VISUAL_PRIMARY_VERSION,
                blender_bin=blender_bin,
                workers=workers,
                timeout_s=render_timeout_s,
                provider_timeout_s=provider_timeout_s,
                vlm_model=visual_model,
                markdown_path=markdown,
                director_agent=director_agent,
                code_agent=code_agent,
                provider_mode="glm",
                codex_command=codex_command,
                code_cache_dir=root_path / "code_cache",
                visual_provider=visual_provider,
                prepare_workers=prepare_workers,
                assistant_local=vlm_provider == "assistant",
                assistant_review_dir=review_root if vlm_provider == "assistant" else None,
                visual_frame_budget=visual_frame_budget,
            )
            if vlm_provider == "assistant":
                for split in ("train", "dev"):
                    split_root = round_root / f"attempt-{attempt_number:02d}" / "real" / split
                    review_dir = review_root / split
                    case_ids = _rendered_case_ids(split_root)
                    _wait_for_assistant_reviews(
                        review_dir,
                        case_ids=case_ids,
                        timeout_s=assistant_review_timeout_s,
                    )
                    if case_ids:
                        attempt_result["splits"][split] = _complete_assistant_batch(
                            split_root,
                            dataset_root=dataset,
                            review_dir=review_dir,
                            blender_bin=blender_bin,
                            markdown_path=markdown,
                        )
                attempt_report_path = (
                    round_root / f"attempt-{attempt_number:02d}" / "attempt_report.json"
                )
                attempt_report_path.write_text(
                    json.dumps(attempt_result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return attempt_result

        per_attempt_audits: list[dict[str, Any]] = []

        def transition(attempt_number: int, reports: list[dict[str, Any]]) -> dict[str, Any]:
            del reports
            attempt_root = round_root / f"attempt-{attempt_number:02d}" / "real"
            patch_audit_path = round_root / f"attempt-{attempt_number:02d}" / "harness_proposal.json"
            patch_audit = run_outer_loop(
                attempt_root / "train",
                attempt_root / "dev",
                patch_audit_path,
                forbidden_case_ids=set(test_splits["test"]),
            )
            per_attempt_audits.append(
                {
                    "attempt": attempt_number,
                    "path": str(patch_audit_path.resolve()),
                    "audit": patch_audit,
                }
            )
            if patch_audit.get("status") == "no_patch":
                return {
                    "action": "stop",
                    "status": "round_completed_without_harness_patch",
                    "reason": "no repeated train-only failure with sufficient evidence",
                    "source_split": "train",
                    "test_used_for_selection": False,
                }
            # This unattended runner records a valid proposal but never
            # fabricates or auto-applies source edits.  A supplied Host patch
            # controller can return {action: patch} here and the bounded
            # state machine will execute attempts 2--5; test100 remains out of
            # this callback by construction.
            return {
                "action": "stop",
                "status": "proposal_recorded_for_harness_owner",
                "reason": "train evidence requires a Harness-owner patch controller before another attempt",
                "source_split": "train",
                "test_used_for_selection": False,
            }

        outer_loop = run_bounded_outer_attempts(
            run_attempt=run_attempt,
            transition=transition,
            max_attempts=5,
        )
        attempt = outer_loop["reports"][-1]
        patch_audit = {
            "status": "no_patch" if all(item["audit"].get("status") == "no_patch" for item in per_attempt_audits) else "proposal_recorded",
            "attempts": per_attempt_audits,
            "test_used_for_selection": False,
        }
        test_review_dir = round_root / "test" / "assistant_reviews"
        test_evaluation = run_round_test(
            root_path,
            round_number=round_number,
            test_case_ids=list(batch["test_evaluation"]["test_cases"]),
            dataset_root=dataset,
            test_dataset_root=test_dataset,
            harness_version="t2blendercodeharness-v5-executable-director",
            evaluator_version=VISUAL_PRIMARY_VERSION,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=render_timeout_s,
            provider_timeout_s=provider_timeout_s,
            vlm_model=visual_model,
            markdown_path=markdown,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode="glm",
            codex_command=codex_command,
            code_cache_dir=root_path / "code_cache",
            visual_provider=visual_provider,
            prepare_workers=prepare_workers,
            assistant_local=vlm_provider == "assistant",
            assistant_review_dir=test_review_dir if vlm_provider == "assistant" else None,
            visual_frame_budget=visual_frame_budget,
        )
        if vlm_provider == "assistant":
            test_root = round_root / "test" / "real"
            test_case_ids = _rendered_case_ids(test_root)
            _wait_for_assistant_reviews(
                test_review_dir,
                case_ids=test_case_ids,
                timeout_s=assistant_review_timeout_s,
            )
            if test_case_ids:
                test_evaluation["report"] = _complete_assistant_batch(
                    test_root,
                    dataset_root=test_dataset,
                    review_dir=test_review_dir,
                    blender_bin=blender_bin,
                    markdown_path=markdown,
                )
                (round_root / "test_report.json").write_text(
                    json.dumps(test_evaluation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        round_report = {
            "round": round_number,
            "execution_mode": "unattended_evidence_training",
            "batch": batch,
            "attempt": attempt,
            "outer_loop": {
                **outer_loop,
                "max_attempts": 5,
                "patch_audit": patch_audit,
                "test_excluded": True,
            },
            "patch_audit": patch_audit,
            "test": test_evaluation,
        }
        round_reports = [item for item in round_reports if int(item.get("round", -1)) != round_number]
        round_reports.append(round_report)
        round_reports.sort(key=lambda item: int(item.get("round", 0)))
        round_root.mkdir(parents=True, exist_ok=True)
        (round_root / "round_report.json").write_text(
            json.dumps(round_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_checkpoint(root_path, protocol=protocol, round_reports=round_reports, markdown_path=markdown, status="running")

    _write_checkpoint(root_path, protocol=protocol, round_reports=round_reports, markdown_path=markdown, status="complete")
    return _load_json(root_path / "six_round_training_report.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
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
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument(
        "--vlm-provider",
        choices=["assistant", "codex-cli"],
        default="assistant",
        help="visual reviewer; assistant uses Codex-authored local frame reviews",
    )
    parser.add_argument("--assistant-review-timeout-s", type=int, default=7200)
    parser.add_argument("--markdown-path", required=True)
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--prior-round-report")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    result = run_sequence(
        root=args.root,
        dataset_root=args.dataset_root,
        test_dataset_root=args.test_dataset_root,
        blender_bin=args.blender_bin,
        workers=args.workers,
        render_timeout_s=args.render_timeout_s,
        provider_timeout_s=args.provider_timeout_s,
        visual_timeout_s=args.visual_timeout_s,
        visual_fallback_timeout_s=args.visual_fallback_timeout_s,
        visual_frame_budget=args.visual_frame_budget,
        visual_model=args.visual_model,
        codex_command=args.codex_command,
        prepare_workers=args.prepare_workers,
        markdown_path=args.markdown_path,
        vlm_provider=args.vlm_provider,
        assistant_review_timeout_s=args.assistant_review_timeout_s,
        start_round=args.start_round,
        prior_round_report=args.prior_round_report,
        resume=not args.no_resume,
    )
    print(json.dumps({"status": result.get("status"), "completed_round_count": result.get("completed_round_count")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
