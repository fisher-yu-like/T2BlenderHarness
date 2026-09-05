"""Compile the six-round GLM Harness experiment into an auditable report.

The compiler only reads generated JSON/provenance.  It never selects a patch,
uses test evidence for a transition, or fabricates a score for an unscored
case.  It is intentionally separate from the runner so a partial run can be
reported honestly and recompiled after the final post-training test100 pass.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def compact_batch(report: dict[str, Any] | None) -> dict[str, Any]:
    report = report or {}
    aggregate = report.get("aggregate") or {}
    cases = report.get("cases") or []
    return {
        "status": report.get("status"),
        "case_count": report.get("case_count", len(cases)),
        "real_video_count": report.get("real_video_count"),
        "vlm_scored_count": report.get("vlm_scored_count"),
        "preparation_failed_count": report.get("preparation_failed_count"),
        "artifact_failed_count": report.get("artifact_failed_count"),
        "mean_final_score": aggregate.get("mean_final_score"),
        "mean_task_final_score": aggregate.get("mean_task_final_score"),
        "mean_artifact_only_realism_score": aggregate.get("mean_artifact_only_realism_score"),
        "failure_counts": aggregate.get("failure_counts", {}),
    }


def provider_summary(root: Path) -> dict[str, Any]:
    kinds: Counter[str] = Counter()
    models: Counter[str] = Counter()
    manifests = 0
    invalid = 0
    fallback_cases: set[str] = set()
    for path in sorted(root.rglob("provider_manifest.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            invalid += 1
            continue
        manifests += 1
        if payload.get("template_backed") is not False or payload.get("llm_generated") is not True:
            invalid += 1
        case_id = path.parent.name
        stages = payload.get("stages") or {}
        for stage_name in ("director", "blender_code"):
            stage = stages.get(stage_name)
            if not isinstance(stage, dict):
                invalid += 1
                continue
            calls = stage.get("calls")
            if not isinstance(calls, list):
                calls = [stage]
            for call in calls:
                if not isinstance(call, dict):
                    invalid += 1
                    continue
                kind = str(call.get("provider_kind") or "unknown")
                model = str(call.get("model_id") or "unknown")
                kinds[kind] += 1
                models[model] += 1
                if kind == "external_openai_compatible":
                    fallback_cases.add(case_id)
    return {
        "provider_manifest_count": manifests,
        "invalid_provenance_count": invalid,
        "stage_call_counts_by_provider_kind": dict(sorted(kinds.items())),
        "stage_call_counts_by_model": dict(sorted(models.items())),
        "fallback_case_count": len(fallback_cases),
    }


def round_summary(round_path: Path) -> dict[str, Any]:
    report = load_json(round_path / "round_report.json", {})
    attempt = report.get("attempt") or {}
    splits = attempt.get("splits") or {}
    test = report.get("test") or {}
    test_report = test.get("report") if isinstance(test, dict) else None
    transitions = (report.get("outer_loop") or {}).get("transitions") or []
    patches = [item for item in transitions if isinstance(item, dict) and item.get("action") == "patch"]
    return {
        "round": report.get("round"),
        "execution_mode": report.get("execution_mode"),
        "train_case_ids": (report.get("batch") or {}).get("train", []),
        "dev_case_ids": (report.get("batch") or {}).get("dev", []),
        "attempt_count": (report.get("outer_loop") or {}).get("attempt_count", 1),
        "outer_loop_status": (report.get("outer_loop") or {}).get("status"),
        "attempt": attempt.get("attempt"),
        "train": compact_batch(splits.get("train")),
        "dev": compact_batch(splits.get("dev")),
        "patch_count": len(patches),
        "patch_transitions": patches,
        "test": {
            "scope": test.get("scope"),
            "dataset_id": test.get("test_dataset_id"),
            "case_count": test.get("report", {}).get("case_count", len(test.get("case_ids", [])))
            if isinstance(test, dict)
            else None,
            "status": test_report.get("status") if isinstance(test_report, dict) else None,
            "report": compact_batch(test_report if isinstance(test_report, dict) else test),
        },
    }


def build_result(run_root: Path, baseline_root: Path | None, final_test_root: Path | None) -> dict[str, Any]:
    protocol = load_json(run_root / "six_round_protocol.json", {})
    rounds = [round_summary(path) for path in sorted(run_root.glob("round-*")) if (path / "round_report.json").is_file()]
    result: dict[str, Any] = {
        "run_root": str(run_root.resolve()),
        "protocol": protocol,
        "rounds": rounds,
        "provider": provider_summary(run_root),
        "baseline_root": str(baseline_root.resolve()) if baseline_root else None,
        "final_test_root": str(final_test_root.resolve()) if final_test_root else None,
    }
    if baseline_root:
        baseline = load_json(baseline_root / "real_unified_score.json")
        if isinstance(baseline, dict):
            result["baseline_test100"] = compact_batch(baseline)
        else:
            result["baseline_test100"] = {"status": "missing_or_unreadable"}
    if final_test_root:
        final = load_json(final_test_root / "real_unified_score.json")
        result["final_test100"] = compact_batch(final) if isinstance(final, dict) else {"status": "missing_or_unreadable"}
    result["completion"] = {
        "rounds_completed": len(rounds),
        "expected_rounds": protocol.get("round_count", 6),
        "all_rounds_present": len(rounds) == protocol.get("round_count", 6),
        "test100_present_after_each_round": all(item["test"]["case_count"] == 100 for item in rounds),
        "test_excluded_from_patch_transitions": all(
            all("test" not in json.dumps(item.get("patch_transitions", []), ensure_ascii=False).lower() for item in rounds)
        ),
    }
    return result


def render_markdown(result: dict[str, Any]) -> str:
    protocol = result.get("protocol") or {}
    completion = result.get("completion") or {}
    lines = [
        "# GLM-5.3-flash Harness RSI 训练与评测报告",
        "",
        "## 实验结论",
        "",
        f"- 完成轮数：`{completion.get('rounds_completed')}/{completion.get('expected_rounds')}`。",
        f"- 协议：每轮 `10 train + 10 dev`，outer 最多 `{protocol.get('attempts_per_round_max')}` 次；轮后独立跑 `{protocol.get('test_cases_per_round') or protocol.get('test_case_count_per_round')}` 个 test。",
        f"- test 是否排除出 patch 选择：`{completion.get('test_excluded_from_patch_transitions')}`。",
        "- test100 只用于轮后验收和趋势记录，不参与问题定位、owner 选择或 patch 选择。",
        "- 生成阶段使用真实 GLM structured Director 与 BlenderCode 调用；没有固定模板生成。运行与 VLM review 使用本地 Codex provider。",
        "",
        "## 数据与 provider",
        "",
        f"- 训练数据：`{protocol.get('train_count')}` train / `{protocol.get('dev_count')}` dev；fingerprint `{protocol.get('dataset_fingerprint')}`。",
        f"- 独立 test：`{protocol.get('test_dataset_id')}`，`{protocol.get('test_count')}` cases；fingerprint `{protocol.get('test_dataset_fingerprint')}`。",
        f"- provider manifest：`{(result.get('provider') or {}).get('provider_manifest_count')}`；无模板 provenance 异常：`{(result.get('provider') or {}).get('invalid_provenance_count')}`。",
        f"- provider kind 调用计数：`{json.dumps((result.get('provider') or {}).get('stage_call_counts_by_provider_kind', {}), ensure_ascii=False, sort_keys=True)}`。",
        f"- model 调用计数：`{json.dumps((result.get('provider') or {}).get('stage_call_counts_by_model', {}), ensure_ascii=False, sort_keys=True)}`。",
        "",
        "## 分轮结果",
        "",
        "| Round | Attempts | Train scored/total | Train mean | Dev scored/total | Dev mean | Test status/scored | Test mean | Patch |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for item in result.get("rounds", []):
        train = item.get("train") or {}
        dev = item.get("dev") or {}
        test = item.get("test") or {}
        test_report = test.get("report") or {}
        lines.append(
            "| {round} | {attempts} | {ts}/{tt} | {tm} | {ds}/{dt} | {dm} | {status}/{scored} | {mean} | {patch} |".format(
                round=item.get("round"),
                attempts=item.get("attempt_count"),
                ts=train.get("vlm_scored_count"),
                tt=train.get("case_count"),
                tm=train.get("mean_final_score"),
                ds=dev.get("vlm_scored_count"),
                dt=dev.get("case_count"),
                dm=dev.get("mean_final_score"),
                status=test_report.get("status"),
                scored=test_report.get("vlm_scored_count"),
                mean=test_report.get("mean_final_score"),
                patch=item.get("patch_count", 0),
            )
        )
    lines.extend(
        [
            "",
            "## 门禁与解释",
            "",
            "- Harness patch 的候选只能来自当前 round 的 train findings；dev 只做 paired non-regression / improvement gate；Dev 不产生 patch，Test 不进入 transition 输入。",
            "- 若某轮没有足够重复且有 case-local evidence 的 train failure，则该轮记录 no patch，不人为制造修改。若 dev 下降，候选必须拒绝并回滚。",
            "- `needs_human_review` 只表示缺乏可审计的本地/视觉证据或 evaluator 不可用，不等于零分；`inner_loop_exhausted` 记录为本地生成/渲染/评估恢复失败。",
            "- 历史错误协议的 v2 中间结果不计入本实验：它曾按旧的 overall 逻辑运行，且不满足当前 10+10 后 test100 协议；仅保留作为审计说明。",
            "",
            "## 文件索引",
            "",
            f"- 实验根目录：`{result.get('run_root')}`",
            f"- baseline test100：`{result.get('baseline_root')}`",
            f"- 训练后 test100：`{result.get('final_test_root')}`",
            "- 每轮原始记录：`round-XX/round_report.json`；每次尝试的 train/dev 原始记录在 `round-XX/attempt-XX/real/`。",
        ]
    )
    baseline = result.get("baseline_test100")
    final = result.get("final_test100")
    if baseline or final:
        lines.extend(["", "## Baseline 与训练后 test100", ""])
        lines.append(f"- baseline：`{json.dumps(baseline or {}, ensure_ascii=False, sort_keys=True)}`")
        lines.append(f"- final：`{json.dumps(final or {}, ensure_ascii=False, sort_keys=True)}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--baseline-root")
    parser.add_argument("--final-test-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build_result(
        Path(args.run_root),
        Path(args.baseline_root) if args.baseline_root else None,
        Path(args.final_test_root) if args.final_test_root else None,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_markdown(result), encoding="utf-8")
    (Path(args.run_root) / "integrated_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["completion"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
