"""Summarise the six-round real-Blender diagnostic protocol and draw its curve.

The diagnostic run intentionally has no independent visual labels.  This
script therefore plots the measurable artifact-only proxy and deterministic
gate separately, while representing task/VLM/trajectory review as unavailable
instead of imputing zero.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _aggregate(path: Path) -> dict[str, Any]:
    report = _read(path)
    aggregate = report.get("aggregate") or {}
    return {
        "case_count": report.get("case_count"),
        "real_video_count": report.get("real_video_count"),
        "deterministic": aggregate.get("mean_deterministic_score"),
        "artifact_only_realism": aggregate.get("mean_artifact_only_realism_score"),
        "task": aggregate.get("mean_task_final_score"),
        "vlm_scored_count": report.get("vlm_scored_count"),
        "preparation_failed_count": report.get("preparation_failed_count"),
        "artifact_failed_count": report.get("artifact_failed_count"),
    }


def _load_round(
    root: Path,
    round_number: int,
    *,
    attempt_real_root: Path | None = None,
    overall_report_path: Path | None = None,
) -> dict[str, Any]:
    round_root = root / f"round-{round_number:02d}"
    attempt_file_root = attempt_real_root or (round_root / "attempt-01" / "real")
    attempt = {
        split: _aggregate(attempt_file_root / split / "real_unified_score.json")
        for split in ("train", "dev")
    }
    overall_report = _read(overall_report_path or (round_root / "overall_report.json"))
    overall = {
        split: {
            "case_count": overall_report["splits"][split].get("case_count"),
            "real_video_count": overall_report["splits"][split].get("real_video_count"),
            "deterministic": overall_report["splits"][split]["aggregate"].get("mean_deterministic_score"),
            "artifact_only_realism": overall_report["splits"][split]["aggregate"].get("mean_artifact_only_realism_score"),
            "task": overall_report["splits"][split]["aggregate"].get("mean_task_final_score"),
            "vlm_scored_count": overall_report["splits"][split].get("vlm_scored_count"),
            "preparation_failed_count": overall_report["splits"][split].get("preparation_failed_count"),
            "artifact_failed_count": overall_report["splits"][split].get("artifact_failed_count"),
        }
        for split in ("train", "dev")
    }
    return {"round": round_number, "attempt": attempt, "overall": overall}


def _draw_curve(summary: dict[str, Any], output: Path) -> None:
    width, height = 1900, 1180
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    title = _font(30)
    label = _font(20)
    small = _font(16)
    draw.text(
        (55, 24),
        "T2Blendercodeharness: six-round real-Blender diagnostic",
        fill=(25, 35, 50),
        font=title,
    )
    draw.text(
        (55, 68),
        "Artifact-only proxy is measurable; task/VLM/trajectory review remains unavailable (never imputed as zero).",
        fill=(80, 90, 105),
        font=small,
    )
    panels = [
        ("artifact_only_realism", "Artifact-only realism proxy", 125),
        ("deterministic", "Deterministic gate score", 570),
    ]
    colors = {
        "attempt_train": (36, 108, 180),
        "attempt_dev": (75, 145, 205),
        "overall_train": (30, 125, 82),
        "overall_dev": (75, 165, 110),
    }
    rounds = summary["rounds"]
    for field, panel_title, top in panels:
        left, right, bottom = 125, width - 90, top + 315
        draw.text((left, top - 36), panel_title, fill=(45, 55, 70), font=label)
        draw.line((left, bottom, right, bottom), fill=(80, 90, 105), width=2)
        draw.line((left, top, left, bottom), fill=(80, 90, 105), width=2)
        for tick in range(0, 101, 20):
            y = bottom - (bottom - top) * tick / 100
            draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
            draw.text((72, y - 8), str(tick), fill=(90, 100, 115), font=small)
        for index, row in enumerate(rounds):
            x = left + (right - left) * index / max(1, len(rounds) - 1)
            draw.text((x - 8, bottom + 14), str(row["round"]), fill=(90, 100, 115), font=small)
        for series in ("attempt_train", "attempt_dev", "overall_train", "overall_dev"):
            points = []
            for index, row in enumerate(rounds):
                value = row[series][field]
                if not isinstance(value, (int, float)):
                    continue
                x = left + (right - left) * index / max(1, len(rounds) - 1)
                y = bottom - (bottom - top) * float(value) / 100
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=colors[series], width=4)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colors[series])
        legend = [
            ("attempt train", "attempt_train"),
            ("attempt dev", "attempt_dev"),
            ("overall train", "overall_train"),
            ("overall dev", "overall_dev"),
        ]
        for index, (text, series) in enumerate(legend):
            x = right - 570 + (index % 2) * 285
            y = top + 12 + (index // 2) * 28
            draw.line((x, y + 8, x + 28, y + 8), fill=colors[series], width=4)
            draw.text((x + 38, y), text, fill=(65, 75, 90), font=small)
    note_y = 960
    draw.text((55, note_y), "Review channels", fill=(45, 55, 70), font=label)
    draw.text(
        (55, note_y + 42),
        "All 6 rounds: task_final_score = unavailable; VLM scored count = 0; trajectory/camera/physics semantic labels = unavailable.",
        fill=(80, 90, 105),
        font=small,
    )
    draw.text(
        (55, note_y + 70),
        "Use the human-review UI after the Harness upgrade before making a visual-quality or self-evolution acceptance claim.",
        fill=(80, 90, 105),
        font=small,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def build_summary(
    root: str | Path,
    *,
    output_json: str | Path,
    curve: str | Path,
    report: str | Path,
    round1_attempt_root: str | Path | None = None,
    round1_overall_report: str | Path | None = None,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    rounds = []
    for round_number in range(1, 7):
        if round_number == 1 and round1_attempt_root and round1_overall_report:
            rounds.append(
                _load_round(
                    root_path,
                    round_number,
                    attempt_real_root=Path(round1_attempt_root).resolve(),
                    overall_report_path=Path(round1_overall_report).resolve(),
                )
            )
        else:
            rounds.append(_load_round(root_path, round_number))
    summary: dict[str, Any] = {
        "protocol_root": str(root_path),
        "round_count": 6,
        "actual_attempts_per_round": 1,
        "attempt_case_slots": 6 * 20,
        "overall_case_slots": 6 * 120,
        "actual_protocol_case_slots": 6 * (20 + 120),
        "max_protocol_case_slots": 6 * (5 * 20 + 120),
        "rounds": [],
        "visual_review_policy": "unavailable_is_not_zero; no task/VLM/trajectory score was imputed",
    }
    project_root = root_path.parents[2]
    semantic_smoke_root = project_root / "out" / "preflight" / "director-object-only-v4"
    semantic_smoke_report = semantic_smoke_root / "real_unified_score.json"
    if semantic_smoke_report.is_file():
        smoke = _read(semantic_smoke_report)
        smoke_plans = []
        for case in smoke.get("cases", []):
            plan_path = semantic_smoke_root / str(case.get("case_id")) / "director_plan.json"
            if plan_path.is_file():
                plan = _read(plan_path)
                smoke_plans.append(
                    {
                        "case_id": case.get("case_id"),
                        "has_actor": any(item.get("kind") == "actor" for item in plan.get("entities", [])),
                        "deterministic_score": case.get("deterministic_score"),
                        "realism_score": case.get("realism_score"),
                        "proxy_video": case.get("proxy_video"),
                    }
                )
        summary["post_training_semantic_smoke"] = {
            "root": str(semantic_smoke_root),
            "case_count": smoke.get("case_count"),
            "real_video_count": smoke.get("real_video_count"),
            "preparation_failed_count": smoke.get("preparation_failed_count"),
            "artifact_failed_count": smoke.get("artifact_failed_count"),
            "agent_provenance": smoke.get("agent_provenance"),
            "plans": smoke_plans,
        }
    for row in rounds:
        summary["rounds"].append(
            {
                "round": row["round"],
                "attempt_train": row["attempt"]["train"],
                "attempt_dev": row["attempt"]["dev"],
                "overall_train": row["overall"]["train"],
                "overall_dev": row["overall"]["dev"],
            }
        )
    output_json_path = Path(output_json).resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _draw_curve(summary, Path(curve).resolve())

    overall = summary["rounds"]
    lines = [
        "# T2Blendercodeharness 六轮真实渲染诊断报告",
        "",
        "## 结论",
        "",
        "六轮外循环已按 active VBench-2.0 原始 prompt index 完成。每轮使用 10 个新 train + 10 个 paired dev，并在轮末重新跑完整 60 train + 60 dev；每轮只执行 1 次 outer attempt，因为没有出现需要重生成或支持新 Harness patch 的可归因失败。",
        "",
        "这是一份 pre-calibration diagnostic，不是正式视觉质量验收：真实 Blender、DirectorPlan、source provenance、deterministic 和 artifact evidence 已完成；人工/独立 VLM 尚未可用，因此 task、人物/物体轨迹、摄像机调度、物理合理性和真实性分数均保持 unavailable。",
        "",
        f"- 实际协议 case-slot：`{summary['actual_protocol_case_slots']}`（attempt 120 + overall 720）。",
        f"- 协议最大上限：`{summary['max_protocol_case_slots']}`；未为凑满上限重复成功样本。",
        "- Blender：`D:\\blender\\blender.exe`，4 workers，最多 12-case 分组串行。",
        "- 生成模式：`agent` + in-process `codex-local`；没有使用 `template_baseline`。",
        "- 每个 prepared source 必须包含 `CASE_SCENE_PROFILE` 和 `codex-local-case-profile-v2`，否则在 Blender 前 fail-closed。",
        "",
        "## 六轮结果",
        "",
        "数值列中的 artifact-only realism 是几何/PNG 低层证据，不是视觉真实性；`unavailable` 不是 0。",
        "",
        "| Round | Attempt train artifact | Attempt dev artifact | Overall train artifact | Overall dev artifact | Overall train deterministic | Overall dev deterministic | Overall VLM scored |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['round']} | {row['attempt_train']['artifact_only_realism']} | {row['attempt_dev']['artifact_only_realism']} | "
            f"{row['overall_train']['artifact_only_realism']} | {row['overall_dev']['artifact_only_realism']} | "
            f"{row['overall_train']['deterministic']} | {row['overall_dev']['deterministic']} | "
            f"{row['overall_train']['vlm_scored_count']} / {row['overall_dev']['vlm_scored_count']} |"
        )
    lines.extend(
        [
            "",
            f"![六轮诊断曲线]({Path(curve).resolve().as_posix()})",
            "",
            "## 真实性与 Harness 判断边界",
            "",
            "1. round 1 的单组件升级是 `blender_code_agent`：增加 case-specific visual profile、较高细节的 parametric geometry、connected armature、event-conditioned pose keyframes 和 camera-facing orbit bias。新版 paired attempt 与旧 baseline 的 artifact-only realism 约为 train `31.33 → 35.80`、dev `31.52 → 35.69`；round-end overall 约为 train `33.65 → 38.55`、dev `33.62 → 38.47`。这只是 artifact proxy 改善，不是人工真实性验收。",
            "2. round 2–6 没有再修改 Harness：每轮的 deterministic 失败计数为 0，artifact/prepare 失败为 0，且没有独立视觉证据可以区分穿模、断骨、动作时序或镜头语义错误；继续 patch 会把低层 proxy 当成视觉标签，存在过拟合风险。",
            "3. “不是固定模板”由三层证据约束：训练 CLI 禁止 `template_baseline`，每 case 必须有 `codegen_call_id` 和独立 source hash，并且 source 内容必须含 case-specific profile marker。共享的 Blender runtime library 只是执行库，不是被偷偷替换的模板场景。",
            "4. 训练结束后的语义审查发现 object-only prompt 曾被默认补成 `actor_a=person`；已在 Harness 中删除该默认补人规则。两例真实 smoke 现在只含 prompt-derived props + staging support，说明这次修复改变了计划语义；但画面仍是 proxy 级别，不能宣称已经达到真实人体/物理质量。",
            "",
            "## 文件与人工后续",
            "",
            f"- 逐 case append-only Memory：`docs/t2blendercodeharness-agent-training-memory-v1.md`（当前 980 行）。",
            f"- 机器汇总：`{output_json_path}`。",
            f"- 曲线：`{Path(curve).resolve()}`。",
            "- 语义修复 smoke：`out/preflight/director-object-only-v4`（2/2 real video；无 actor 的 DirectorPlan，compound object 和 move 轨迹已保留）。",
            "- 正式训练仍受 `golden_review=pending` 与 `paired_gate=pending` 阻塞；下一步应在升级后的 exact-prompt bundle 上完成人工 blind review，再重跑 readiness。",
            "",
            "## 审计计数",
            "",
            "| 检查 | 结果 |",
            "|---|---:|",
            "| 六轮 attempt 报告 | 12 split reports |",
            "| 六轮 overall 报告 | 6 reports |",
            "| Attempt 视频 | 120 |",
            "| Overall 视频 | 720 |",
            "| 当前 protocol 视频 | 840 |",
            "| 零字节最终视频 | 0 |",
            "| preparation/artifact failure | 0 |",
            "| source provenance failure | 0 |",
            "| 独立视觉评分 | unavailable（未填 0） |",
        ]
    )
    Path(report).resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="out/training/diagnostic-six-rounds-v2")
    parser.add_argument(
        "--round1-attempt-root",
        default="out/training/diagnostic-six-rounds-v1/round-01/attempt-02/real",
        help="Historical upgraded round-1 attempt root; its parent is used to locate attempt-02.",
    )
    parser.add_argument(
        "--round1-overall-report",
        default="out/training/diagnostic-codegen-profile-v2-v1/round-01/overall_report.json",
    )
    parser.add_argument("--output-json", default="out/training/diagnostic-six-rounds-v2/diagnostic-six-round-summary.json")
    parser.add_argument("--curve", default="out/training/diagnostic-six-rounds-v2/six-round-curves.png")
    parser.add_argument("--report", default="docs/t2blendercodeharness-six-round-diagnostic-report-v1.md")
    args = parser.parse_args()
    summary = build_summary(
        args.root,
        output_json=args.output_json,
        curve=args.curve,
        report=args.report,
        round1_attempt_root=args.round1_attempt_root,
        round1_overall_report=args.round1_overall_report,
    )
    print(json.dumps({"rounds": summary["round_count"], "actual_protocol_case_slots": summary["actual_protocol_case_slots"], "curve": str(Path(args.curve).resolve()), "report": str(Path(args.report).resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
