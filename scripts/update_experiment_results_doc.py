"""Append an idempotent, human-readable results section to training memory."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "t2blendercodeharness-six-round-training-memory.md"
MARKER = "<!-- REAL_TRAINING_CURVES_AND_COMPLEX_REALISM -->"


def summarize(round_number: int) -> dict:
    report = json.loads((ROOT / "out" / "training" / "six-rounds-real-v6" / f"round-{round_number:02d}" / "overall_report.json").read_text(encoding="utf-8"))
    cases = report["cases"]
    rows = {}
    for split in ("train", "dev"):
        split_rows = [row for row in cases if row.get("split") == split]
        scored = [row["video_score"] for row in split_rows if row.get("video_score") is not None]
        rows[split] = {
            "det": sum(row["deterministic_score"] for row in split_rows) / len(split_rows),
            "pass_rate": sum(row["deterministic_status"] == "pass" for row in split_rows) / len(split_rows) * 100,
            "visual": sum(scored) / len(scored) if scored else None,
        }
    return {
        "round": round_number,
        "train": rows["train"],
        "dev": rows["dev"],
        "video_count": sum(row.get("video_exists") is True for row in cases),
        "scored_count": sum(row.get("video_score") is not None for row in cases),
        "det_fail_count": sum(row["deterministic_status"] != "pass" for row in cases),
    }


def main() -> int:
    summaries = [summarize(1), summarize(2)]
    section = f'''\n\n{MARKER}\n## 已完成轮次的结果曲线与复杂真实性探针\n\n本节由 `scripts/plot_training_curves.py` 根据真实 Blender 产物和 `overall_report.json` 生成；只记录已经完成的 round-01、round-02，不把未跑的 round-03 至 round-06 填成零分。\n\n![T2Blendercodeharness 训练曲线](assets/t2blendercodeharness-training-curves.png)\n\n| 轮数 | train deterministic 均值 | dev deterministic 均值 | train 通过率 | dev 通过率 | 真实视频 | 已视觉评分 | deterministic 失败 |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n| 1 | {summaries[0]['train']['det']:.4f} | {summaries[0]['dev']['det']:.4f} | {summaries[0]['train']['pass_rate']:.1f}% | {summaries[0]['dev']['pass_rate']:.1f}% | {summaries[0]['video_count']}/120 | {summaries[0]['scored_count']} | {summaries[0]['det_fail_count']} |\n| 2 | {summaries[1]['train']['det']:.4f} | {summaries[1]['dev']['det']:.4f} | {summaries[1]['train']['pass_rate']:.1f}% | {summaries[1]['dev']['pass_rate']:.1f}% | {summaries[1]['video_count']}/120 | {summaries[1]['scored_count']} | {summaries[1]['det_fail_count']} |\n\nround-02 相对 round-01 的 deterministic 均值变化为 train **{summaries[1]['train']['det'] - summaries[0]['train']['det']:+.4f}**、dev **{summaries[1]['dev']['det'] - summaries[0]['dev']['det']:+.4f}**。视觉评分仍采用真实视频帧的 assistant-local review；外部 `gpt-5.6-luna` / `gpt-5.6-terra` 当前 endpoint 返回 403/1010，因此没有伪造外部 VLM 分数。\n\n## 复杂故事与真实几何探针（未修改 Harness）\n\n- 数据集：`dataset/complex-realism-v1`，8 个互不相同的复杂 prompt，覆盖多角色交接、双物体并行动作、遮挡 reveal、反向 zigzag、交叉调度和多阶段摄像机轨迹。\n- 运行策略：固定当前 Harness 版本 `h-t2-hard-v4-camera-closeup-scene-parser`，使用真实 `D:/blender/blender.exe`，目标 512×512；不修改 scene parser、trajectory planner、camera planner 或 Blender 生成器。\n- evaluator 新增 `geometry-realism-v1`：对每个实际 `proxy.blend` 运行 Blender 内 mesh 审计，检查 required entity coverage、entity kind、顶点数、面数和 primitive hint。`detail_required=true` 时，球、UV 球、圆柱、立方体 stand-in、缺失实体或 topology 不足均为 hard failure，禁止进入 VLM 评分。\n- 这项硬门是为了避免“计划正确、视频可播放、但人物/物体只是粗粒度球柱体”被误判为成功；白膜材质本身不扣分，扣分证据来自真实 blend geometry。\n\n当前复杂探针的真实视频和几何报告将在本节下方逐案追加；若 deterministic/geometry gate 失败，只记录失败原因与视频地址，不补填视觉分数。\n'''
    existing = DOC.read_text(encoding="utf-8") if DOC.exists() else "# T2Blendercodeharness 训练记忆表\n"
    if MARKER in existing:
        existing = existing.split(MARKER, 1)[0].rstrip() + section
    else:
        existing = existing.rstrip() + section
    DOC.write_text(existing, encoding="utf-8")
    print(json.dumps({"document": str(DOC), "rounds": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
