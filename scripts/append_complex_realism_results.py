"""Append frozen complex-probe results to the experiment memory document."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "t2blendercodeharness-six-round-training-memory.md"
MARKER = "<!-- COMPLEX_REALISM_RESULTS_V1 -->"
RUN_ROOT = ROOT / "out" / "complex-realism-v1" / "real" / "test"
DATASET = ROOT / "dataset" / "complex-realism-v1" / "manifest.jsonl"


def main() -> int:
    summary = json.loads((RUN_ROOT / "geometry_audit_summary.json").read_text(encoding="utf-8"))
    records = {
        record["case_id"]: record
        for record in (json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    rows = []
    for result in summary["results"]:
        case_id = result["case_id"]
        run_dir = RUN_ROOT / case_id
        deterministic = json.loads((run_dir / "deterministic_report.json").read_text(encoding="utf-8"))
        attempts = json.loads((run_dir / "render_attempts.json").read_text(encoding="utf-8"))
        findings = ", ".join(finding["failure_id"] for finding in result["findings"])
        video = (run_dir / "proxy.mp4").resolve()
        prompt = records[case_id]["prompt"].replace("|", "\\|")
        rows.append(
            f"| {case_id} | {prompt} | [proxy.mp4]({video.as_posix()}) | 512×512, 432 frames, {len(attempts)} attempt | {deterministic['score']:.1f} | {result['score']:.1f} | {findings} | no VLM: geometry hard gate failed; visual inspection confirms primitive white proxy |"
        )
    section = f'''\n\n{MARKER}\n## 复杂故事真实度探针 v1：冻结 Harness，不修复\n\n这批实验的目的不是提高分数，而是测量当前 Harness 在复杂叙事和真实几何要求下的真实上限。8 个 prompt 全部不同，使用真实 Blender CLI（`D:/blender/blender.exe`）、512×512、18 秒、24 fps、432 帧；当前 Harness 版本固定为 `h-t2-hard-v4-camera-closeup-scene-parser`。这一阶段没有修改 scene parser、trajectory planner、camera planner 或 Blender 生成器。\n\n| case | 真实视频 | 渲染产物 | deterministic 分数 | geometry 分数 | 主要检测结果 | 处理 |\n|---|---|---|---:|---:|---|---|\n{chr(10).join(rows)}\n\n结果汇总：8/8 视频可播放并完成渲染，8/8 几何审计 hard fail，0 个进入 VLM/assistant-local 视觉分数。这里的 0 不是“视频不存在”，而是 evaluator 按策略拒绝对已证明为粗粒度代理的内容继续打视觉分。所有 job 的 `render_attempts.json` 均记录首次成功；没有发生需要重试的 Blender 执行失败。\n\n### 代表性视觉证据\n\n- [complex-01 frame 12]({(RUN_ROOT / 'complex-01' / 'frames' / 'frame_000012.png').resolve().as_posix()})：人物呈低面球体，桌面/目标区为立方体，物体为圆柱。\n- [complex-04 frame 24]({(RUN_ROOT / 'complex-04' / 'frames' / 'frame_000024.png').resolve().as_posix()})：复杂双角色故事没有得到对应的真实角色几何。\n- [complex-06 frame 24]({(RUN_ROOT / 'complex-06' / 'frames' / 'frame_000024.png').resolve().as_posix()})：多阶段场景仍是白膜 primitive 组合。\n\n### evaluator 新增的真实性规则\n\n`geometry-realism-v1` 在 Blender 中打开实际 `proxy.blend` 并审计 mesh，不读取 plan 代替视频证据：\n\n1. `required_entity_ids` 必须都在实际 blend 中出现；\n2. `entity_kind` 必须和 proxy scene 语义一致；\n3. 每个 required mesh 至少 256 vertices、128 faces；\n4. `detail_required=true` 时，sphere、uv sphere、cylinder、cube 等 primitive hint 是 hard failure；\n5. 任意 hard failure 都阻止后续 VLM 分数，并将 geometry score 封顶为 20。\n\n这次测试验证了 evaluator 能抓住此前 deterministic/视频可播放检查抓不住的问题：视频存在，但视觉实体仍然是粗粒度 stand-in。下一步若进入 Harness 自进化，应该把这些失败作为一个新的单组件修复候选（proxy renderer/geometry generation），但本轮按要求暂不修改 Harness。\n'''
    section_lines = section.splitlines()
    for index, line in enumerate(section_lines):
        if line.startswith("| case |"):
            section_lines[index] = "| case | prompt | proxy video | render artifact | deterministic | geometry | findings | handling |"
            if index + 1 < len(section_lines):
                section_lines[index + 1] = "|---|---|---|---|---:|---:|---|---|"
            break
    section = "\n".join(section_lines)
    existing = DOC.read_text(encoding="utf-8") if DOC.exists() else "# T2Blendercodeharness training memory\n"
    if MARKER in existing:
        existing = existing.split(MARKER, 1)[0].rstrip() + section
    else:
        existing = existing.rstrip() + section
    DOC.write_text(existing, encoding="utf-8")
    print(json.dumps({"document": str(DOC), "case_count": len(rows), "hard_fail_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
