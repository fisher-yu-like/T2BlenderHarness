# Golden Review 人工标注交接说明

这份文档是当前两份方案的人工门禁交接。它不代表训练已经获准；只有人工评审、真实 Director/BlenderCodeAgent provider、agent smoke 和 paired gate 都通过后，`training_allowed` 才能变成 `true`。

## 当前生成物

评审包位于：

```text
dataset/golden-review-v1/
```

当前包是由已有的三臂真实 Blender MP4 生成的匿名比较包：

- 30 个 VBench-derived case；每个 case 有 `sample_a`、`sample_b`、`sample_c` 三个真实 MP4，共 90 个视频。
- 每个视频带 3 个以上事件对齐帧和一张联系表。
- `manifest.jsonl` 对外只给原始 VBench `source_prompt`；不提供 arm、branch、commit、Harness 版本或目录映射。
- `blind_manifest.json` 只供评估脚本使用，标注者不要打开。
- 当前 `metadata.json` 的 `render_prompt_mismatch_count=30`，表示这批历史视频按扩展执行 prompt 渲染；因此 bundle 明确是 comparison-only，不能直接作为 raw-prompt calibration 或训练证据。
- 评分完成后应有 `90 × 2 = 180` 条人工记录，每条记录包含 14 个维度。

数据集本身的训练入口不是这个评审包，而是：

```text
dataset/vbench2-agent-training-index-v1/
```

它包含 60 train、60 dev、20 frozen test；每条 `prompt` 必须和 VBench-2.0 原始 `prompt_en` 完全一致。历史的 `dataset/vbench-derived-100-v1` 只用于比较/校准材料，不能作为六轮训练输入。

## 启动标注页面

在当前 worktree 根目录执行：

```powershell
uv run python scripts/golden_review_app.py --bundle dataset/golden-review-v1 --host 127.0.0.1 --port 8766
```

打开 `http://127.0.0.1:8766/`。页面会同时显示：

1. 原始英文 benchmark prompt，这是评分依据；
2. 中文辅助翻译，不能替代原始 prompt；
3. 三个匿名视频及事件对齐帧；
4. 14 个评分维度、证据、弱点、失败归因和置信度。

页面的“保存并下一个”会按样本推进；切换 case 或评分者时，如果有未保存修改会弹出保护提示。每位评分者必须使用不同的 ID，例如：

```text
reviewer-01
reviewer-02
```

两位评分者必须独立观看和评分，不能先看对方的 `human_scores.jsonl`，不能根据 sample 标签猜测实验臂。

## 评分标准

每个维度用 0–100 分：

- 0–10：要求缺失或出现明显错误；
- 25–45：只实现了部分要求，关键证据不足；
- 50–65：基本完成但有明显问题；
- 70–85：较好，只有局部问题；
- 90–100：几乎无明显问题且视频证据充分。

重点维度是镜头覆盖/调度创新、人物轨迹、物体轨迹、事件时序和时间连续性；真实性维度单独记录，不和任务分数混成一个无法解释的数字。视频里无法确认的事件必须降低相应分数，并在证据或弱点中写“无法从视频确认”，不能根据 plan 猜测。

## 完成后的自动验收

两位评分者都完成 90 个视频后，在项目根目录执行：

```powershell
uv run python scripts/finalize_golden_review.py --root dataset/golden-review-v1
uv run python scripts/validate_golden_review_set.py --root dataset/golden-review-v1
```

`finalize_golden_review.py` 会：

- 检查每个 case/sample 是否有两位不同评分者；
- 检查所有 14 个分数均为有限的 0–100 数值；
- 按固定排序选取两名评分者，计算每个维度的 ICC(2,1)；
- 原子更新 `metadata.json`，写入完成状态、评分者集合和一致性结果；
- 再运行完整 bundle validator，失败则恢复原 metadata。

在两人标注完成前，validator 的失败是预期的 `awaiting_human_annotations` 状态；本包另外会因 `comparison_only` 保持不通过。不得用历史分数、frame statistics、模板分数或 VLM unavailable 代替人工记录。真正校准包需要 active benchmark prompt 的 agent 重新渲染后再走同一页面和 finalizer。

## 当前仍需人工/外部完成的事项

人工需要完成：

1. 一名评分者完成 90 个视频的 14 维评分；
2. 另一名独立评分者用不同 ID 重复完成 90 个视频；
3. 运行 finalizer 和 validator，把输出报告交回。

如果只有一名人工评分者，页面仍可保存进度，但 golden gate 不会通过。第二名评分者最好在不知道第一人结果的独立浏览器会话中完成。

外部/环境门禁仍需真实证据：

- 外部 structured provider 成功返回一例 Director plan，本地 `CodexExecProvider`
  成功返回一例 Blender code；
- 由 agent 生成而非固定模板的 Blender smoke，且产出完整真实 MP4 artifact；
- 20-case agent/template paired gate；
- 如使用 VLM，必须是可审计的 `gpt-5.6-luna` 或 `gpt-5.6-terra`；不可用时只能使用有证据的人工视觉审查，不能伪造数字。

这些门禁没有全部通过前，六轮训练入口会 fail-closed，不创建训练轮次或渲染结果。
