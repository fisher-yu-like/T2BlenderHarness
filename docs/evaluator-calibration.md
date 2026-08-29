# Evaluator Calibration Protocol

## Current actionable handoff (2026-08-28)

The active blind-review bundle is `dataset/golden-review-exact-v2`. It contains 30
cases and 90 real MP4 files copied from the historical three-arm comparison
artifacts. The public manifest shows the verbatim VBench `source_prompt` in
`prompt` and `prompt_en`, plus a Chinese helper translation in the UI; it does
not expose the hidden arm mapping.

This bundle is currently `awaiting_human_annotations`: 180 rows are required
(90 videos × two independent annotators). Run the local UI with:

```powershell
uv run python scripts/golden_review_app.py --bundle dataset/golden-review-exact-v2 --host 127.0.0.1 --port 8765
```

After two reviewers finish, run `scripts/finalize_golden_review.py`; it computes
ICC(2,1) for all 14 dimensions, atomically updates `metadata.json`, and invokes
the strict bundle validator. Until that sequence passes, no visual score can
unlock Harness patch selection. The video artifacts are comparison-only
historical renders; the active training input remains the exact-prompt
`dataset/vbench2-agent-training-index-v1`.

> **Historical snapshot.** The active evaluator contract is documented in
> [evaluator-v5-calibration.md](evaluator-v5-calibration.md) and
> [harness-architecture-v2.md](harness-architecture-v2.md). Do not use any
> legacy deterministic/VLM fusion formula in this file for active training.

## 当前黄金集

黄金集由真实 Blender 渲染的 MP4 和事件对齐帧组成，位置为：

`dataset/golden-review-exact-v2/`

- `videos/<case_id>/sample_a.mp4`、`sample_b.mp4`、`sample_c.mp4`：三路匿名视频；
- `sheets/<case_id>.png`：同一 case 的八帧联系表，帧按时间从 `f1` 到 `f8`；
- `manifest.jsonl`：prompt 与匿名样本路径；
- `blind_manifest.json`：只供 evaluator 使用，包含真实实验臂映射，人工评审者不得打开；
- `human_scores.jsonl`：人工评分输入文件。

## 交互界面

在项目工作区启动本地评分页面：

```powershell
uv run python scripts/golden_review_app.py --bundle dataset/golden-review-exact-v2 --host 127.0.0.1 --port 8765
```

然后打开 `http://127.0.0.1:8765/`。页面会显示三个真实 MP4、当前 case 的 prompt、14 个维度说明和保存进度；点击“保存当前视频评分”后，记录会原子写入 `human_scores.jsonl`，相同 `case_id + sample_id + annotator_id` 再保存会覆盖旧记录而不会重复累加。

每条人工记录必须唯一标识 `case_id + sample_id + annotator_id`。同一个盲法视频需要至少两位独立人工评审者，不能把 `sample_a/b/c` 合并为一个 case 分数。当前完整黄金包包含 30 个 case、90 个视频，因此完整填写需要 180 条评分记录；“至少 10 个 case”是校准的最低统计门槛，不是当前完整包校验的替代数量。

## 评分规则

只看 prompt、对应 MP4 和匿名帧，不看 plan、telemetry、目录映射或其他样本分数。对下面 14 个维度分别给 0–100 的整数或小数：

| 维度 | 要回答的问题 |
|---|---|
| `prompt_compliance` | prompt 要求的实体、动作和事件是否真的出现？ |
| `physical_plausibility` | 是否悬空、穿模、失去支撑或违反基本物理？ |
| `camera_coverage` | 关键人物、物体和交互是否持续可见？ |
| `camera_innovation` | 镜头是否执行了要求的 follow/orbit/dolly/reveal 等调度？ |
| `character_trajectory` | 人物路径、姿态和交互是否清楚、连续？ |
| `object_trajectory` | 物体移动、交接、放置和最终归属是否清楚？ |
| `event_timing` | 事件顺序、预备动作、停顿和交接时机是否正确？ |
| `temporal_smoothness` | 帧间是否跳变、抖动或突然改变位置？ |
| `visual_clarity` | 身份、动作和空间关系是否容易辨认？ |
| `appearance_detail` | 模型、材质和局部细节是否仍是粗粒度代理？ |
| `physical_realism` | 接触、阴影、光照、比例和重力是否可信？ |
| `spatial_consistency` | 身份、尺度、位置和场景布局是否稳定？ |
| `motion_naturalness` | 人物肢体、手部和物体运动是否自然？ |
| `visual_presentation` | 构图、曝光、清晰度和整体呈现是否良好？ |

推荐锚点：`0–20` 缺失/明显错误，`25–45` 只有部分证据，`50–65` 基本完成但问题明显，`70–85` 较好，`90–100` 几乎无明显问题且证据充分。帧中无法确认的事件不要根据 plan 猜测，降低分数并在 `weaknesses` 中写“evidence insufficient”；不要因为是代理渲染就自动给 0，也不要轻易给 100。

可选但建议记录：

- `pass_fail`: `pass`、`borderline` 或 `fail`；
- `primary_failure_owner`: `director_prompt_interpreter`、`director_event_scheduler`、`director_trajectory`、`director_camera`、`blender_code_agent`、`blender_executor`、`proxy_renderer`、`evaluator` 或 `none`；
- `weaknesses`: 只写可从视频观察到的具体问题；
- `confidence`: 0–1。

示例（数值仅示范格式，不要照抄）：

```json
{"case_id":"vbench2-camera_motion-001","sample_id":"sample_a","annotator_id":"annotator-sy","scores":{"prompt_compliance":70,"physical_plausibility":65,"camera_coverage":55,"camera_innovation":60,"character_trajectory":65,"object_trajectory":50,"event_timing":45,"temporal_smoothness":70,"visual_clarity":60,"appearance_detail":40,"physical_realism":45,"spatial_consistency":65,"motion_naturalness":50,"visual_presentation":55},"pass_fail":"borderline","primary_failure_owner":"director_camera","weaknesses":["handoff is not visible in the sampled frames"],"confidence":0.75}
```

完成后先运行：

```powershell
uv run python scripts/validate_golden_review_set.py --root dataset/golden-review-exact-v2
```

校准脚本再把人工维度与 VLM、frame-statistics、deterministic 的同视频记录对齐，计算维度相关性、任务分数和 realism 分数。人工评分不直接修改阈值，也不能访问 frozen test split；校准未通过前不允许据此选择 Harness patch。
