# Exact-prompt 盲评包交接说明

更新时间：2026-08-29

## 当前状态

人工校准素材已经准备完成，但校准门禁仍然是 pending。素材来自
VBench-2.0 原始 prompt，未使用自建 prompt，也没有把冻结 test 集放入校准包。

| 项目 | 结果 |
|---|---|
| 校准 case | 30（train 16，dev 14） |
| 盲评视频 | 90 个真实 MP4 |
| 盲评臂 | `codex-local` Harness、显式 `template_baseline`、`direct_code` 消融 |
| prompt 一致性 | 30/30 case 的三条 run 均通过 `prompt_hash` 核对，mismatch=0 |
| 视觉维度 | 14 个，每个 0–100 |
| 当前状态 | `awaiting_human_annotations` |

素材包在 `dataset/golden-review-exact-v2`。页面由
`scripts/golden_review_app.py` 提供，地址为：

`http://127.0.0.1:8765/`

## 需要人工做什么

需要两名独立评审者分别完成 90 个盲评视频，每条视频填写全部 14 个维度、
可见证据、问题/弱点、总体判断和置信度。正式门禁需要 180 行评分记录；
同一个人不能用两个 ID 冒充独立评审者。

操作顺序：

1. 打开页面，在右上角填写自己的评审 ID，例如 `reviewer-01`。
2. 每个 case 依次完整观看 `sample_a`、`sample_b`、`sample_c`，再逐个选择样本评分。
3. 只依据视频中可见的内容评分，不查看 plan、目录名、隐藏映射或 Harness 版本。
4. 轨迹和摄像机重点看：人物/物体是否按事件顺序移动，交接或接触是否连续，镜头是否覆盖关键事件，是否出现遮挡、跳变和身份混淆。
5. 真实度重点看：实体外观细节、材质/光照/接触可信度、空间尺度稳定性、运动自然度和整体呈现质量。
6. 每次点击“保存并下一个”，不要在没有证据时填写 100；如果视频无法判断，降低对应维度并在证据栏写明原因。

评分只用于校准 evaluator 和比较三条盲评臂，不能直接决定 Harness patch。
`template_baseline` 仅是显式对照，不是生产失败时的 fallback。

## 完成后的自动检查

人工提交后，运行：

```powershell
uv run python scripts/validate_golden_review_set.py --root dataset/golden-review-exact-v2
uv run python scripts/finalize_golden_review.py --root dataset/golden-review-exact-v2
```

只有当每个 blind video 都有两名评审者、14 个维度齐全、ICC(2,1) 可计算且
prompt mismatch 仍为 0 时，校准包才会通过。随后再运行 evaluator calibration
和 readiness 检查；在 `training_allowed=true` 以前，不启动六轮 Harness 训练。

## 外循环安全约束

每轮最多五次外循环 attempt。`run_bounded_outer_attempts` 只接受
`patch`、`accept`、`stop` 三种转换；没有明确的单组件 Harness patch 时，
第一轮证据完成后进入 `awaiting_harness_patch`，不会自动重复渲染同一版本。
单个 case 的 plan/code/render 失败最多重新生成三次，耗尽后记录
`NOT_RENDERED`，不换用模板视频或合成分数。
