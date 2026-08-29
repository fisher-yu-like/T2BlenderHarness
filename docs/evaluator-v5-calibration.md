# Evaluator v5 校准与评分边界

状态：`pending_exact_prompt_rerender_and_human_golden_review`

本文档是当前 `visual-primary-v6-independent-channels` 实现的校准说明。
它记录公式和可验证边界，但不伪造人工相关系数；在双人 blind golden
review 完成前，任何“校准通过”结论都不能写入训练 acceptance。

## 1. 评估顺序

```text
真实 Blender artifact
  -> RealArtifactGate
  -> deterministic / Director / interaction / independent-oracle diagnostics
  -> eligible chronological frames
  -> one shared visual review
  -> separate task and realism channels
```

### Artifact gate

必须存在并可读：manifest、DirectorPlan、SceneContract、TrajectoryPlan、
CameraPlan、冻结的 `blender_job.py`、`proxy.blend`、可播放的 `proxy.mp4`、
`telemetry.json`、`frames/index.json` 和至少三张可读 PNG。Manifest 中的
`code_hash` 必须与 job source 一致；DirectorPlan hash 不一致时直接 hard
fail。Blender 进程失败最多重试同一份冻结源代码两次，不能生成替代模板。

### Deterministic channel

deterministic evaluator 是 eligibility gate 和诊断通道，不是视觉质量主分。
它检查时间基准、事件顺序、required event coverage、trajectory phase、
support-before-grasp、attachment/handoff、速度连续性、camera active、
telemetry identity/kind、相机可见性，以及真实运行产生的 runtime findings。

Independent oracle 再将生成的 contract/plan/telemetry 与数据集预先写入的
`oracle_expectations` 对照，检查：

- 事件顺序与必须存在的实体；
- required camera event、camera type/constraint；
- actor/prop 的轨迹与 attachment lifecycle；
- `no_prop_penetration`、`no_identity_swap`、
  `handoff_requires_same_window_detach_attach`；
- `all_required_targets_visible_in_event_shot`；
- `no_unplanned_actor_crossing`：不依赖 `character` 这个旧名字，而是按
  actor 稳定 ID 检查轨迹 lane-order 是否在非 handoff/非显式 crossing 窗口内反转。

缺少验证所需的 runtime evidence 时，oracle 产生
`oracle_negative_evidence_missing`，而不是默认通过。

## 2. 视觉审查维度

只有 artifact gate 通过后，才把原始 prompt、DirectorPlan 摘要、按事件中点
对齐的 chronological frames 和统一 schema 交给一次视觉审查。支持的模型
ID 是小写 `gpt-5.6-luna` 与 `gpt-5.6-terra`；也支持带证据的
`human_review`/`codex_local_visual_review`。transport、schema 或置信度低于
0.6 时记录 `unavailable`/`needs_human_review`，不转换为 0、100 或 plan-derived
score。

### Task channel

四个语义维度用几何平均（GM）汇总，以防单个关键维度为 0 时被其它高分掩盖：

```text
semantic = GM(prompt_compliance,
               physical_plausibility,
               object_trajectory,
               event_timing)

choreography = GM(camera_coverage,
                  camera_innovation,
                  character_trajectory,
                  temporal_smoothness)

task_score = 0.45 * semantic
           + 0.45 * choreography
           + 0.10 * visual_clarity

task_final_score = task_score
```

因此 camera choreography、character trajectory 和 object trajectory 是任务
主分的核心；`visual_clarity` 只占 0.10，避免清晰但不执行 prompt 的视频得高分。
`physical_plausibility` 与 `event_timing` 约束动作是否符合物理和顺序，不能
只凭“画面里出现了对象”给高分。

### Realism channel

真实性不复用 task 分数，也不与 task 相加：

```text
realism_vlm = GM(appearance_detail,
                  physical_realism,
                  spatial_consistency,
                  motion_naturalness,
                  visual_presentation)

reviewed_realism = 0.15 * geometry_score
                 + 0.15 * frame_evidence_score
                 + 0.70 * realism_vlm
```

VLM 负责真实性的主要判断，因为材质细节、自然动作、空间一致性和视觉呈现
不能由文件结构或 plan 可靠推出。geometry/frame evidence 只作为独立的
可审计辅助，不能把“有 mesh”冒充成“看起来真实”。

`artifact_only_proxy` 只说明 artifact 和可观测结构满足部分要求，并带
`realism_claim=not_established`；它永远不填充 `realism_vlm`。

## 3. Finding 严重度

严重度是 gate/诊断分类，不是把视频质量粗暴线性扣分的人工常数：

| Severity | 语义 | 处理 |
|---|---|---|
| `hard` | 无法证明 case 正确，或违反不可接受约束 | 阻断视觉评分和 patch acceptance |
| `error` | 可执行证据显示明显局部错误 | 保留 finding，通常拒绝有回归的候选 |
| `warning` | 质量风险或较弱证据 | 进入聚合和人工审查，不自动宣称通过 |
| `info` | 可追踪观察 | 记录，不单独触发拒绝 |

Deterministic 的 score 是稳定的 finding 摘要，便于诊断历史；它不再通过
`0.60 * deterministic + 0.40 * VLM` 压低视觉主分。真实视觉质量的主结果是
`task_final_score` 与独立的 `reviewed_realism`，而不是一个混合黑箱数字。

## 4. Golden review 校准

校准集必须包含 30–50 个来自不同场景和事件组合的 blind cases，每个 case
至少三个匿名 sample，每个 sample 至少两名独立标注者，覆盖全部 14 个维度：

```text
prompt_compliance, physical_plausibility, camera_coverage,
camera_innovation, character_trajectory, object_trajectory, event_timing,
temporal_smoothness, visual_clarity, appearance_detail, physical_realism,
spatial_consistency, motion_naturalness, visual_presentation
```

标注者看真实视频/帧，不看 arm、branch、commit、Harness version 或候选分数。
`patch_selection_allowed=false`。校准脚本报告每个维度的 Spearman/Pearson、
bootstrap CI 和 inter-rater agreement；若 task/realism 相关性不足，优先
修订维度定义、证据提示和标注协议，不在训练过程中反复调权重刷分。

当前仓库尚未有完整双人标注 bundle，因此 Gate 0 仍为 pending。运行：

```powershell
uv run python scripts/validate_golden_review_set.py --root dataset/golden-review-exact-v2
uv run python scripts/calibrate_evaluator.py --records <golden-records.jsonl> --out <calibration-report.json>
```

验证器失败、bundle 不存在或标注不足，都必须原样记录，不能用 baseline、
artifact-only 或 VLM unavailable 结果替代。

## 5. 2026-08-28 合并审计状态

本次两方案融合审计已经把评估器的可执行边界和训练准入边界分开：

- `out/two-plan-convergence/full-test.xml` 记录全量 `395 passed, 28 warnings`；
- active dataset 已切换为 `dataset/vbench2-agent-training-index-v1`，验证为
  60 train / 60 dev / 20 frozen test、140 条 VBench-2.0 原始 prompt；每条
  `prompt` 与 raw `prompt_en` 完全一致，不含本地编写的事件/实体/oracle 标签。
- `dataset/frozen-eval-v1` 已对 VBench-derived、trajectory-v4 和 active training
  三个参考集完成 case/prompt/source/semantic 四类零泄漏；
- 旧 template baseline 的真实 MP4/artifact smoke 只作为执行层证据，不能作为
  agent 视觉或训练准入证据；
- `out/training_readiness_report.json` 的 `training_allowed` 当前为 `false`，原因是
  agent smoke、golden review、dynamic provider pair 和 paired gate 尚未全部通过。

因此 evaluator 代码和自动化边界已收口，但“与人工判断一致的校准完成”仍是
`pending_exact_prompt_rerender_and_human_golden_review`，不会用 baseline 或 unavailable review 代替。
