---
name: t2blendercodeharness-zh
description: 使用中文运行 T2Blendercodeharness，将复杂 prompt 转换为 Blender proxy 场景、轨迹和相机计划，执行 MCP/CLI、artifact gate、deterministic/oracle/VLM 评测，并依据 train/dev 失败以单组件方式进化 Harness。
---

# T2Blendercodeharness 中文版

这是一个 contract-first Text-to-Blender Harness。保持 Codex Host、MetaHarnessOptimizer、DesignHarness、Dataset、Evaluator 的边界清晰。

核心原则：计划不是渲染，telemetry 不是完整 artifact，deterministic pass 不是 VLM 分数，proposal 不是 accepted patch。

## 1. 标准流程

```text
Prompt
 → SceneContract
 → TrajectoryPlan + CameraPlan
 → Blender MCP / CLI proxy job
 → .blend + PNG + telemetry + proxy.mp4
 → artifact gate
 → deterministic evaluator
 → independent oracle
 → optional VLM
 → train failure aggregation
 → one-owner code patch
 → train/dev acceptance
 → frozen test
 → append-only Memory
```

项目入口：

- Parser：`src/videoact/scene_contract.py`
- Trajectory：`src/videoact/trajectory.py`
- Camera：`src/videoact/camera.py`
- Executor：`src/videoact/blender_adapter.py`
- Artifact gate：`src/videoact/real_artifacts.py`
- Deterministic evaluator：`evaluator/deterministic.py`
- Independent oracle：`evaluator/independent_oracle.py`
- Meta optimizer：`src/videoact/meta_harness.py`
- Memory：`training/harness_memory.py`

## 2. Prompt 解析和规划

使用 `SceneContractBuilder`，禁止绕过 contract 直接写临时 Blender 代码。

Contract 至少描述：

- entities：character、support、target、drop zone、occluder、assistant；
- events：walk、reach、grasp、lift、carry、place、release、reveal；
- relations：例如 target 在 support 上；
- camera constraints：follow、orbit、dolly、hold、close-up、reveal；
- physics constraints：support-before-grasp、attachment lifecycle、no-penetration。

复杂动作通常应表达为：

```text
walk → reach → grasp → lift → carry → place → release
```

必须拒绝空 prompt、非法时间、未知引用、重复 ID、越界 event 和矛盾顺序。

`TrajectoryPlanner` 生成 frame-indexed states、motion primitives、velocity、attachment 和 observability。必须检查 frame 从 1 开始、state 单调、primitive 不越界、grasp 前有 support、attachment 有接触、速度没有异常跳变。

`CameraPlanner` 生成 establishing/follow/orbit/dolly/close-up/reveal shots。每个 `must_show` event 至少被一个 shot 覆盖。

## 3. Blender 执行和 artifact gate

优先执行 Blender MCP，失败后才按策略 fallback 到 CLI。必须保存 prompt/plan/Harness/evaluator fingerprint、MCP response、执行状态、stdout/stderr 和 immutable job source。

真实 run 必须包含：

```text
run_manifest.json
scene_contract.json
trajectory.json
camera_plan.json
blender_job.py
proxy.blend
proxy.mp4
telemetry.json
frames/index.json
至少 3 张可读 sampled PNG
```

只有 telemetry 没有完整 artifact 时必须 hard fail。

## 4. Evaluator 标准

Deterministic evaluator 检查：

- event order 和 required event coverage；
- camera coverage；
- support-before-grasp；
- attachment contact；
- velocity continuity；
- artifact completeness；
- telemetry entity/kind/timebase/active camera；
- sampled frame 可读性。

Finding 必须有 `failure_id`、`owner`、`severity`、`evidence`、`repair_route`。

普通分数：

```text
score = max(0, 100 - 25 × finding_count)
```

Hard dataset 额外用 independent oracle 检查 gold event order、camera types、constraints、motion primitives、attachment actions、proxy entities：

```text
final_score = deterministic_score - 15 × oracle_finding_count
```

该分数不是 VLM 视频质量分数。只有 artifact complete 且 deterministic pass 的样本才能进入 VLM；VLM 网络/权限/schema 失败记录为 `unavailable`，不能转成 0 分。

另外统计 `plan_hash` 的 unique count 和 collision rate；不同 prompt/scene 生成同一个 plan 是泛化风险。

## 5. 错误聚合和 Harness 调整

train 失败按以下 key 聚合：

```text
failure_id + owner + category + severity
```

同一 failure 至少影响两个不同 train case，才生成 actionable proposal。

owner 约定：

- `scene_parser`：entity/event/constraint/同义词/关系；
- `trajectory_planner`：states/primitives/attachment/velocity；
- `camera_planner`：shots、follow/orbit/dolly、coverage；
- `blender_executor`：MCP/CLI、执行和运行时错误；
- `proxy_renderer`：blend/PNG/MP4/telemetry；
- `evaluator`：规则、分数和报告 schema。

外循环必须是：

```text
train records → failure aggregation → one-owner proposal
→ 修改一个组件 → rerun train → rerun dev
→ acceptance/rejection/rollback → acceptance 后 frozen test
```

Acceptance gate：

```text
train_after > train_before
dev_after >= dev_before
hard_dev_regression == false
```

禁止同时修改多个 owner，禁止使用 test 选择 patch，禁止把 baseline 视频直接当 candidate 视频。当前“训练”是 code-level Harness evolution，不是神经网络权重训练。

## 6. Harness Memory 和 Skill 自进化

每轮都写入 append-only Memory：

```text
proposal → patch_applied → train_evaluated → dev_evaluated
→ accepted/rejected/rollback/no_patch → test_evaluated
```

Memory 保存 parent/candidate version、owner、dataset/evaluator fingerprint、affected cases、score before/after、evidence 和决策原因。

Skill 自身不能运行时偷偷改写。使用 `propose_skill_update.py` 先生成 proposal，再人工审核、capability check、全量测试和 forward-test。

## 7. 当前项目命令

```text
scripts/build_hard_trajectory_dataset.py
scripts/validate_hard_trajectory_dataset.py
scripts/prepare_real_jobs.py
scripts/run_benchmark.py
scripts/run_hard_end_to_end_training.py
scripts/evaluate_real_runs.py
scripts/evaluate_real_videos.py
scripts/build_dataset_prompt_proxy_score_index.py
```

当前训练结果：`h-t2-hard-v0 → h-t2-hard-v1`，owner 为 `scene_parser`，train `57.25 → 61.50`，dev `30.25 → 50.50`，test `22.75 → 40.00`，decision 为 accepted。

Memory：`out/training/t2blendercodeharness-hard-final-v2/memory/harness_updates.jsonl`。

## 8. 停止条件

Contract/plan 失败、MCP/CLI failed 或 timeout、artifact 缺失、deterministic hard fail、VLM unavailable、train 无重复 failure、proposal 多 owner、dev regression、test 泄漏、只有 telemetry 没有视频时，都必须停止晋级并报告。
