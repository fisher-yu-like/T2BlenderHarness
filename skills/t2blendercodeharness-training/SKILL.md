---
name: t2blendercodeharness-training
description: Use when training or evaluating T2Blendercodeharness with real Blender CLI proxy videos, the five-round single-character/single-prop phase, shared VLM or local Codex visual review, separate task and realism scores, one-component Harness evolution, and persistent Markdown memory.
---

# T2Blendercodeharness Training

这是 T2Blendercodeharness 的外循环训练入口。训练对象是 Harness 组件，不是神经网络权重；必须保留真实 `.blend`、`proxy.mp4`、PNG、telemetry、evaluator 报告和 append-only Memory。

## 当前 phase-1 协议

- 当前训练集使用 `dataset/trajectory-v3-single`：120 个 case，50 train、50 dev、20 frozen test，严格每个 prompt 一个 character 和一个 prop。
- 原始 `dataset/trajectory-v3-hard` 保留不动，用于后续多人物/多物体升级；不要把它混入 phase-1 统计。
- 五轮，每轮从一个 10-case train family 和一个 10-case dev family 取样；每次 attempt 10 train + 10 dev，最多 5 次。
- 每轮结束对累计 train/dev 做整体评测；当前 phase 每轮整体范围为 20、40、60、80、100 个 case。保守视频上限为 `5 × (5 × 20 + 100) = 1000` 个真实视频。
- Blender CLI 失败时每 case 最多重试 2 次；重试记录在 `render_attempts.json`，耗尽后写 `NOT_RENDERED`，不能补造视频或分数。
- 使用独立 case 目录和并行 worker；本机优先 12 workers 或更高，但不能让并行覆盖已有 artifact。

## Harness 流程

```text
Prompt → SceneContract → character/object trajectory + CameraPlan
       → Blender CLI → proxy.blend + PNG + proxy.mp4 + telemetry
       → artifact gate → deterministic + independent oracle
       → geometry/frame evidence → shared visual review（if/else）
       → task_score 与 realism_score 分开记录
       → train failure aggregation → one-owner Harness patch
```

允许的 Harness owner：`scene_parser`、`trajectory_planner`、`camera_planner`、`blender_executor`、`proxy_renderer`。Evaluator 和数据集在训练开始后冻结；只有明确的 evaluator 审计任务才可以修改 evaluator。

## Shared visual review

外部 VLM 和本地 Codex 不是两套评分器，而是同一个接口的两个来源：

```text
if external gpt-5.6-luna / gpt-5.6-terra is available:
    source = external_vlm
else:
    source = assistant_local_review
```

两种来源都必须使用相同的 prompt、contract、事件对齐帧、维度、JSON schema、confidence 和 evidence 字段。外部失败时由 Codex 读取真实帧并输出本地复核，不得把 deterministic 或低层 realism 分数冒充 VLM。

只有 artifact 完整、MP4 可解码、deterministic 无 hard failure、geometry 无 hard failure 时才进行一次 shared review；每个 case 至多一次，最多 8 个真实 PNG 帧。硬失败 case 不消耗视觉 review 调用。

## 两个公开分数

报告和 Markdown 只把下面两个分数作为主结果，不生成混合总分。

### `task_score`

表示 prompt 的事件、人物/物体轨迹和摄像机编排是否完成。review 内部检查 story fidelity、camera choreography、character/object trajectory、timing continuity；具体诊断字段可以保存在 JSON，但不必作为用户主表格的分数名。

```text
task_score = .20 × deterministic_score + .80 × task_review
```

`task_review` 使用真实帧的事件顺序、相机 follow/orbit/dolly、人物运动、物体抓取/携带/放置和时间连续性。这里的 `+` 很重要，不能写成减号。

### `realism_score`

表示人物、物体、动作、空间和画面是否具有可信的真实感。geometry 与 PNG 只提供低层证据，shared review 是主体：

```text
realism_score = .15 × geometry_score
              + .15 × frame_evidence_score
              + .70 × realism_review
```

`realism_review` 使用真实帧检查 appearance detail、physical realism、spatial consistency、motion naturalness、visual presentation。VLM/local review 必须给出每个维度的可见证据和 confidence；低于 0.6 的 review 标为 `needs_human_review`。

没有 shared review 时，只能输出低层 `artifact_only_proxy`，使用：

```text
artifact_raw = .60 × geometry_score + .40 × frame_evidence_score
artifact_only_proxy = min(80, .80 × artifact_raw)
```

它不能叫真实性，不能进入 `realism_pass`，也不能代替 `realism_score`。geometry 达到 100 只说明结构合同通过，不说明视频真实。

## Deterministic 与 artifact gate

Deterministic evaluator 使用实际 contract、trajectory、camera plan、telemetry 和独立 oracle，检查 event order、event coverage、camera intent、support/contact、attachment lifecycle、velocity continuity、timebase、实体类型和可见性意图。

finding 按 root cause 去重后扣分：`hard=30`、`error=18`、`warning=8`、`info=2`，基础分为 `max(0, 100 - penalty_sum)`。任何 hard failure 阻断 shared visual review；task 和 realism 都不能互相抵消硬失败。

Realism geometry audit 必须打开真实 `proxy.blend`，检查 required entities、entity kind、topology、primitive stand-in、geometry style、connected components。PNG audit 必须读取真实 `frames/index.json` 指定的图片，检查可读率、分辨率、前景覆盖、边缘结构和时间变化。

## 外循环自进化

每个 attempt 完成后只从 train 失败中聚合重复模式：`failure_id + owner + category + severity + root_cause_id`。MetaHarnessOptimizer 每次只产生一个 owner 的 patch；不能同时改 parser、planner、renderer、evaluator 或数据集。

候选必须在相同 phase 数据集和 evaluator 上重跑：

```text
train_after > train_before
paired_dev_after >= paired_dev_before
overall_dev_after >= overall_dev_before
无 hard dev regression
```

若失败，记录 rejected/rollback；若没有跨样本重复证据，记录 `no_patch`。test 只能在五轮完成后做一次 blind final evaluation。

## Memory 表格

每个 split 完成后立即更新 `docs/t2blendercodeharness-six-round-training-memory-v7.md`。每个 case 一行，必须包含：

| 轮数 | Prompt | Proxy 视频地址 | task_score | realism_score | review 来源/置信度 | Harness 问题 | 单组件修复及方法 | 修复前后变化 | 自然语言处理 |
|---:|---|---|---:|---:|---|---|---|---:|---|

还必须保存：

```text
round-XX/attempt-YY/real/{train,dev}/
round-XX/overall/real/{train,dev}/
round-XX/attempt_report.json
round-XX/overall_report.json
round-XX/patch_manifest.json
memory/harness_updates.jsonl
```

视频地址只能是真实 `proxy.mp4` 绝对路径；未渲染只能写 `NOT_RENDERED` 和失败原因。自然语言必须解释发现依据、重试、owner、接受/拒绝/rollback，以及为什么没有把 dev-only 提升当作泛化。

## 推荐命令

```powershell
& "$env:LOCAL_PYTHON" scripts\train_real_harness.py --mode protocol --dataset-root dataset\trajectory-v3-single --round-root out\training\single-five-rounds-v1
& "$env:LOCAL_PYTHON" scripts\train_real_harness.py --mode attempt --round 1 --attempt 1 --dataset-root dataset\trajectory-v3-single --round-root out\training\single-five-rounds-v1 --blender-bin D:\blender\blender.exe --workers 12 --vlm-model gpt-5.6-luna --markdown-path docs\t2blendercodeharness-six-round-training-memory-v7.md
& "$env:LOCAL_PYTHON" scripts\train_real_harness.py --mode overall --round 1 --dataset-root dataset\trajectory-v3-single --round-root out\training\single-five-rounds-v1 --blender-bin D:\blender\blender.exe --workers 12 --vlm-model gpt-5.6-luna --markdown-path docs\t2blendercodeharness-six-round-training-memory-v7.md
```

开始真实实验前必须完成全量 tests、skill capability check、protocol fingerprint 核验和至少一个真实 Blender smoke；实验过程中不伪造 VLM、视频、分数或 patch 结果。
