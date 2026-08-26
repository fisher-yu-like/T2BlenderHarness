# T2Blendercodeharness Harness Training Process Log

> This file preserves the early experiment. The active protocol is [the six-round real-video protocol](t2blendercodeharness-six-round-protocol.md) and [the training Skill](../skills/t2blendercodeharness-training/SKILL.md): 60 train, 60 dev, 20 frozen test, with VLM `gpt-5.6-luna` or `gpt-5.6-terra`.

项目：T2BlenderCode  
Skill：`t2blendercodeharness`  
文档用途：记录每一轮 Harness 训练、失败分析、代码 patch、train/dev/test 对比、Memory 和真实视频状态。

## 0. 先回答：我们到底在训练什么

当前不是训练神经网络权重，也没有 gradient descent。这里的“训练 Harness”是 code-level Harness evolution：

```text
固定 prompt/proxy 数据集
  -> 当前 Harness 生成 contract/trajectory/camera/proxy job/video
  -> evaluator 产生逐 case 分数和 failure evidence
  -> MetaHarnessOptimizer 聚合 train failure
  -> Coding Agent 只修改一个 Harness owner
  -> 用同一批输入重新生成候选结果
  -> train 提升且 dev 不下降才接受
```

所以正确的比较是：

| 固定不变 | 每轮允许变化 |
|---|---|
| prompt | Harness source code |
| proxy scene specification | parser/planner/executor 某一个 owner |
| train/dev/test case IDs | 重新生成的 contract/trajectory/camera |
| evaluator version | candidate proxy job、视频、telemetry |
| render settings/backend | candidate score |

“同一批视频”更准确地说是“同一批 case 的重新渲染结果”。候选 Harness 不能直接复用 baseline 的视频，否则只能评估 evaluator 或 video judge 的变化，不能证明 Harness 生成能力提升。

## 1. 训练数据和结果定义

当前 hard 数据集：

- dataset：`dataset/trajectory-v3-hard/`
- cases：100
- split：train 60、dev 20、test 20
- split policy：`template_family_holdout_compositional`
- fingerprint：`5809b5c862e7b5c6b9d4454a55b15887231d41205c0c8e75f4da03c1e8a45eba`
- 每个 case 有唯一 prompt hash 和 proxy scene ID
- proxy scene 包含布局、support/drop-zone、障碍物、路径形状和 artifact contract

当前数据资产：

- `dataset/trajectory-v3-hard/manifest.jsonl`
- `dataset/trajectory-v3-hard/proxy_specs.jsonl`
- `dataset/trajectory-v3-hard/splits.json`
- `dataset/trajectory-v3-hard/labels.jsonl`

当前 100 个 immutable Blender jobs：

- `out/real/trajectory-v3-hard-prepared-all/`
- job 数：100/100
- job source compile：100/100
- proxy.mp4：0/100，因为当前环境没有 Blender executable/MCP 执行结果

## 2. 一轮训练的标准定义

### 2.1 Baseline run

使用 parent Harness 对固定 dataset 的 train/dev 运行：

```text
prompt + proxy_scene
  -> parent contract
  -> parent trajectory/camera plan
  -> parent proxy job/video
  -> parent deterministic/oracle score
```

必须保存：

- parent Harness version；
- dataset/evaluator/backend fingerprint；
- 每个 case 的 score、status、failure IDs；
- plan hash、proxy scene ID；
- aggregate mean score、pass rate、plan collision rate。

### 2.2 Failure aggregation

只使用 train failure 选择 patch。按以下 key 聚合：

```text
failure_id + owner + category + severity
```

同一个 failure 至少影响两个不同 train case，才算 repeated actionable failure。

禁止：

- 使用 test failure 选择 owner；
- 将 VLM unavailable 当成失败标签；
- 将 parser、trajectory、camera、executor 同时改进合并成一个 patch；
- 因为 dev 好看而修改 test 标签。

### 2.3 Candidate run

候选 Harness 只允许修改一个 owner，然后用完全相同的：

- prompt；
- proxy scene specification；
- split；
- evaluator；
- backend；
- render settings；
- seed。

重新生成 candidate contract、trajectory、camera、proxy job 和视频。

### 2.4 Acceptance gate

候选只有同时满足以下条件才能接受：

```text
train_after > train_before
dev_after >= dev_before
hard_dev_regression == false
```

接受之后才运行 frozen test。test 结果用于最终报告，不得反向修改 patch。

## 3. 分数和视频状态的含义

本项目有三种不同的“结果”，不能混写：

| 结果 | 含义 |
|---|---|
| deterministic score | contract/plan/artifact/telemetry 规则分数 |
| independent-oracle score | 对数据集作者 gold expectations 的任务符合度 |
| VLM score | 真实 sampled frames/video 的视觉质量判断 |

当前 hard benchmark 的 score 是 deterministic score 减去 independent-oracle findings 的惩罚，不是 VLM 分数。

如果真实视频还没有生成：

```text
status = job_prepared
proxy video = NOT_RENDERED
```

不能把 job path 写成已经存在的 `proxy.mp4`，也不能把 fake benchmark score 描述成视频视觉分数。

## 4. 当前训练记录：h-t2-hard-v0 → h-t2-hard-v1

### 4.1 Run metadata

| 字段 | 值 |
|---|---|
| run ID | `t2blendercodeharness-hard-2026-08-25-parser-patch` |
| dataset | `trajectory-v3-hard` |
| evaluator | `deterministic-v2-independent-oracle` |
| parent Harness | `h-t2-hard-v0` |
| candidate Harness | `h-t2-hard-v1` |
| owner | `scene_parser` |
| affected files | `src/videoact/scene_contract.py` |
| affected train cases | 53 |
| Memory ID | `memory-2ea07c5a9e095993` |
| decision | accepted |

### 4.2 Score comparison

| Split | Cases | Parent mean | Candidate mean | Parent pass rate | Candidate pass rate |
|---|---:|---:|---:|---:|---:|
| train | 60 | 57.25 | 61.50 | 0.1167 | 0.1167 |
| dev | 20 | 30.25 | 50.50 | 0.0000 | 0.0000 |
| test | 20 | 22.75 | 40.00 | 0.0000 | 0.0000 |

test 是在 candidate 通过 acceptance 后才执行的 frozen final evaluation。

### 4.3 Candidate patch

本轮只修改 Scene/Prompt Parser，增加了：

- stroll、advance、seize、snatch、hoist、elevate、convey；
- settle、unhand、uncouple；
- foreground blocker、partition、assistant、opening。

本轮没有修改 trajectory planner、camera planner、evaluator，因此分数变化可以归因到一个 Harness owner。

### 4.4 Memory event chain

```text
0 proposal
1 patch_applied
2 train_evaluated
3 dev_evaluated
4 accepted
5 test_evaluated
```

完整 Memory：

`out/training/t2blendercodeharness-hard-final-v2/memory/harness_updates.jsonl`

完整训练报告：

`out/training/t2blendercodeharness-hard-final-v2/training_report.json`

## 5. 当前仍然暴露出的错误

Candidate 之后，主要 failure 仍然包括：

- `oracle_event_order_mismatch`：多物体重复 handoff、pause、return 等事件尚未完整表达；
- `oracle_constraint_missing`：no-cut、no-penetration、post-release-hold 等约束未充分进入 contract；
- `oracle_camera_intent_missing`：hold、反向弧线、复杂 reveal camera 尚未完整规划；
- `oracle_motion_primitive_missing`：S-curve、zigzag、dolly/orbit 约束仍会折叠成基础 primitive；
- `oracle_attachment_lifecycle_mismatch`：双物体、双角色 handoff 尚未完整建模；
- `oracle_proxy_entity_mismatch`：部分 blocker、assistant、opening 等 proxy entity 仍未进入 runtime plan。

因此当前状态是“parser patch 已被接受，但 Harness 尚未达到通过标准”。下一轮应该选择 `trajectory_planner` 或 `camera_planner` 的一个 owner，不能两个一起改。

## 6. 下一轮训练记录模板

复制以下区块追加到本文件末尾，或另存为新的 run log：

```text
## Run: <date>-<short-name>

### Metadata
- dataset:
- dataset fingerprint:
- evaluator:
- backend:
- parent Harness:
- candidate Harness:
- owner:
- affected files:
- proposal case IDs:
- Memory ID:

### Failure evidence
- repeated failure ID:
- count:
- distinct train cases:
- evidence paths:
- desired behavior:

### Patch
- patch summary:
- exact files:
- tests added/updated:

### Scores
| Split | Cases | Parent mean | Candidate mean | Parent pass rate | Candidate pass rate |
|---|---:|---:|---:|---:|---:|
| train | | | | | |
| dev | | | | | |
| test | | | | | |

### Acceptance
- train improved:
- dev non-regressed:
- hard dev regression:
- decision: accepted / rejected / rollback / no_patch
- reason:

### Real video status
- prepared jobs:
- rendered videos:
- artifact-complete:
- deterministic-pass:
- VLM status:

### Memory events
1. proposal
2. patch_applied / rejected / no_patch
3. train_evaluated
4. dev_evaluated
5. accepted / rollback
6. test_evaluated

### Interpretation
- what improved:
- what still fails:
- next owner candidate:
```

## 7. Recommended commands for the next run

```powershell
$py = 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

# Validate fixed dataset
& $py scripts\validate_hard_trajectory_dataset.py

# Generate immutable jobs for the same cases
& $py scripts\prepare_real_jobs.py `
  --split dev `
  --dataset-root dataset\trajectory-v3-hard `
  --out-dir out\real\trajectory-v3-hard-prepared-next `
  --harness-version <candidate-version> `
  --evaluator-version deterministic-v2-independent-oracle

# Run strict train/dev acceptance and frozen test
& $py scripts\run_hard_end_to_end_training.py `
  --dataset-root dataset\trajectory-v3-hard `
  --baseline-root <baseline-output> `
  --out-root <candidate-output> `
  --candidate-harness-version <candidate-version>

# Build prompt -> proxy video/job -> score index
& $py scripts\build_dataset_prompt_proxy_score_index.py `
  --dataset-root dataset\trajectory-v3-hard `
  --benchmark-root <candidate-output> `
  --real-root <real-job-root> `
  --output-dir out\indexes\<run-name>

# Verify project and skill
& $py skills\t2blendercodeharness\scripts\capability_check.py --project-root .
& $py -m pytest -q
```

## 8. 记录原则

1. 同一 case 的 baseline/candidate 必须使用同一 prompt 和 proxy spec。
2. Candidate 必须重新生成 plan/job/video；不能复用 baseline 视频冒充 Harness 改进。
3. evaluator 版本变化时，必须单独标记为 evaluator recalibration，不得混入 Harness score。
4. VLM unavailable 只能记录 unavailable，不能生成负样本。
5. test 只在 acceptance 后运行，且不能影响下一轮 patch。
6. 每一个 proposal、patch、reject、rollback、accepted 和 no_patch 都必须写入 Memory。
7. 分数提升但 pass rate 下降时，优先检查 score aggregation 和 per-case regression，不能只看 mean score。
