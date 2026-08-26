# T2Blendercodeharness Hard Generalization Experiment

> Historical 100-case baseline. The active anti-overfitting protocol is the 140-case dataset with 60 train, 60 dev, 20 frozen test and six 10-case rounds; see [six-round protocol](t2blendercodeharness-six-round-protocol.md).

实验日期：2026-08-25  
项目：T2BlenderCode  
技能：`t2blendercodeharness`  
数据集：`trajectory-v3-hard`  
评测器：`deterministic-v2-independent-oracle`

## 1. 结论先行

上一版 `trajectory-v2` 的 train/dev/test 全部 100 分不能作为 Harness 泛化证据。原因是旧 benchmark 主要检查 parser/planner 自己生成的结构是否自洽，且样本是同一模板族的词汇变体，存在模板泄漏和 evaluator 自证循环。

本轮重新构造了 100 个不同 prompt 和不同 proxy scene specification，并使用模板族留出、独立 oracle 约束、proxy 场景元数据和 plan-collapse 指标。

| 阶段 | Train | Dev | Test |
|---|---:|---:|---:|
| Hard baseline `h-t2-hard-v0` | 57.25 | 30.25 | 22.75 |
| Parser patch `h-t2-hard-v1` | 61.50 | 50.50 | 40.00 |

这次 patch 的 acceptance gate 通过：train 严格提升，dev 不下降；但三套 split 的 pass rate 仍未达到可接受水平，当前结论是“发现并部分修复了泛化问题”，不是“Harness 已训练完成”。

## 2. 研究问题

1. Harness 是否只记住模板词汇，而不能组合泛化到未见过的动作、实体和摄像机组合？
2. 真实 proxy scene 的布局、障碍物、目标位置和路径形状是否真正进入执行链，而不是只存在于 prompt？
3. 独立 gold/oracle 检查能否暴露 self-consistency evaluator 无法发现的错误？
4. 一个只修改单一 owner 的 Harness patch 是否能在 train 提升的同时保持 dev 不下降？

## 3. 方法与 pipeline

### 3.1 内循环

```text
prompt + proxy_scene
  -> SceneContract
  -> TrajectoryPlan + CameraPlan
  -> fake deterministic execution / immutable Blender job
  -> normal deterministic evaluator
  -> independent oracle evaluator
  -> score + failure evidence
```

### 3.2 外循环

```text
train failures
  -> normalized failure aggregation
  -> one-owner proposal
  -> apply one Harness component patch
  -> rerun train
  -> rerun dev
  -> accept iff train_after > train_before and dev_after >= dev_before
  -> frozen test only after acceptance
```

测试集没有参与 patch owner、patch 内容或 acceptance 决策。按组合泛化 benchmark 的通行做法，split 按结构/模板族留出，而不是只做随机样本切分；随机 IID split 在组合任务上可能接近完美，但新组合 split 会显著降低性能。[TACL compositional generalization study](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00361/98090/Latent-Compositional-Representations-Improve)

## 4. Hard 数据集

### 4.1 规模与切分

| 字段 | 值 |
|---|---:|
| cases | 100 |
| train | 60 |
| dev | 20 |
| test | 20 |
| template families | 10 |
| unique prompt hashes | 100 |
| unique proxy scene IDs | 100 |
| split policy | `template_family_holdout_compositional` |
| fingerprint | `5809b5c862e7b5c6b9d4454a55b15887231d41205c0c8e75f4da03c1e8a45eba` |

Train 使用前 6 个 family，dev 使用第 7、8 个 family，test 使用第 9、10 个 family。family 不跨 split，防止“同模板只换物体名称”的泄漏。

### 4.2 轨迹和 prompt 难点

数据覆盖以下类型：

- 标准 `walk → reach → grasp → lift → carry → place → release`；
- 同义动作：`stroll / advance / seize / snatch / hoist / elevate / convey / unhand / uncouple`；
- 双物体 handoff 和重复 attachment lifecycle；
- foreground blocker、partition、opening、occlusion/reveal；
- pause、return、receive、hold 等当前 schema 尚未完整表达的事件；
- locked hold、follow、orbit、dolly 的混合镜头节奏；
- S-curve、zigzag、reverse arc、对角线搬运和不允许穿透；
- assistant/receiver 等第二角色；
- “do not cut away during contact”等反事实相机约束。

### 4.3 每个 case 的 proxy scene

每条 manifest record 都有独立的 `proxy_scene`：

- 唯一 `scene_id` 和 `scene_seed`；
- support、drop zone、character start 的三维位置；
- support/object scale；
- `straight / arc / zigzag / s_curve / reverse_arc` 路径形状；
- lighting rig 变体；
- foreground blocker、partition、opening、assistant 等实体；
- `proxy.blend`、`proxy.mp4`、`telemetry.json`、sampled frames 的 artifact contract。

因此 prompt 相同语义时，proxy geometry 也不再相同。数据文件：

- `dataset/trajectory-v3-hard/manifest.jsonl`
- `dataset/trajectory-v3-hard/proxy_specs.jsonl`
- `dataset/trajectory-v3-hard/splits.json`
- `dataset/trajectory-v3-hard/metadata.json`
- `dataset/trajectory-v3-hard/labels.jsonl`

labels 当前明确标记为 `unreviewed`；它们不是伪造的 VLM 偏好标签。

## 5. Evaluator 修正

旧 evaluator 仍保留，用于检查 plan 内部合法性：事件时间、camera coverage、support-before-grasp、attachment contact、velocity continuity。

新增 `evaluator/independent_oracle.py`，由数据集作者写入的 oracle expectation 检查：

- gold event order 是否真正被 parser 表达；
- camera trajectory types 是否覆盖 prompt 要求；
- motion primitives 是否包含 linear/ease/hold/orbit/dolly 等要求；
- attach/detach 生命周期和重复 handoff 是否一致；
- prompt 需要的实体和 proxy entity kinds 是否存在。

如果 runtime planner 自己生成一个内部自洽但任务语义错误的 plan，normal evaluator 可以通过，而 independent oracle 会失败。`run_benchmark.py` 同时输出：

- score；
- failure IDs；
- `unique_plan_count`；
- `plan_collision_rate`。

plan collision 是额外的过拟合诊断：不同 prompt/scene 生成同一个 plan hash，说明 Harness 忽略了输入中的部分条件。

## 6. 训练与 Harness 更新

### 6.1 Baseline

产物：`out/training/t2blendercodeharness-hard-baseline/`

| Split | Cases | Mean score | Pass rate |
|---|---:|---:|---:|
| train | 60 | 57.25 | 0.1167 |
| dev | 20 | 30.25 | 0.0000 |
| test | 20 | 22.75 | 0.0000 |

最高频 train failure 是 `oracle_event_order_mismatch`，40 个 case，owner 为 `scene_parser`。MetaHarnessOptimizer 只生成了一个 scene-parser proposal，没有把 camera、trajectory、evaluator 混成一个 patch。

### 6.2 已应用 patch

修改文件：`src/videoact/scene_contract.py`

本次只扩展 Scene/Prompt Parser：

- stroll、advance、seize、snatch、hoist、elevate、convey、settle、unhand、uncouple 等词汇归一化；
- foreground blocker、partition、assistant、opening 等实体识别。

没有同时修改 camera planner、trajectory planner 或 evaluator，因此 acceptance 结果可以归因于一个 owner。

### 6.3 Candidate 与 acceptance

产物：`out/training/t2blendercodeharness-hard-final-v2/`

| Split | Baseline | Candidate | 变化 |
|---|---:|---:|---:|
| train | 57.25 | 61.50 | +4.25 |
| dev | 30.25 | 50.50 | +20.25 |
| test | 22.75 | 40.00 | +17.25（接受后冻结评测） |

Acceptance：

```text
owner: scene_parser
train_before: 57.25
train_after: 61.50
dev_before: 30.25
dev_after: 50.50
hard_regression: false
decision: accepted
```

### 6.4 Memory

Memory：`out/training/t2blendercodeharness-hard-final-v2/memory/harness_updates.jsonl`  
Memory ID：`memory-2ea07c5a9e095993`

事件完整保留为：

```text
0 proposal
1 patch_applied
2 train_evaluated
3 dev_evaluated
4 accepted
5 test_evaluated
```

`test_evaluated` 位于 `accepted` 之后，满足 frozen-final-only 策略。

## 7. 端到端 proxy job 验证

使用 `scripts/prepare_real_jobs.py` 为 dev 20 条样本生成 immutable Blender jobs：

- jobs generated：20/20；
- unique proxy scene IDs：20/20；
- job source compile：20/20；
- 每个 run directory 有 `proxy_scene.json`；
- generated job 将 proxy scene 的 scene seed、布局、路径形状写入 telemetry。

产物：`out/real/trajectory-v3-hard-dev-prepared-final/`

本轮当前环境的 shell 没有 `blender` executable，且没有在此阶段连接新的 Blender MCP session，因此 v3-hard 的 20 个 job 做到了“准备、编译、契约检查”，没有把未执行的 job 伪称为真实渲染。真实 `.blend/.mp4/PNG` 仍需在连接 Blender MCP 后执行；VLM endpoint 仍按项目规则记录为 unavailable。

## 8. Plan collapse 诊断

Candidate 的独立 plan 统计：

| Split | Cases | Unique plans | Collision rate |
|---|---:|---:|---:|
| train | 60 | 31 | 48.33% |
| dev | 20 | 14 | 30.00% |
| test | 20 | 17 | 15.00% |

这说明“prompt 不同、proxy scene 不同”并不自动意味着 Harness 已正确利用这些信息。当前 scene parser patch 提升了语义词汇覆盖，但路径形状、具体布局、多物体生命周期和复杂相机约束仍大量折叠为相同计划；这正是下一轮 trajectory/camera owner patch 的候选证据。

## 9. Skill 命名与能力检查

技能已从项目内的 `skills/autodesign-harness/` 重命名为：

```text
skills/t2blendercodeharness/
```

Frontmatter：`name: t2blendercodeharness`。能力检查结果：

```text
skill_version: t2blendercodeharness-v1
status: pass
required_components: pass
imports: pass
contract_and_plan: pass
evaluator_interfaces: pass
one_owner_proposal: pass
no_action_on_clean_records: pass
```

能力检查脚本：`skills/t2blendercodeharness/scripts/capability_check.py`。

## 10. 复现实验

```powershell
$py = 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'

& $py scripts\build_hard_trajectory_dataset.py
& $py scripts\validate_hard_trajectory_dataset.py

# baseline evidence / benchmark
& $py scripts\train_harness_with_memory.py `
  --dataset-root dataset\trajectory-v3-hard `
  --out-root out\training\t2blendercodeharness-hard-baseline `
  --harness-version h-t2-hard-v0 `
  --evaluator-version deterministic-v2-independent-oracle

# strict candidate acceptance + frozen test
& $py scripts\run_hard_end_to_end_training.py `
  --dataset-root dataset\trajectory-v3-hard `
  --baseline-root out\training\t2blendercodeharness-hard-baseline `
  --out-root out\training\t2blendercodeharness-hard-final-v2 `
  --candidate-harness-version h-t2-hard-v1

# prepare real Blender jobs
& $py scripts\prepare_real_jobs.py `
  --split dev `
  --dataset-root dataset\trajectory-v3-hard `
  --out-dir out\real\trajectory-v3-hard-dev-prepared-final `
  --harness-version h-t2-hard-v1 `
  --evaluator-version deterministic-v2-independent-oracle

& $py skills\t2blendercodeharness\scripts\capability_check.py --project-root .
& $py -m pytest -q
& $py -m compileall -q src evaluator blender scripts training tests skills\t2blendercodeharness
```

## 11. 当前局限与下一步

1. pass rate 仍为 0，分数上升只代表部分约束被修复，不代表端到端质量达标。
2. trajectory planner 还没有完整消费 proxy layout 的 path shape、障碍物几何和多物体生命周期。
3. camera planner 对 hold、反向弧线、遮挡穿越、接触期间 no-cut 等要求表达不足。
4. v3-hard 的 labels 仍未完成人工/VLM calibration；VLM unavailable 不能转成 0 分。
5. 20 个 real jobs 已生成和编译，但需要连接 Blender MCP 后才能完成真实 `.blend`、PNG animation、MP4、telemetry 和 artifact evaluator。

下一轮应优先选择 `trajectory_planner` 或 `camera_planner` 的单 owner patch，目标是降低 plan collision、提升 dev pass rate，并在 test 完全冻结的情况下重新执行 acceptance gate。

## 12. 产物索引

- Skill：`skills/t2blendercodeharness/SKILL.md`
- Hard dataset：`dataset/trajectory-v3-hard/`
- Independent oracle：`evaluator/independent_oracle.py`
- Benchmark：`scripts/run_benchmark.py`
- Strict training：`scripts/run_hard_end_to_end_training.py`
- Baseline report：`out/training/t2blendercodeharness-hard-baseline/training_report.json`
- Final report：`out/training/t2blendercodeharness-hard-final-v2/training_report.json`
- Memory：`out/training/t2blendercodeharness-hard-final-v2/memory/harness_updates.jsonl`
- Real job batch：`out/real/trajectory-v3-hard-prepared-all/`
- Prompt/proxy/score index：`out/indexes/trajectory-v3-hard/prompt_proxy_score_index.md`
- Harness training process log：`docs/t2blendercodeharness-training-process-log.md`
