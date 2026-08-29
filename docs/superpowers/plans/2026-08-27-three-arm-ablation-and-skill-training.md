# Three-Arm Harness Comparison and Skill Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一批复杂 prompt 上比较训练前 Harness、当前训练后 Harness 和“不经过 DirectorAgent 的 prompt→Blender code”消融组，覆盖全部事件类型、轨迹/摄像机调度、task 与 realism，并用真实视频证据驱动下一轮 skill 自进化。

**Architecture:** 保持现有 `DirectorAgent` Harness 作为 trained arm 和 pretraining arm 的唯一规划入口；新增仅用于实验的 `direct_prompt_code` ablation adapter，它只接收原始 prompt、duration、FPS 和 seed，不导入 DirectorPlan、SceneContract 或 TrajectoryPlan。三臂共享 Blender、渲染设置、artifact gate、盲视觉评分和统计汇总，但 Director 合同分与跨臂视觉分严格分开。

**Tech Stack:** Python 3.11+, existing Pydantic contracts, Blender 5.1.2 at `D:\blender\blender.exe`, Blender CLI, EEVEE Next, JSON/JSONL, PNG contact sheets, existing task/realism evaluator, `pytest`.

---

## 1. Frozen experiment inputs

| Item | Frozen value |
|---|---|
| Dataset | `dataset/vbench-derived-100-v1` |
| Dataset size | 100 unique prompts; 70 train-like / 30 dev-like |
| Event variants | `direct_transfer`, `reveal_elliptical_return`, `subjectless_handoff_return`, `parallel_transfer`；每类 25 条 |
| Trained arm | current worktree, commit `306e4d2`, label `h-t2-hard-v4-director-prompt-elliptical-return-order-v1` |
| Untrained arm | `C:\Users\sy\Desktop\T2BlenderCode\.worktrees\vbench-pretrain`, detached commit `7fe017a` |
| Ablation arm | current worktree + `direct_prompt_code` adapter；不调用 `DirectorAgent` |
| Blender | `D:\blender\blender.exe` |
| Render | EEVEE Next, 128×128, 1 sample, 12 FPS, 6 seconds, 72 animation frames |
| Concurrency | 12 isolated Blender CLI workers |
| Retry | 每个 Blender job 失败最多重试 2 次 |
| VLM | 首选小写 `gpt-5.6-luna` / `gpt-5.6-terra`；不可用则 Codex local review；两者都不可用写 `unavailable` |

现有 trained/pretrain 200 个 VBench 视频已经是真实 Blender 产物，可以复用但必须重新走盲评；ablation arm 新生成 100 个真实 MP4。不能把旧的 action-specific local preset 直接作为本实验最终分数。

## 2. Three-arm execution matrix

### Arm A: pretraining Director Harness

从 detached worktree `7fe017a` 运行既有 `prepare_real_jobs.py`，输入完整 `prompt`，由该版本的 DirectorAgent 生成计划、轨迹、摄像机和 Blender job。使用现有的 baseline 视频作为 artifact baseline，并重新读取其实际 PNG/MP4。

### Arm B: trained Director Harness

从当前 worktree `306e4d2` 运行相同命令、相同 dataset、相同 render settings 和相同 case IDs。生成的 `director_plan_hash`、trajectory、camera plan 与 Arm A 分别保存，不能覆盖既有 benchmark 输出。

### Arm C: direct prompt→Blender code ablation

新增实验专用入口：

- `scripts/prepare_direct_prompt_jobs.py`
- `blender/direct_prompt_code.py`
- `blender/direct_prompt_job.py`

输入仅允许 `prompt`, `case_id`, `duration_s`, `fps`, `seed` 和 frozen render settings。该 adapter 不得导入 `videoact.director`、`DirectorPlan`、`SceneContract` 或 `TrajectoryPlan`，也不得读取 dataset 的 `event_graph`、`oracle_expectations` 或 `camera_evidence` 来偷渡答案。

它可以从原始 prompt 中抽取最基本的名称/颜色 token，生成一个直接 Blender Python job，并把 raw prompt 与 code hash 写入 manifest。它不生成 Director plan，不伪造 Director score；缺少计划级证据在报告中标为 `not_applicable`。它必须仍然产出真实 `proxy.blend`、`proxy.mp4`、telemetry、PNG index 和 render attempt log。

## 3. Evaluator protocol

### 3.1 Eligibility and structural channels

三臂全部先执行 artifact gate、MP4 可播放检查、PNG 可读检查和 geometry audit。A/B 额外执行 Director/interaction deterministic 检查；C 只执行通用 artifact/runtime 检查，不能因为没有 DirectorPlan 而伪造 0 分 Director score。

保存这些独立字段：

```text
artifact_completion
deterministic_score (A/B only when Director contract exists)
director_plan_score (A/B only)
interaction_score (A/B when lifecycle exists)
geometry_score
frame_evidence_score
```

### 3.2 Blind visual review

对每个 case 的三个 arm 随机改名为 `sample_a/sample_b/sample_c`，contact sheet 和 review request 不暴露 arm、branch、commit 或 action variant。reviewer 只看真实 MP4 的固定时间采样帧；不能从 plan、telemetry 或 action label 推断不可见运动。

Task 原始维度全部记录：

```text
prompt_compliance
physical_plausibility
object_trajectory
event_timing
camera_coverage
camera_innovation
character_trajectory
temporal_smoothness
visual_clarity
```

Realism 原始维度全部记录：

```text
appearance_detail
physical_realism
spatial_consistency
motion_naturalness
visual_presentation
```

保持现有主公式，不再用 action variant 直接加分：

```text
semantic = GM(prompt_compliance, physical_plausibility,
               object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_vlm = .45 × semantic + .45 × choreography + .10 × visual_clarity

reviewed_realism = GM(appearance_detail, physical_realism,
                       spatial_consistency, motion_naturalness,
                       visual_presentation)
realism_final = .15 × geometry + .15 × frame_evidence + .70 × reviewed_realism
```

跨三臂的主排名使用 `task_vlm` 和 `realism_final`；`deterministic_score`、`director_plan_score` 和 `interaction_score` 作为合约/结构诊断单独报告。这样不会因为消融组没有 DirectorPlan 而把它的跨臂视觉能力直接压成零。

为体现轨迹和摄像机调度，报告同时按原始维度给出均值，并增加仅用于诊断的：

```text
trajectory_diagnostic = GM(object_trajectory, character_trajectory,
                           event_timing, temporal_smoothness)
camera_diagnostic = GM(camera_coverage, camera_innovation, visual_clarity)
```

不把这两个诊断分数再次加入 `task_vlm`，避免重复加权。

### 3.3 Statistics

对每个 action variant、VBench source dimension、train-like/dev split 和全体 100 条分别报告：

- 三臂均值和 paired delta：`trained - pretrain`、`trained - direct_code`、`pretrain - direct_code`；
- task_vlm、realism_final、trajectory_diagnostic、camera_diagnostic；
- 9 个 task 维度和 5 个 realism 维度；
- 95% paired bootstrap confidence interval；
- artifact completion、hard-failure rate、review availability；
- 不允许用空值、不可用 review 或缺视频补成 0。

这次四种 action variant 都是正式比较对象，不再把 `direct_transfer` 和 `parallel_transfer` 只作为隐含 control 组。

## 4. Real-video skill self-evolution

skill 自进化的证据来源必须是三臂真实视频的 artifact/evaluator/review 记录，而不是只读 plan 或人工猜测。

执行一个与 Harness 外循环相似但对象不同的 skill loop：

1. 固定 70 train-like / 30 dev-like，先生成三臂真实视频并完成统一盲评。
2. 只从 train-like 的重复 failure 中提取 proposal；同一 normalized failure 必须影响至少两个不同 case。
3. 将 failure 路由到一个 skill owner：`director-agent`、`t2blendercodeharness`、`blender-proxy-executor`、`harness-evolution` 或 `evaluator-interface`。
4. 一次只修改一个 skill 文件的一个章节；不能修改 Harness 源码、dataset labels 或 evaluator 公式来提高分数。
5. 每次最多 5 个 proposal attempts；每次都重新运行 capability check、skill validator、全量测试和原 prompt/variation forward-test。
6. Skill 改动如果只改变操作约束而不改变运行时 Harness，必须明确记录“视频分数不应变化”；它的验收指标是 fail-closed、证据完整、owner 路由和回归安全。
7. 只有当下一次真实运行确实按新 skill 生成视频、产生可审计 evidence 且 train/dev 规则不回退，才记录 `accepted`；否则记录 `rejected` 或 `blocked`。
8. `trajectory-v4-multi` 的 30 test 只在最后做一次 blind audit，不能选择 skill proposal。

Skill 训练不是神经网络权重训练；它是由真实三臂视频失败模式驱动的、proposal-only 的操作规范进化。任何 score 提升都必须归因到实际 Harness patch，不能归因给只改变文字的 skill。

## 5. Files and tests

**Create:**

- `scripts/prepare_direct_prompt_jobs.py`
- `blender/direct_prompt_code.py`
- `blender/direct_prompt_job.py`
- `scripts/prepare_three_arm_reviews.py`
- `scripts/aggregate_three_arm_ablation.py`
- `scripts/build_three_arm_skill_records.py`
- `tests/test_direct_prompt_ablation.py`
- `tests/test_three_arm_evaluator.py`
- `tests/test_three_arm_blind_review.py`
- `docs/t2blendercodeharness-three-arm-ablation-v1.md`

**Modify:**

- `skills/t2blendercodeharness/SKILL.md`
- `skills/t2blendercodeharness-zh/SKILL.md`
- `skills/director-agent/SKILL.md`
- `skills/t2blendercodeharness-training/SKILL.md`

**Do not modify:**

- `dataset/vbench-derived-100-v1`
- `dataset/trajectory-v4-multi`
- existing `out/benchmarks/vbench-100-current-vs-pretrain-v1`
- existing `out/training/multi-five-rounds-v1`
- evaluator score formulas during the benchmark

## 6. Execution order and acceptance

- [x] Write failing tests for the no-Director import boundary, three-arm schema, blinded arm labels, all 14 visual dimensions, null handling, and paired action summaries.
- [x] Run the failing tests and preserve the RED output.
- [x] Implement the direct prompt-to-code ablation adapter and minimal job compiler.
- [x] Run focused tests and compileall. Capability check remains a post-render acceptance gate.
- [x] Prepare 100 direct-code jobs; reuse only the immutable current/pretrain jobs/videos after verifying their fingerprints.
- [x] Render the 100 direct-code jobs with 12 total Blender CLI workers and up to 2 retries; record every result immediately.
- [x] Rebuild randomized blind review sheets for all 300 arm/case videos and review exact sampled frames.
- [x] Aggregate all four action variants and all task/realism/trajectory/camera dimensions with no preset arm bonus.
- [x] Build real-video skill records from train-like failures, emit a one-owner proposal, and apply at most one skill-section change for this attempt.
- [x] Re-run capability, dataset validation, full tests, blind leakage checks, and compile validation after proposal selection. Frozen test remains intentionally untouched because this experiment is an ablation/skill-evidence run, not a new Harness patch.
- [x] Generate the Chinese final report with per-case video paths, raw dimensions, paired deltas, failure findings, skill changes, confidence intervals and curves.

Acceptance requires: 300 real videos or an explicit immutable reuse record for the 200 existing videos; zero unlogged render failures; all four action variants present; all 14 visual dimensions present or explicitly `unavailable`; no action-specific score injection; no cross-split leakage; no Director import in the ablation; and no source/evaluator/dataset mutation outside the declared experiment files.
