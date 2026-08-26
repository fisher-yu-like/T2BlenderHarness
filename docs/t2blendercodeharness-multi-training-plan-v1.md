# T2Blendercodeharness 多实体五轮真实训练方案

状态：待用户批准；本文件创建时尚未启动批量训练。  
分支：`codex/director-multi-entity-harness`  
协议：`multi-five-rounds-v1`

## 1. 当前 Harness 架构

```text
Codex Host
├── director-agent skill / t2blendercodeharness skill
├── DirectorAgent
│   ├── director_prompt.py        稳定实体 ID、evidence、uncertainty
│   ├── director_schedule.py      事件顺序、依赖、时间窗口
│   ├── director_trajectory.py    人物/物体状态、owner、handoff 生命周期
│   ├── director_camera.py        follow/orbit/dolly、多目标可见性
│   └── director.py               单一规划入口和结果组装
├── Compatibility projections
│   ├── scene_contract.py         DirectorPlan → SceneContract
│   └── director_projection.py    Director trajectories/camera → legacy plan
├── Blender execution
│   ├── blender/real_proxy_job.py 真实 proxy.blend 和 telemetry job
│   ├── scripts/render_proxy_jobs_parallel.py
│   └── D:\blender\blender.exe  每 case 隔离、失败最多重试 2 次
├── Evaluator
│   ├── RealArtifactGate           文件、hash、MP4、PNG、telemetry
│   ├── deterministic.py            合同/运行时 hard gate 与独立 task 分
│   ├── director_metrics.py        Director integrity/coverage/collision
│   ├── interaction_metrics.py     attach/transfer/detach/owner/support
│   ├── geometry + visual evidence 几何与采样帧真实性前置检查
│   ├── shared_review.py            一次 review 同时产 task/realism 两通道
│   ├── aggregate.py                VLM 主导的 task_final 汇总
│   └── realism.py                  单独 realism 汇总，不与 task 相加
└── MetaHarnessOptimizer
    ├── repeated failure/root-cause 聚合
    ├── one-owner patch proposal
    └── paired train/dev/overall dev acceptance gate
```

## 2. 输入如何经过 Harness

对每个 exact prompt，DirectorAgent 先做证据绑定的解释：例如把 Alice、
Carla、green ball 分成 `actor_a`、`actor_b`、`green_ball`，并给 carry、
handoff、place 建立依赖和时间窗。它随后生成人物轨迹、物体 attachment
事件和 current owner，同时生成覆盖 giver、receiver、prop 的 handoff
two-shot 以及后续 follow/dolly/reveal 镜头。只有 DirectorPlan 通过 schema
与 evaluator integrity 检查后，才投影到旧的 SceneContract/TrajectoryPlan
接口并编译 Blender job。

真实执行保存 `director_plan.json`、`trajectory.json`、`camera_plan.json`、
`blender_job.py`、`proxy.blend`、动画帧序列、host MP4、telemetry、
`render_attempts.json` 和所有 fingerprint。任何失败都进入
`NOT_RENDERED`，不会有虚假的视频地址或分数。

## 3. Evaluator 逻辑

确定性层先检查 artifact 完整性、实体 kind、timebase、camera active、
required event coverage、轨迹阶段、support-before-grasp、contact、
velocity continuity。多实体独立层再检查 evidence/assumption、稳定 ID、
dependency、路径碰撞、多目标 visibility 和 telemetry identity；interaction
层检查 giver detach、receiver attach、handoff window、最终 owner/support。
这些分数和 finding 不与 VLM/realism 混成一个黑箱分数。

有合格视觉 review 时，任务分为：

```text
semantic = GM(prompt_compliance, physical_plausibility,
              object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_vlm = .45 * semantic + .45 * choreography + .10 * visual_clarity
task_final = .20 * deterministic_score + .80 * task_vlm
```

真实性单独为：

```text
artifact_only_proxy = min(80, .80 * (.60 * geometry + .40 * frame_evidence))
reviewed_realism = .15 * geometry + .15 * frame_evidence +
                   .70 * GM(appearance_detail, physical_realism,
                             spatial_consistency, motion_naturalness,
                             visual_presentation)
```

没有 external `gpt-5.6-luna`/`gpt-5.6-terra` 或 Codex local review 的可审计
可见证据时，task review score 为 unavailable；只保留 capped
`artifact_only_proxy`，并写 `realism_claim=not_established`。当前 smoke 的
artifact-only realism 为 68.5519，不视为真实性通过。

## 4. 数据集冻结内容

路径：`dataset/trajectory-v4-multi`  
fingerprint：`4d0abd0eb387bc58c4b4cd03259874670a5c7010ecbc484f056cbf3b255a0f41`

| split | 数量 | difficulty 均值 | 组成 |
|---|---:|---:|---|
| train | 50 | 4.5 | sequential_transfers、repeated_handoffs、concurrent_independent_work、occlusion_reveal、role_swap_pause_return_crossing，各 10 |
| dev | 60 | 7.5 | 6 个未与 train composition 重合的 family，各 10 |
| test | 30 | 9.5 | test_role_reversal_final_owner、test_counterfactual_camera_visibility、test_prohibited_crossing_support，各 10 |

全量 140 条 prompt 都有不同 case/prompt hash，并带 authored event graph、
interaction lifecycle、camera evidence、negative constraints 和
oracle expectations。实体规模为 104 条双人物、36 条三人物；prop 规模为
110 条双物体、15 条三物体、15 条四物体。dev/test 更难，且 family 与
composition 没有跨 split 泄漏。

Prompt 不是简单关键词句：包含人物/物体身份、carry/handoff/place 或
return/pause/crossing 顺序、ownership/support 接触、相机 follow/orbit/dolly/
reveal 约束、进入/等待/遮挡时机以及负面约束。例如：

> Alice carries the green ball, then Alice hands the green ball to Carla and
> Carla places the green ball; then Carla carries the yellow book and places
> the yellow book through a separate lane. The scene contains green ball and
> yellow book; use a wide establishing hold, a curved follow, and a short
> rack-like reveal. Preserve identity, event order, handoff timing, support
> contact, and readable camera coverage.

## 5. 五轮外循环执行

每轮只新增 10 个 train，并用 10 个 paired dev；每次尝试最多 20 个真实
视频，最多 5 次 outer attempt。Blender 单 case 最多重试 2 次。每轮结束
重新生成该 Harness 版本的累计 train 与完整 60 dev，不能用 paired dev
冒充整体 dev。各轮最大视频量是 170、180、190、200、210，总计 950；
第五轮结束后追加一次 30-case frozen test（no proposal）。

每次按以下顺序执行：

1. 固定 fingerprint，写 protocol manifest。
2. 用 12 workers 准备和真实渲染 10 train + 10 paired dev。
3. 每个 case 完成 artifact、deterministic、Director、interaction、几何
   和 frame gate 后，调用一次 shared visual review。
4. endpoint 不可用时记录 unavailable；Codex 若进行 local review，只能
   根据 request 中列出的真实 PNG 写有 evidence、weaknesses、confidence
   的审计 payload，confidence < 0.6 不进数字聚合。
5. 立即把每个 case 写入 Memory 表；失败写 `NOT_RENDERED` 和原因。
6. 只从两个不同 train case 的重复 root cause 形成 proposal。
7. 一轮最多修改一个 Harness owner；不得改 evaluator、dataset、Blender
   或 plan 以追分。
8. 以同 fingerprint 重跑 paired train/dev；train 严格上升、paired dev
   不下降、整体 dev 不下降、hard failure 不增加、artifact completion
   不下降才接受。renderer/realism patch 还必须 realism 上升且 task 不降。
9. 没有新 evidence、无重复 failure、patch 被拒或达到第 5 次就停止本轮
   尝试；每轮结束写 overall report 和 Memory。

## 6. Memory 表格和交付物

Memory 每行必须包含：轮数、attempt、split、case、完整 prompt、绝对 proxy
视频地址或 `NOT_RENDERED`、immutable DirectorPlan hash、Director score、
task score、realism score、review source/confidence、Harness 错误点、
Harness 修改位置/方法、修复前后 delta 和自然语言处理说明。

最终交付：

- `docs/t2blendercodeharness-multi-training-memory-v1.md`
- `docs/t2blendercodeharness-multi-training-report-v1.md`
- `docs/figures/multi-training-curves-v1.png`
- `out/training/multi-five-rounds-v1/` 下的每轮真实 jobs、reports、patch
  manifest、`memory/harness_updates.jsonl`
- 最终 full suite、capability、dataset validator 和 blind test 证据

当前状态：Task 15 smoke 已完成；Task 16 批量训练尚未启动，等待用户批准
本方案和 visual-review 分支。
