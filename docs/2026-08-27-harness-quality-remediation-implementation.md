# T2Blendercodeharness 质量整改实施记录

> 本文件保留整改过程证据。训练前的当前架构、评分边界和门禁以
> [harness-architecture-v2.md](harness-architecture-v2.md)、
> [evaluator-v5-calibration.md](evaluator-v5-calibration.md) 和统一主计划为准；
> 本文件中的旧 smoke/历史协议数字不能替代 active agent-codegen 结果。

日期：2026-08-27  
分支：`codex/director-multi-entity-harness`  
当前 skill：`t2blendercodeharness-v5-executable-director`  
依据计划：`E:\2026-08-27-harness-quality-remediation.md`

## 1. 当前边界

本次工作优化的是 T2Blendercodeharness 的执行表达力、可观测性和评估边界，
不是修改数据集标签、评分公式或生成视频后用规则刷分。

用户已经暂停“每个 case 必须由 per-case Codex exec 生成 `blender_job.py`”这条要求，
因此当前仍使用项目内的 `blender/real_proxy_job.py` 编译真实 Blender CLI job；没有偷偷
回退到旧的 direct 模板，也没有把失败转换成成功。

## 2. 最新 Harness 架构

```text
exact prompt
    │
    ▼
DirectorAgent
    ├─ PromptInterpreter：实体、证据 span、隐含意图、uncertainty
    ├─ EventScheduler：事件顺序、依赖图、并发组、interaction lifecycle
    ├─ MultiEntityTrajectoryComposer：人物/物体轨迹、owner、lane、attachment
    └─ MultiTargetCameraChoreographer：目标集合、follow/orbit/dolly、coverage
    │
    ▼
DirectorPlan + compatibility SceneContract/TrajectoryPlan
    │
    ▼
real_proxy_job compiler
    ├─ detailed parametric mesh + per-kind Principled material
    ├─ ground plane + key/fill/rim 三点布光
    ├─ per-character armature、nearest-bone skinning、hand IK、walk cycle
    ├─ Child Of influence：attach → transfer → detach/support
    ├─ Camera DSL：orbit 弧线、follow、dolly、hold、TrackTo
    └─ runtime telemetry：camera finding、constraint curve、penetration finding
    │
    ▼
真实 Blender CLI (`D:\blender\blender.exe`)
    ├─ 每 case 隔离目录
    ├─ 失败最多重试 2 次
    └─ PNG animation → host MP4
    │
    ▼
artifact gate
    │
    ├─────────────── deterministic / Director / interaction
    │                 结构、事件、轨迹、相机、身份、约束、telemetry
    │
    ├─────────────── frame_statistics_only-v1
    │                 只检查 PNG 可读性、黑帧、静帧和低层统计；语义字段=None
    │
    └─────────────── independent visual review
                      lowercase `gpt-5.6-luna` / `gpt-5.6-terra`
                      或明确、可审计的人工 review payload
    │
    ▼
outer-loop MetaHarnessOptimizer
    ├─ 重复 train failure 聚合
    ├─ PatchProposal 预测 fix/regression
    ├─ attribution 先于 root-cause distillation
    ├─ refuted → 文件级 rollback
    └─ paired train/dev gate → 只接受一个 Harness owner 的 patch
```

## 3. Evaluator 逻辑和分数边界

### 3.1 顺序

1. `RealArtifactGate` 先检查 manifest、plan、job source、`.blend`、MP4、telemetry、
   frame index 和至少 3 张可读 PNG。
2. artifact 不完整或 Blender 失败时，case hard fail，不进入视觉评分和训练 preference。
3. deterministic evaluator 检查实体/类型/timebase、事件顺序、事件覆盖、轨迹连续性、
   support-before-grasp、attachment lifecycle、camera active 和 telemetry 一致性。
4. Director/interaction evaluator 独立检查 evidence、依赖、身份、handoff window、
   giver/receiver、Child Of constraint、最终 owner/support 和相机 finding。
5. frame statistics 只作为 artifact-health 辅助。它不能回答“prompt 是否实现”、
   “动作是否自然”或“镜头是否真的看见事件”。
6. 只有 artifact-complete 且 deterministic-pass 的真实视频才调用独立 VLM；VLM 的
   transport/schema 失败保持 `unavailable`，不替换为 0、100 或 plan-derived 分数。

### 3.2 task 分数

有效视觉 review 返回以下 14 个维度中的 task 维度：

```text
semantic = GM(prompt_compliance,
               physical_plausibility,
               object_trajectory,
               event_timing)

choreography = GM(camera_coverage,
                  camera_innovation,
                  character_trajectory,
                  temporal_smoothness)

task_vlm = 0.45 × semantic
         + 0.45 × choreography
         + 0.10 × visual_clarity

task_final = 0.20 × deterministic_score
           + 0.80 × task_vlm
```

几何平均 `GM` 用来避免单个维度极低时被其他维度线性平均掩盖。`deterministic_score`
是结构/硬门信号，不代表主要视频质量；因此当前版本把真实视觉 review 作为 task 主体。
deterministic hard gate 失败时，视觉高分不能救活该 case。

### 3.3 realism 分数

realism 与 task 分开记录，不相加：

```text
reviewed_realism = 0.15 × geometry_score
                + 0.15 × frame_evidence_score
                + 0.70 × GM(appearance_detail,
                             physical_realism,
                             spatial_consistency,
                             motion_naturalness,
                             visual_presentation)
```

没有独立视觉 review 时，只能保留单独的 capped `artifact_only_proxy`，并标记
`realism_claim=not_established`；geometry 100 只表示结构检查通过，不能解释成真实感 100。

## 4. 本轮实际修复块与测试

| 块 | 实际修改 | 验证 |
|---|---|---|
| VLM adapter | `evaluator/vlm_providers.py`；OpenAI-compatible Chat Completions，多帧 data URL，lowercase model ID，schema/transport fail-closed | real endpoint smoke：`complete`，3 帧；`tests/test_vlm_provider_dispatch.py` |
| artifact-only boundary | `evaluator/visual_evidence.py`、`evaluator/realism.py`、`scripts/evaluate_real_videos.py`；低层帧统计不再产生语义分数 | `tests/test_frame_statistics_boundary.py`、`tests/test_frame_statistics_patch_gate.py` |
| proposal attribution | `PatchProposal` 预测字段、`src/videoact/patch_attribution.py`、outer-loop 记录和 rollback 信号 | `tests/test_patch_attribution.py`、`tests/test_outer_loop_attribution.py`、`tests/test_patch_proposal_predictions.py` |
| rollout signal | rollout fingerprint、unique seed、mean/std/pass-rate；renderer 接受 rollout seed | `tests/test_multi_rollout_signal.py` |
| dynamic Director parsing | `director_prompt.py` 允许非固定颜色/名称的实体短语；LLM structured interpreter 要求 quote evidence span | `tests/test_director_prompt_llm.py` 及 Director 回归测试 |
| detailed render quality | per-kind Principled 材质、ground plane、三点布光 | `tests/test_render_quality_settings.py` |
| character rig | 每个 character 独立 armature；解剖骨骼层级；nearest-bone vertex groups；左右手 IK；walk cycle | `blender/character_rig.py`、`tests/test_character_rig.py` |
| attachment execution | carry/handoff 不再写竞争性的 prop location curve；使用 Child Of influence；support detach 约束；telemetry 记录曲线 | `tests/test_director_trajectory.py`、`tests/test_interaction_evaluator.py`、`tests/test_multi_entity_blender_job.py` |
| camera execution | 参数化 orbit 弧线、至少 8 个采样点、follow/dolly、TrackTo、包围盒目标、ray occlusion/continuity findings | `blender/camera_dsl.py`、`tests/test_camera_dsl_execution.py`、real smoke telemetry |

每个代码块都遵循“先加失败测试、再实现、再 focused test”的顺序。全量回归曾出现 2
个旧兼容失败，已修复后 focused 回归为 `13 passed`；在最后一次全量回归前又加入了
IK/penetration/曲线契约，需在交付前再次运行全量测试。

## 5. 真实 Blender smoke 证据

最终 v5 smoke 使用：

- case：`multi-train-001`
- Blender：5.1.2，路径 `D:\blender\blender.exe`
- 分辨率：256×256，24 fps，480 animation frames
- CLI：return code 0，retry count 0
- MP4：20 秒，可播放
- `.blend`：2 个独立 armature、4 个 hand IK constraint、actor mesh ARMATURE modifier
- telemetry：真实记录 `Attach__actor_a` 在 161 帧降为 0、`Attach__actor_b` 在 161 帧升为 1、
  support constraint 在 480 帧升为 1；`attachment_penetration=[]`
- camera：真实发现 handoff shot 的 `camera_occlusion_exceeded`，未静默删除
- VLM：`gpt-5.6-luna` 通过完整 `evaluate_vlm_run` 入口返回 `status=scored`、confidence `0.96`；
  task score `29.9295`，独立 realism score `31.55`。审阅明确指出没有可见的 carry、handoff、
  placement 和清晰事件覆盖，说明低分来自真实帧证据而非 plan 分数。

关键文件：

- [真实 proxy 视频](../out/phase3-real-smoke-v5/multi-train-001/proxy.mp4)
- [telemetry](../out/phase3-real-smoke-v5/multi-train-001/telemetry.json)
- [render report](../out/phase3-real-smoke-v5/cli_render_report.json)
- [handoff sample frame](../out/phase3-real-smoke-v5/multi-train-001/frames/animation/frame_0161.png)
- [VLM report](../out/phase3-real-smoke-v5/multi-train-001/vlm_report.json)

这份 smoke 证明真实执行层、可观测性和 VLM 入口均成立，但不是完整训练结果，也不构成
evaluator calibration；采样帧中暴露的相机遮挡和动作不可见应作为后续 Harness 训练候选问题。

## 6. 训练准入与下一步

当前不能声称“已完成六轮/五轮 Harness 训练”：质量整改计划要求先完成 evaluator calibration，
而仓库中的 golden review 仍是 placeholder/不足够的人工标注。虽然 ai-pixel 的真实多帧
VLM adapter 已成功 smoke，但仍需用至少 10 个跨难度样本建立人工 golden set，报告：

- 各视觉维度的相关性/一致性；
- task 与 realism 的独立校准；
- 主 failure owner attribution accuracy；
- VLM unavailable 比例和 schema 失败比例。

校准通过后再开始外循环训练：固定 dataset/evaluator/backend/render settings，每轮最多 5 次，
每次 10 train + 10 paired dev，每轮末尾跑累计 train + 全 60 dev；只允许一个 Harness owner
的 runtime patch，接受条件为 paired train 严格提升、paired/all-dev 不下降、hard failure 不增加、
artifact completion 不下降。旧训练 memory 保留在
`docs/t2blendercodeharness-multi-training-memory-v1.md`，本次 smoke 另作为质量整改证据追加，
不冒充训练轮次。
