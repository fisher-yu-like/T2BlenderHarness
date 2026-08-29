---
name: t2blendercodeharness-zh
description: 使用中文运行或进化 T2Blendercodeharness，将复杂或含蓄 prompt 通过 DirectorAgent 转换为可审计的 Blender 场景、事件、人物/物体轨迹和摄像机调度，执行真实 MCP/CLI 视频、artifact gate、deterministic/独立视觉评测，并按 train/dev 证据进行单组件更新。
---

# T2Blendercodeharness 中文版

## 人工标注与数据集检查

标注前端由 `scripts/golden_review_app.py` 提供，页面显示原始英文 VBench
prompt、中文辅助翻译、匿名真实 MP4 和 14 个视觉维度。生产评审包是
`dataset/golden-review-exact-v2`：30 个 case、90 个视频，必须由两个不同 ID 的
评分者独立完成 180 条记录。完成后运行
`scripts/finalize_golden_review.py` 计算 ICC(2,1)，再运行严格 validator。
训练输入仍然只能是 `dataset/vbench2-agent-training-index-v1`，不能把历史
的 `vbench-derived-100-v1` 当作训练集；prompt/source SHA、split 和零重叠
检查未通过时必须停止。

这是 contract-first 的 Text-to-Blender Harness 操作规范。保持 Codex Host、MetaHarnessOptimizer、DesignHarness、Dataset、Evaluator 边界清晰。

当前参考实现：`t2blendercodeharness-v5-executable-director`（本分支工作树，尚未提交）。历史训练基线仍保留为 `h-t2-hard-v4-pretraining-baseline`，commit `7fe017a`，仅用于历史对照。

Director 必须先建立 evidence-backed event order，再应用 generic carry 或 handoff 匹配。必须保留 `then`、`after`、`while`、reveal、subjectless clause、pause 和 return 的顺序；generic carry 不能抹掉已经识别出的 reveal、handoff 或 return。编译 Blender 前至少用一个 subjectless handoff 和一个 reveal-to-return variation 做 forward-test；如果 prompt 含 reveal 但 plan 没有显式 reveal event，必须产生 hard Director finding 并在 Blender 编译前停止。

## 1. 架构和唯一入口

```text
Prompt
 → DirectorAgent
    ├─ prompt/entity interpretation + evidence
    ├─ event graph + timing + interaction lifecycle
    ├─ character/object trajectories
    └─ multi-target camera choreography + visibility
 → SceneContract / TrajectoryPlan compatibility projection
 → Blender code job
 → Blender MCP / CLI real render
 → artifact gate
 → deterministic + independent visual review
 → train failure aggregation
 → one-owner Harness patch
 → train/dev acceptance
```

`DirectorAgent` 是唯一外部 planning 入口；不能再把旧的 SceneContract/TrajectoryPlan parser 作为另一条主路径。Compatibility projection 只负责兼容已有 executor/evaluator。

必须保存 exact prompt、evidence span、assumption、uncertainty、`director_plan_hash`、事件依赖、交互生命周期和摄像机可见性谓词。

## 2. 规划约束

拒绝空 prompt、未知引用、非法时间、矛盾顺序、依赖环、未解决 hard uncertainty、轨迹碰撞和无法观测的 camera target。

每个 actor/prop 都必须有连续轨迹；handoff 必须有 giver、receiver、transfer window、detach/attach 和 final owner。每个 required event 都必须被 camera shot 覆盖；handoff shot 必须同时看到 giver、receiver 和 prop；并发轨迹必须覆盖所有 active lanes。

典型生命周期：

```text
walk → reach → grasp → lift → carry → place → release
```

含蓄 prompt 的补全必须带 evidence；无依据的补全只能是 soft assumption，hard uncertainty 必须在编译前停止。

## 3. 执行和 artifact gate

优先使用项目的受控 adapter/MCP，失败后按策略使用 Blender CLI。真实视频使用 `D:\blender\blender.exe`；渲染失败最多重试 2 次，每个 case 隔离目录，最多并发 12 workers。

完整 artifact 必须包含：

```text
run_manifest.json
director_plan.json / scene_contract.json
trajectory.json
camera_plan.json
blender_job.py
proxy.blend
proxy.mp4
telemetry.json
frames/index.json + 至少 3 张可读 PNG
```

只有 telemetry、plan 或 PNG 而没有可播放 MP4 时必须 hard fail；没有完整 artifact 的 case 不进入视觉评分、训练记录或 patch acceptance。

## 4. Evaluator 和评分

Deterministic/Director/interaction evaluator 检查 artifact、entity/kind/timebase、事件顺序、轨迹连续性、attach/handoff/final owner、camera coverage、active camera、telemetry identity 和 sampled frames。它是硬门和结构分，不代表主要视频质量。

合规视觉 review 后，task 与 realism 分开：

```text
semantic = GM(prompt_compliance, physical_plausibility,
               object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_score = .45 × semantic + .45 × choreography + .10 × visual_clarity
task_final = task_score

reviewed_realism = GM(appearance_detail, physical_realism,
                       spatial_consistency, motion_naturalness,
                       visual_presentation)
realism_final = .15 × geometry + .15 × frame_evidence + .70 × reviewed_realism
```

realism 不加入 task。几何 100 只表示结构 gate 通过，不等于真实感 100。外部模型只能使用小写 `gpt-5.6-luna` 或 `gpt-5.6-terra`；endpoint 或 schema 不可用时使用可审计的 `assistant_local_review`，仍不可用则记录 `unavailable`，不得填 0 或伪造分数。

## 10. 当前 Agent Codegen 与六轮协议

生产代码链为 `DirectorAgent → BlenderCodeAgent → D:\blender\blender.exe`。
`BlenderCodeAgent` 读取每个 case 的 DirectorPlan 和 `blender/lib` 签名，按
case 生成并冻结 `blender_job.py`；`generate-once-freeze` 使用
`plan_hash + harness_version` 复用同一份源码。`CodexExecProvider` 只是结构化
Codex Host 传输边界，不把 endpoint/token 写入 skill。

`template_baseline` 只能作为显式历史对照臂，绝不是 agent 或 L4 的 fallback。
Director/codegen/schema/coverage 失败必须 fail-closed，不产生 agent 视频；Blender
进程失败最多对同一冻结源码重试 2 次，源码发生变化立即失败。

训练前的动态源审计不只比较 SHA。每个 case 的 `blender_job.py` 必须真的包含
`CASE_SCENE_PROFILE` 和 `codex-local-case-profile-v2`，并且 profile、DirectorPlan
和实体/镜头轨迹都来自该 case；只有给通用脚手架换一个 case id 的源码会被标记为
`missing_case_specific_generation_profile` 并在渲染前 fail-closed。这样可以确认
没有偷偷使用固定模板。profile 的 `case_signature` 还必须绑定 DirectorPlan hash
前缀，job 中的 code hash 必须与磁盘源码一致；共享运行时库仅作为执行基础。

另外，DirectorAgent 不得因为 prompt 里出现物体就自动补一个人。水滴、气球、
液体、表面、容器等 object-only prompt 的 DirectorPlan 和运行时 telemetry 中不得
出现没有 prompt 证据的 actor；可以保留明确标为 staging assumption 的中性支撑面。
这属于 Harness 语义门禁，不能用 deterministic 通过或 artifact 分数掩盖。

当前训练数据为 VBench-2.0 原始 prompt 索引
`dataset/vbench2-agent-training-index-v1`：60 train、60 dev、20 frozen test，
六个 train 家族和六个 dev 家族；每条 `prompt` 与
`data/vbench-source/VBench2_full_info.json` 的 `prompt_en` 完全一致，不添加本地
事件、实体、proxy 或 oracle 标签。DirectorAgent 在运行时从原始 prompt 生成
plan。历史 `dataset/trajectory-v5-agent-codegen`、`vbench-derived-100-v1` 和
`trajectory-v4-multi` 仅用于对照/回归，训练入口会拒绝它们。每轮最多 5 次外循环
尝试，每次新增 10 train + 10 paired dev；每轮结束另跑完整 60 train + 60 dev，共
120 个视频。上限为
`6 × (5 × 20 + 120) = 1320`，没有新重复失败时允许提前停止。

每次 split 和每轮结束都要更新本地 Markdown/JSONL Memory 表，至少保留轮数、
prompt、绝对 proxy 视频地址或 `NOT_RENDERED: reason`、真实 Director/task/realism
分数、review 来源、Harness 问题、修复位置/方法、前后 delta 和自然语言处理结论。
只允许一个 Harness owner；重复失败至少覆盖两个 train case 才能提出 patch，dev
不下降、artifact 不下降、test 不参与 patch 选择。

每个 finding 必须含 `failure_id`、`owner`、`category`、`severity`、`evidence` 和 `repair_route`。

训练前必须运行 `scripts/validate_benchmark_prompt_index.py`。它校验原始 source
SHA、逐条 prompt provenance、split/fingerprint、source identity 唯一性和参考集
重叠；readiness 与 `train_real_harness.py` 都是 fail-closed。benchmark provenance
通过只说明输入合法，不说明 agent、Blender 或 evaluator 已通过。训练 CLI
在真正准备/渲染 case 前还会再次要求 readiness 报告中的
`training_allowed=true`；缺失、blocked 或旧报告都不能启动六轮。

## 5. Harness 外循环进化

```text
固定 prompt/dataset/evaluator/backend
 → 真实渲染 train/dev
 → 聚合重复 train failure
 → 只选一个 owner
 → 修改一个 Harness 组件
 → 重跑相同 train + paired dev
 → train 严格提升且 dev 不下降才接受
 → test 仅最后盲测
```

同一 normalized failure 必须影响至少 2 个不同 train case，才能产生 proposal。owner 可为 `director_prompt_interpreter`、`director_event_scheduler`、`director_trajectory`、`director_camera`、`scene_parser`、`trajectory_planner`、`camera_planner`、`blender_code_agent`、`blender_executor`、`proxy_renderer`、`evaluator` 或 `meta_harness`。

禁止同时修多个 owner，禁止修改 dataset label/evaluator 规则来换分，禁止用 test 选择 patch，禁止把视频失败误报为 plan 成功。

## 6. Skill 自进化

Skill 自进化是“根据真实历史证据更新操作规范”，不是运行时偷偷改写自身。先保存 skill/dataset/evaluator/Harness fingerprint，执行 capability check 和全量测试；再把历史 `patch_manifest.json` 和新 train batch 转为 JSONL：

```powershell
uv run python skills/t2blendercodeharness/scripts/build_self_evolution_records.py `
  --round-root out/training/multi-five-rounds-v1 `
  --out out/skill-self-evolution-v1/historical_records.jsonl
uv run python skills/t2blendercodeharness/scripts/propose_skill_update.py `
  --records out/skill-self-evolution-v1/historical_records.jsonl `
  --out out/skill-self-evolution-v1/proposal.json
```

proposal 必须是 proposal-only、标明旧 skill hash、证据路径、affected cases、单一目标章节和回归命令；明确人工批准后才可修改最小章节。修改后必须跑 capability check、全量测试、原数据集 forward-test 和 prompt variation，并确认 artifact gate、Director evidence、one-owner、VLM unavailable、retry、train/dev/test 隔离都未变弱。

## 7. Memory 和交付

每个 case 和每次 patch 追加到 Markdown/JSONL Memory，至少记录轮数、attempt、split、prompt、绝对 proxy 视频地址、plan hash、Director/task/realism 分数、review source/confidence、Harness 问题、修复位置、前后 delta 和自然语言处理结论。缺视频写 `NOT_RENDERED: reason`。

最终必须交付逐样本表格、proposal、skill pre/post hash、artifact/render/evaluator 审计、train/dev 曲线和 test 盲测结论。当前 VBench-derived-100 对比结果保存在 `docs/vbench-100-harness-comparison-experiment.md`。

## 8. 停止条件

contract/Director/trajectory/camera validation 失败、CLI/MCP 或重试后仍失败、artifact 不完整、deterministic hard fail、proposal 多 owner、无重复 train failure、dev regression、test 泄漏或 skill 回归测试失败，都必须停止晋级并报告。

## 9. Evaluator v5 边界与可执行自进化

`frame_statistics_only-v1` 只能回答 artifact health（`artifact_health`）和低层帧观测，不能
回答 prompt compliance、event timing、physical plausibility、
character/object trajectory、physical realism 或 motion naturalness；这些
字段必须显式为 `None`，整体 `score` 也必须为 `None`。`review_source` 为
`frame_statistics` 的记录不得进入 realism 融合或 patch acceptance gate。
语义与视频质量必须来自真实 VLM 或经过校验的人工 review；模型 ID 统一使用
小写 `gpt-5.6-luna`、`gpt-5.6-terra`，网络/策略/schema 失败保持
`unavailable`，不改成 0。

每个 proposal 必须包含 `predicted_fixes`、`predicted_regressions`、
`prediction_rationale`。attribution 要在 root-cause 蒸馏前执行；发现未预测的
负 delta 时，产生 `refuted` 并按文件粒度 rollback。仍然必须满足单 owner、
同一失败影响至少两个不同 train case。

Skill 自进化优先沉淀可执行的 `function_library`、`owner_mapping` 和追加式
`memory_entry`，纯 `prose_guidance` 修改不算训练收益，除非另有 runtime
证据。orbit 的弧线、`continuity_group`/遮挡、handoff 约束和穿模检查都必须
在执行层产生可审计 finding，不能只写在说明里。
## 11. 当前本地 Codex 执行政策

正式训练统一使用进程内 `codex-local`：DirectorAgent 生成 DirectorPlan，
BlenderCodeAgent 按当前 case 生成 `blender_job.py`，无 external endpoint。
`CodexExecProvider` 仅保留为显式诊断适配器，不是训练路径。

每个 case 的内循环最多三次（`max_inner_attempts=3`）。plan 不合规、代码
coverage 失败、真实 Blender 渲染失败或 artifact 不完整时，重新生成完整
plan/code/candidate；不在场景内打补丁、不偷偷切换模板。三次都失败就写入
`NOT_RENDERED`，并保留全部失败证据。外循环仍最多五次，每次只修改一个
Harness owner；内循环是执行恢复，不是 Harness 自进化。
