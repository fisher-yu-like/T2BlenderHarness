# T2BlenderCodeHarness

面向文本到 Blender 视频生成的真实 Harness、严格视频评估和外循环自进化实验工程。

本项目的目标不是把所有 prompt 映射到一个固定场景，而是让每个 benchmark case 都经历同一条可审计链路：

    原始 VBench prompt
      -> DirectorAgent
      -> DirectorPlan
      -> 每 case 独立 BlenderCodeAgent
      -> D:\blender\blender.exe
      -> proxy.mp4 + proxy.blend + runtime telemetry
      -> artifact / deterministic gate
      -> 真实视频视觉审查
      -> task / realism / trajectory / camera 等分数
      -> Harness 外循环只修改一个组件

本文档以当前 director-multi-entity-harness 工作树为准，区分已验证能力、训练前门禁和正式六轮训练规则。

## 当前状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 动态 DirectorAgent | 已实现 | 正式 model 路径由外部结构化 provider 按 case 调用；Blender code 仍由本地 Codex 生成；codex-local 仅为显式 baseline |
| per-case Blender code | 已实现 | 每个 case 独立生成 blender_job.py，并记录 plan/code/artifact provenance |
| 固定模板回退 | 禁止 | provider、schema、coverage、source 或 render 失败均 fail-closed |
| 真实 Blender 渲染 | 已验证 | 使用 D:\blender\blender.exe 生成真实 MP4、BLEND、PNG 和逐帧 telemetry |
| MP4 + runtime 评估 | 已实现 | 必须解码真实 proxy.mp4，并要求 Blender runtime_observations |
| 严格 VLM 证据门槛 | 训练前固化 | 正式训练前启用逐维度证据绑定和缺证据拒绝规则，之后冻结 evaluator |
| 六轮 Harness 训练 | 未开始 | 当前 readiness 仍被 golden_review 和 paired_gate 阻塞 |
| 最近全量测试 | 以当前 CI/验证报告为准 | 不在 README 中手工固定通过数量；运行命令会生成 machine-readable 报告 |

当前 training_allowed=false 代表训练接受门禁尚未全部满足，不代表真实视频链路没有跑通。

本分支按 `D:/sy/T2BlenderHarness-improvement-plan-zh.md` 增加了可复核的
`source-fingerprint-v1`、`paired-statistics-v1`、
`physics-oracle-v2-obb-bvh-contact-ownership`、`experiment-fingerprint-v1` 和
`active-sampling-v2-replay`。正式放行采用 G0（身份/可信边界）→ G1（evaluator
冻结与人工校准）→ G2（10 train + 10 dev paired pilot）→ G3（不改 Harness
的 60 train + 60 dev shadow）→ G4（六轮训练）；
`scripts/check_formal_release_gates.py` 产生带内容哈希的 G0--G3 报告，正式
训练入口会验证该报告。

T16 的历史回放审计入口为
[`scripts/audit_active_sampling.py`](scripts/audit_active_sampling.py)，协议见
[`docs/active-sampling-replay-v2.md`](docs/active-sampling-replay-v2.md)。它只接受
train/dev 的真实历史 paired delta 和 render cost；达不到 30% render cost 节省或
95% 决策一致率时直接阻止该效率门禁，不把合成 fixture 当作正式证据。

## 1. Harness 架构

### 1.1 组件职责

#### Codex Host 与 Skills

Codex Host 固定数据集、训练轮次、渲染资源、失败状态、审计字段和 append-only memory。

主要入口：

- skills/t2blendercodeharness/SKILL.md：Harness 执行边界。
- skills/t2blendercodeharness-training/SKILL.md：训练、门禁、记忆和回滚。
- scripts/prepare_real_jobs.py：准备每个 case 的 plan、source 和 coverage。
- scripts/train_real_harness.py：六轮外循环和 aggregate。
- scripts/evaluate_real_videos.py：真实视频视觉审查入口。

Host 不负责给视频编造语义分数；它只串联真实生成、真实评估和证据记录。

#### DirectorAgent

Scene/Prompt Parser、Character/Object Trajectory Planner 和 Camera Choreography Planner 已融合到 DirectorAgent 的生产路径。

核心代码：

- src/videoact/director.py
- src/videoact/director_prompt_llm.py
- src/videoact/codex_self_provider.py
- src/videoact/director_schedule.py
- src/videoact/director_trajectory.py
- src/videoact/director_camera.py

DirectorAgent 根据原始 prompt 生成：

- 有 prompt evidence 的 actor、prop、support/environment 实体；
- 有序事件图、参与者和目标对象；
- 人物与物体的阶段性轨迹；
- camera shot、cue、target、时间窗口和事件 coverage；
- prompt span、assumption、uncertainty、provider/policy fingerprint；
- director_plan_hash。

例如：

    One person passes a ball to another.

必须生成 actor_a、actor_b 和 prop_01_ball，并让 handoff 的 participant_ids 为 actor_a、actor_b。不能只生成一个人物再把另一个人物的动作忽略。

对于水滴沿铝片滑动、气球移动等 object-only prompt，DirectorAgent 不得凭空添加人物。

#### DirectorPlan Gate

Blender 编译前检查：

- schema 和 JSON 是否完整；
- 实体 ID、事件 ID、参与者和目标是否存在；
- evidence span 是否越界；
- 事件顺序和时间窗是否可执行；
- 每个必需实体是否拥有轨迹；
- 每个 camera cue 是否有对应 shot 和 coverage obligation；
- 是否有 unresolved hard uncertainty；
- 是否有循环依赖、所有权矛盾或 handoff 生命周期缺失。

任一 hard failure 都写入 NOT_RENDERED 或 preparation failure，不生成模板视频。

#### BlenderCodeAgent

BlenderCodeAgent 的输入是当前 case 的 DirectorPlan、已验证的 blender/lib 能力签名、约束和 Harness 版本，输出一次 case-specific blender_job.py。

核心代码：

- src/videoact/blender_code_agent.py
- src/videoact/codex_self_provider.py
- blender/lib/
- scripts/prepare_real_jobs.py

代码 gate 包括：

1. provider/schema 响应检查；
2. Python AST 安全检查；
3. 禁止 eval、exec、越权进程和旧模板引用；
4. 必须绑定当前 DirectorPlan；
5. 必须调用 verified library；
6. 必须输出 telemetry、frame index、BLEND 和 animation render；
7. 必须覆盖当前 case 的实体、事件、轨迹和 camera cue；
8. 检查 plan_hash -> code_hash -> artifact_hash；
9. 检查不同 case 是否错误复用同一 source。

template_baseline 只能作为显式对照实验，永远不是 agent fallback，也不能计入 Harness 成功率。

#### Blender Executor

每个 case 使用隔离目录和真实 Blender：

    D:\blender\blender.exe

合格 artifact 至少包括：

- proxy.mp4；
- proxy.blend；
- blender_job.py；
- director_plan.json；
- scene_contract.json；
- trajectory.json；
- camera_plan.json；
- telemetry.json；
- frames/index.json 和可读 PNG sample frames；
- render attempts、manifest 和 hash provenance。

当前安全策略是每组最多 12 个 case，默认 4 个 Blender workers。上一组渲染时可以评估已经完成的上一组，但同一个 artifact 不启动多个 evaluator。

#### Inner loop

Inner loop 只做执行恢复：

- DirectorPlan 不合规、代码生成失败、coverage 失败或 Blender render 失败时重新生成 fresh Director + Blender code candidate；
- 每个 case 最多 3 次；
- 不在场景内部反复修补；
- 不修改 evaluator、数据集或既有 Harness owner；
- 三次仍失败就写 NOT_RENDERED: reason，并保留全部尝试证据。

#### Outer loop

Outer loop 才修改 Harness：

- 从多个 train case 聚合重复失败；
- 同一 root cause 至少在两个不同 train case 出现后才提出 patch；
- 一次只修改一个 Harness owner；
- 重跑相同 paired train/dev；
- train 严格提升且 dev 不下降才接受；
- 拒绝的 patch 只回滚该 owner，不能删除失败记忆。

### 1.2 实际数据流

    exact prompt
      -> 外部结构化 DirectorAgent (provider_mode=model)
      -> DirectorPlan / evidence / event graph
      -> plan schema + coverage gate
      -> per-case BlenderCodeAgent
      -> frozen blender_job.py
      -> source/hash audit
      -> D:\blender\blender.exe
      -> proxy.mp4 + runtime observations
      -> artifact + deterministic + independent oracle
      -> MP4 decode + runtime trajectory review
      -> strict visual review
      -> separate scores
      -> append-only memory
      -> one-owner Harness patch

### 1.3 Prompt 到真实 proxy 的训练链路：Python 与大模型的边界

训练时不是“Python 模板生成视频”，也不是让大模型直接绕过验证器写文件。两者的职责固定如下：

| 阶段 | Python 固定部分 | 大模型动态部分 | 产物 |
|---|---|---|---|
| 选择 case | 读取原始 VBench manifest、固定 train/dev 批次、保留 prompt 原文 | 不修改 prompt | case manifest |
| Director 解释 | Pydantic contract、事件调度、轨迹/相机 composer、证据和 fail-closed 校验 | 外部 OpenAI-compatible structured provider 根据完整 prompt 解释实体、事件顺序、参与者、物体所有权和 camera cue | DirectorPlan、trajectory、camera plan |
| 计划门禁 | 检查 schema、ID、时间、coverage、hard uncertainty 和 evidence span | 不得绕过 gate | pass 或 NOT_RENDERED |
| Blender code 生成 | 构造 codegen payload、提供 verified `blender/lib` 签名、AST/运行时/coverage 校验 | 本地 `CodexExecProvider` 根据当前 case 的 DirectorPlan 生成 case-specific `blender_job.py` | 冻结 source、code_hash |
| 真实执行 | Python Host 隔离目录、冻结 source、启动 Blender、收集日志和重试状态 | 不在运行中偷偷改场景 | `proxy.blend`、`proxy.mp4`、PNG、telemetry |
| 画面和物理证据 | 解码 MP4、读取逐帧 transform/bbox/camera/rig、运行 deterministic evaluator | local Codex visual review 或显式 VLM 审查真实帧 | 14 维 review、四个真实通道 |
| 外循环进化 | 聚合 train/dev、执行接受门禁、写 memory、回滚和画曲线 | 根据重复失败提出一个 Harness owner 的候选修复 | patch manifest、新 Harness 版本 |

实际执行顺序是：

1. `scripts/train_real_harness.py` 读取 `dataset/vbench2-agent-training-index-v1`，只取原始 prompt，不生成新 prompt。
2. 正式 `provider_mode=model` 下，Python `DirectorAgent` 通过外部结构化 provider 发起当前 case 的 Director call。外部大模型只负责语义解释；Python 负责把返回值转成 typed DirectorPlan，并验证实体、事件、时间和 coverage。外部 endpoint、模型和 call hash 会写入 Director provenance。
3. 通过计划 gate 后，Python `BlenderCodeAgent` 把当前 plan 和 verified library 能力传给本地 `CodexExecProvider` codegen call。本地 Codex 只生成当前 case 的 Blender source；Python 负责 AST、安全、运行时标记、case coverage 和 hash 校验。两个阶段不能共享 provider 记录或互相替代。
4. Python Host 冻结 `blender_job.py`，再调用 `D:\blender\blender.exe`。这个 job 是已经由大模型针对该 case 生成的 Python，不是预先写好的场景模板。
5. Blender 内部执行该 job，生成真实几何、人物 rig、动作 keyframes、物体轨迹、摄像机调度、`proxy.blend` 和动画渲染结果。Host 从真实 MP4 解码帧，并读取 Blender 写出的 runtime observations。
6. evaluator 先运行 artifact/deterministic gate，再让符合配置的 blind `gpt-5.6-luna`/`gpt-5.6-terra` 或显式本地 Codex 审查真实时间序列帧。缺帧、缺 telemetry 或缺逐维证据不会补成分数；本地路径无 external endpoint 且只能作诊断。
7. Python 外循环只根据重复失败决定是否提出 patch；每次 patch 只能修改一个 Harness owner，随后重新生成受影响 case 的 plan 和 Blender code，并用 paired dev 及完整 dev 验证。

在 Codex app 内还支持一个明确隔离的 `assistant_diagnostic` 路径：本次
`vbench2-dev-06-15` 没有调用 `codex exec`，而是由当前 Codex 会话直接提供
case-specific Director interpretation 和 Blender source，再交给同一个
`BlenderCodeAgent` 静态 gate、`D:\blender\blender.exe`、trusted observer 和
evaluator。该路径记录 `provider_kind=assistant_generated`，禁止模板/fallback，
只用于验证“当前 Codex 生成 → 真实 Blender → 真实观测”的链路；它不是一个可由
独立脚本自行调用的模型 provider，也不能满足正式 `provider_mode=model`、G0 或六轮
训练准入。真实运行记录见
`out/preflight/assistant-full-chain-vbench-06-15-v5/assistant_diagnostic_provenance.json`。

因此，Python 文件负责“可重复、可审计、不能被绕过的基础设施”；大模型负责“不能靠固定关键词可靠完成的语义解释、case-specific code 和视觉判断”。任何一侧失败都 fail-closed，不会偷偷换回本地模板。

## 2. Harness 的创新点

### 2.1 动态 Director，而不是固定关键词模板

原始 prompt 不先压缩成固定动作关键词。DirectorAgent 需要解释实体、事件顺序、参与者关系、物体所有权和 camera cue，并显式保存不确定性。

如果 prompt 信息不足以形成可靠的可执行计划，系统拒绝继续，而不是用通用厨房、桌子、人物或球体填充。

### 2.2 轨迹与摄像机是可执行义务

人物、物体和摄像机不是渲染后补出的 metadata，而是 DirectorPlan 的一部分：

    事件顺序
      -> 人物轨迹
      -> 物体轨迹 / 所有权
      -> camera target / shot / cue
      -> Blender keyframes
      -> runtime observations

因此可以比较计划轨迹和 Blender 实际轨迹，发现人物未靠近、物体未交接、相机未覆盖或事件时序错误。

### 2.3 每个 case 独立生成 Blender source

共享的是 verified library 的执行原语，不是共享场景模板。生成 source 必须绑定当前 prompt、DirectorPlan 和 case profile。重复 source、缺少 case 语义或隐式 template fallback 都会被拒绝。

### 2.4 真实 MP4 和逐帧运行时闭环

评估同时使用：

- 真实 proxy.mp4 的解码帧；
- Blender 每帧的 root transform；
- world-space bbox；
- screen-space bbox；
- camera transform；
- actor pose points；
- connected-rig 状态。

这使系统能分别判断“计划写了什么”和“视频实际呈现了什么”。

### 2.5 task 与 realism 分开

一个视频可以完成事件但外观粗糙，也可以画面漂亮但没有完成事件。两种情况不能互相掩盖：

- task：prompt、事件、轨迹和摄像机调度；
- realism：外观、材质、物理可信度、空间稳定性、动作自然度和 presentation；
- visual、physical、trajectory、camera：额外的真实视频通道。

### 2.6 Fail-closed 和可审计自进化

存在 plan 不等于完成动作，MP4 可播放不等于物理合理，有人物模型不等于人物完成了交互。证据不足时不猜分、不补分、不切模板。

每次结果都可以沿以下链路追溯：

    prompt_hash
      -> director_plan_hash
      -> code_hash
      -> artifact_hash
      -> evaluator evidence
      -> patch owner
      -> train/dev delta

## 3. Evaluator 逻辑

### 3.1 Artifact / deterministic gate

主要代码：

- evaluator/deterministic.py
- evaluator/real_artifacts.py
- evaluator/independent_oracle.py
- evaluator/geometry_realism.py
- evaluator/physics_metrics.py
- evaluator/interaction_metrics.py

第一层回答“证据是否真实存在、是否完整、是否违反硬约束”，不替代主要视觉质量分数。

检查项目包括：

- MP4 是否存在并可解码；
- BLEND、PNG、telemetry、frame index 是否完整；
- frame count、FPS、duration 和 resolution 是否一致；
- plan、scene contract、trajectory、camera 和 telemetry 是否一致；
- 实体是否存在、类型是否正确；
- world-space bbox 是否发生不允许的重叠或穿模；
- 物体是否低于地面；
- 人物 rig 是否断裂；
- handoff、attach、detach、support 是否满足声明；
- plan/code/artifact hash 是否匹配；
- 是否出现模板复用；
- camera target 和 event coverage 是否存在。

hard failure 时不进入视觉分数，也不能据此接受 Harness patch。

### 3.2 真实视频证据

代码：evaluator/real_video_metrics.py

真实视觉/运行时评估要求：

    proxy.mp4 能够解码
    AND
    telemetry.runtime_observations 存在并覆盖渲染帧

只存在 plan、PNG、BLEND 或非空 MP4 都不能生成视觉、物理或轨迹分数。

当前四个追踪通道：

    visual_score =
        mean(visual_clarity, appearance_detail, visual_presentation)

    physical_score =
        实际 bbox 碰撞、地面、rig 连接和运动平滑证据

    trajectory_score =
        mean(object_trajectory,
             character_trajectory,
             event_timing,
             temporal_smoothness)

    camera_score =
        mean(camera_coverage, camera_innovation)

当前不再用 95/90 的人工封顶掩盖不确定性。分数、confidence 和 evidence completeness 分离；低证据进入 `needs_human_review`，而不是机械压分。

### 3.3 14 个视觉维度

任务与调度：

1. prompt_compliance：prompt 的实体、动作和事件是否真的出现；
2. physical_plausibility：是否悬空、穿模、失去支撑或违反基本物理；
3. camera_coverage：关键人物、物体和交互是否持续可见；
4. camera_innovation：follow、orbit、dolly、zoom、pan、reveal 等 cue 是否执行；
5. character_trajectory：人物路径、姿态和交互是否连续；
6. object_trajectory：物体移动、交接、放置和最终归属是否清楚；
7. event_timing：事件顺序、准备、动作、停顿和交接时机是否正确；
8. temporal_smoothness：是否跳帧、抖动或突然变位；
9. visual_clarity：实体、动作和空间关系是否易于辨认。

真实性：

10. appearance_detail：模型、材质和局部结构是否仍是粗粒度 primitive；
11. physical_realism：接触、阴影、光照、比例和重力是否可信；
12. spatial_consistency：实体尺寸、位置和场景布局是否稳定；
13. motion_naturalness：人物躯干、手部和物体运动是否自然；
14. visual_presentation：构图、曝光、清晰度和整体呈现是否合格。

### 3.4 当前 task / realism 公式

几何平均用于短板保护。一个关键维度很低时，不能被其他高分完全抵消。

    semantic_core = GM(applicable required dimensions:
                       prompt_compliance,
                       physical_plausibility,
                       object_trajectory,
                       event_timing,
                       character_trajectory only when an actor is applicable)

    choreography = GM(camera_coverage,
                      camera_innovation only when camera motion is required,
                      character_trajectory only when an actor is applicable,
                      temporal_smoothness)

    observability = GM(camera_coverage, visual_clarity)

    task_score = 0.75 * semantic_core
               + 0.25 * observability

    realism_vlm = GM(appearance_detail,
                     physical_realism,
                     spatial_consistency,
                     motion_naturalness,
                     visual_presentation)

    overall_vlm_score = 0.70 * task_score
                     + 0.30 * realism_vlm

evaluator/realism.py 还生成 reviewed realism：

    reviewed_realism = 0.15 * geometry_score
                     + 0.15 * frame_evidence_score
                     + 0.70 * realism_vlm

其中：

- realism_vlm_score 是五个真实性维度的几何平均；
- realism_score 通常指包含 geometry/frame evidence 的 reviewed realism；
- task_score 和 realism_score 不互相替代；
- overall_vlm_score 只是汇总观察值，不能隐藏单项回归。
- `camera_innovation` 只在 prompt/plan 明确要求相机运动时适用；其实现
  结果以 `camera_effectiveness` 命名，静态镜头不会因为没有运动被扣 task。
- 任一 required event 的证据分数低于 25 时，语义状态为
  `failed_required_event`，task 上限为 49；缺证据为 `uncertain`，不当作 0。

### 3.5 严格 VLM 证据门槛

这套规则在正式训练前启用，随后冻结 evaluator。

#### 证据必须绑定到视频

每个维度必须绑定真实视频的帧号或帧区间：

- 实体：指出人物/物体在哪些帧可见；
- 交互：提供交互前、交互中、交互后的证据；
- 时序：提供按时间排列的事件证据；
- 相机：提供不同时间点的构图或相机运动证据；
- 真实性：指出材质、接触、阴影、穿模、关节和运动的可见证据。

DirectorPlan、文件名和单独的 telemetry 不能作为视觉证据。telemetry 可以辅助验证真实 transform，但不能代替“画面是否看清”。

#### 严格封顶规则

| 情况 | 处理 |
|---|---|
| 关键实体缺失 | prompt_compliance 不得超过 20 |
| 关键事件未在视频确认 | event_timing 不得超过 25 |
| 事件顺序错误 | event_timing 不得超过 10 |
| camera cue 未执行 | camera_innovation 不得超过 25 |
| 轨迹只在 plan 中存在、画面不可见 | 对应轨迹维度不得超过 40 |
| 穿模、断骨骼、悬空等严重问题 | deterministic hard failure，不进入视觉主分数 |
| 只有“大致像”而没有逐维证据 | needs_human_review，不允许 patch acceptance |
| 采样帧不足以判断 | unavailable 或 needs_human_review，不伪造数字 |

“明确观察到没有完成”会给低分；“证据不足无法判断”则不填 0，而是保留为不可用/待人工复核。

#### 分数带

    0–19   缺失、相反或完全错误
    20–39  局部相似，但关键要求缺失
    40–59  意图可见，动作/时序/轨迹明显不完整
    60–74  主要要求完成，但有明显粗糙问题
    75–89  要求基本完整，仅有局部问题
    90–100 每个关键要求都有连续证据且没有明显错误

90 分以上不能因为“场景里有一个人和一个球”获得，必须同时证明实体、事件、顺序、轨迹、摄像机、物理和画面细节。

### 3.6 VLM 来源

本地 Codex 路径：

    review_source = codex_local_visual_review
    vlm_model = codex-local

该路径由当前 Codex 进程处理真实 MP4 帧和 Blender runtime evidence，不调用外部
endpoint；它是显式诊断/人工辅助路径，不满足正式的独立 judge gate，也不会把
缺失 review 改写成数字。正式 model-driven 生成仍必须记录外部结构化
Director provider 与本地 `CodexExecProvider` Blender-code 两次调用。

如果显式启用外部 OpenAI-compatible endpoint，只允许使用小写模型标识：

    gpt-5.6-luna
    gpt-5.6-terra

两种来源都必须经过同一 schema、证据门槛、置信度和拒绝规则。

## 4. 数据集

正式训练只使用原始 VBench-2.0 prompt index：

    dataset/vbench2-agent-training-index-v1

原始来源：

    data/vbench-source/VBench2_full_info.json

| split | 数量 | 用途 |
|---|---:|---|
| train | 60 | 聚合失败并提出 Harness patch |
| dev | 60 | paired holdout 和每轮完整防过拟合评估 |
| frozen test | 20 | 里程碑/最终盲测，不产生 patch |

约束：

- prompt 必须与原始 VBench prompt_en 完全一致；
- 不加入本地自造 prompt、实体标签、事件标签或 oracle 标签；
- train/dev/test 不重叠；
- 六个 train family 和六个 dev family 各十个 case；
- vbench-derived-100-v1、trajectory-v4-multi、trajectory-v5-agent-codegen 只作历史或对照，不进入正式训练。

## 5. 六轮外循环训练计划

### 5.1 训练前门禁

正式训练前必须全部通过：

1. 全量 pytest；
2. Harness capability check；
3. VBench prompt index validator；
4. frozen evaluation validator；
5. 真实 Blender agent smoke；
6. 动态 Director + BlenderCodeAgent provenance smoke；
7. 严格视觉 golden review 校准；
8. paired agent/dev gate。

任一 gate 未通过都保持 training_allowed=false。可以做诊断运行，但诊断结果不能用于接受 Harness patch。

### 5.2 每轮预算

总共 6 轮，每轮最多 5 个 outer attempt。

每次 attempt：

- 10 个新的 train case；
- 10 个对应 paired dev case；
- 20 个真实 Blender 视频；
- 完成 artifact、deterministic、runtime 和视觉评估；
- 立即更新 Markdown memory。

每轮结束：

- 用当前已接受的 Harness version；
- 完整评估 60 train + 60 dev，共 120 个真实视频；
- 记录 task、realism、visual、physical、trajectory、camera、artifact completion 和 hard failure；
- 更新 train/dev 曲线；
- 决定是否进入下一轮。

理论上限：

    每轮 = 5 * 20 + 120 = 220 个视频执行
    六轮 = 6 * 220 = 1320 个视频执行

这是上限。如果没有新的可重复失败，允许提前结束该轮。每个失败 case 的 fresh candidate 最多 3 次，candidate generation 上限为 1320 * 3 = 3960，但同一失败 source 不进行隐藏 retry。

### 5.3 一次 attempt 的步骤

1. 固定当前 Harness、dataset、evaluator 和 Blender binary；
2. 用 10 个 train 和 10 个 paired dev 生成新 DirectorPlan 和新 source；
3. 以最多 12 个 case 为一组渲染，默认 4 workers；
4. 解码真实 MP4，读取 runtime_observations；
5. 由严格视觉评估器逐维给分并绑定证据；
6. 聚合至少两个不同 train case 中重复出现的 root cause；
7. 只提出一个 Harness owner patch；
8. 对相同 paired case 重新生成 plan 和 source；
9. 通过 anti-overfit gate 才接受。

### 5.4 Patch 接受标准

全部条件必须满足：

- train task_score 严格提升，建议最小有效提升为 1.0 分；
- train realism_score、visual_score、physical_score、trajectory_score、camera_score 不下降；
- paired dev 的 task、realism 和四个真实视频通道不下降；
- 本轮结束的完整 60 train + 60 dev 不下降；
- artifact completion 不下降；
- 不新增 hard failure；
- 不出现 source/template reuse；
- visual review coverage 和证据完整度不下降；
- 一次 patch 只属于一个 owner，例如 director_prompt_interpreter、director_event_scheduler、director_trajectory、director_camera、blender_code_agent 或 blender_executor。

拒绝 patch 时只回滚 owner 代码，保留所有视频、分数、失败和 patch 证据。

### 5.5 训练记忆和曲线

每个 split 完成后立即追加：

    docs/t2blendercodeharness-agent-training-memory-v1.md

每行至少包含：

| Round | Attempt | Split | Case ID | Prompt | Proxy video | Task | Realism | Visual | Physical | Trajectory | Camera | Harness problem | Harness fix | Before -> after | Natural-language handling |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|

失败 case 也必须保留：

    NOT_RENDERED: <具体原因>

机器可读证据保存在：

    round-XX/attempt-YY/real/{train,dev}/
    round-XX/overall/real/{train,dev}/
    round-XX/attempt_report.json
    round-XX/overall_report.json
    round-XX/patch_manifest.json
    memory/harness_updates.jsonl

曲线分别绘制：

- train/dev task；
- train/dev realism；
- visual、physical、trajectory、camera；
- DirectorPlan coverage；
- artifact completion；
- hard failure rate；
- train-dev gap；
- 按 Harness owner 的 failure count。

缺失视觉 review 是缺失点，不允许插值成 0 或 100。

## 6. 已验证的真实回归样本

原始 VBench prompt：

    One person passes a ball to another.

修复前只生成 actor_a。修复后真实链路生成：

    actor_a
    actor_b
    prop_01_ball
    support_surface

Blender telemetry 覆盖 120 帧，两个人物 rig 均连接正常。证据目录：

    out/preflight/one-case-vlm-v3/vbench2-train-02-01/

本次 codex-local visual review：

| 通道 | 分数 |
|---|---:|
| visual | 78.83 |
| physical | 92.27 |
| trajectory | 60.94 |
| camera | 95.00 |
| task | 65.57 |
| realism | 59.75 |

event_timing=6.09、motion_naturalness=37.58，说明实体解析已经修复，但传球时机和人物动作自然度仍不合格。这正是严格评估需要暴露的问题：不能因为画面里同时出现两个人和一个球，就判定“传球完成”。

## 7. 常用命令

代码测试：

    uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp
    uv run python skills/t2blendercodeharness/scripts/capability_check.py --project-root .

数据集验证：

    uv run python scripts/validate_benchmark_prompt_index.py --root dataset/vbench2-agent-training-index-v1 --source data/vbench-source/VBench2_full_info.json --reference-root dataset/frozen-eval-v2 --reference-root dataset/vbench-derived-100-v1 --reference-root dataset/trajectory-v4-multi --reference-root dataset/trajectory-v5-agent-codegen

    uv run python scripts/validate_frozen_eval_set.py --root dataset/frozen-eval-v2 --reference-root dataset/vbench2-agent-training-index-v1 --reference-root dataset/vbench-derived-100-v1 --reference-root dataset/trajectory-v4-multi --reference-root dataset/frozen-eval-v1

启动中文人工校准界面：

    uv run python scripts/golden_review_app.py --bundle dataset/golden-review-exact-v2 --host 127.0.0.1 --port 8765

正式 readiness 通过后运行六轮：

    uv run python scripts/check_formal_release_gates.py --g0 out/preflight/g0.json --g1 out/preflight/g1.json --pilot out/preflight/paired-pilot.json --shadow out/preflight/shadow-round.json --out out/formal_release_gate_report.json
    uv run python scripts/train_real_harness.py --mode six-rounds --dataset-root dataset/vbench2-agent-training-index-v1 --readiness-report out/training_readiness_report.json --formal-release-report out/formal_release_gate_report.json --round-root out/training/agent-six-rounds-v1 --blender-bin D:\blender\blender.exe --workers 4 --vlm-model gpt-5.6-luna --markdown-path docs/t2blendercodeharness-agent-training-memory-v1.md

## 8. 相关文档

- docs/harness-architecture-v2.md：Harness 详细架构和训练前边界。
- docs/evaluator-v5-calibration.md：evaluator 校准和证据规则。
- docs/t2blendercodeharness-six-round-protocol.md：六轮协议。
- docs/superpowers/plans/2026-08-28-two-plan-convergence-completion.md：两份 remediation/codegen 方案融合状态。
- skills/t2blendercodeharness-training/SKILL.md：训练 Skill。
- skills/t2blendercodeharness/SKILL.md：Harness 使用 Skill。
- out/preflight/one-case-vlm-v3/vbench2-train-02-01/：真实双人物 handoff 回归样本。

## 9. 设计底线

1. 原始 benchmark prompt 不被改写成自造标签。
2. 不用固定模板冒充动态 Harness 能力。
3. 不用 plan、文件列表或 deterministic 分数冒充视觉质量。
4. VLM 缺失或证据不足时不填假分数。
5. Inner loop 只恢复执行，Harness 修改只能发生在 outer loop。
6. 每轮最多五个 attempt，每个 attempt 十个 train 加十个 paired dev。
7. 每轮结束都评估完整 60 train + 60 dev。
8. 每个 patch 只修改一个 Harness 组件。
9. 所有失败、回滚和分数变化都写入 append-only memory。
10. 任何提升都必须通过真实视频、严格 evaluator 和 dev 防过拟合检验。

## 10. 远程仓库发布边界

本仓库发布的是可复用的 Harness 工程，而不是把本机实验目录整体上传。会提交：

- src/videoact、evaluator、blender/lib 和必要的 Blender 执行代码；
- scripts、tests、Skills、pyproject.toml 和 uv.lock；
- 原始 VBench prompt source、当前 VBench train/dev/frozen index 和 schema；
- 架构、evaluator 校准、训练协议和方案审计文档；
- README 中引用的真实回归结果说明，但不复制大体积视频文件。

不会提交：

- out/ 下的 MP4、BLEND、PNG、日志和缓存；
- golden-review 的视频帧目录和本机人工标注临时文件；
- Python cache、pytest 临时目录和本地密钥/环境文件；
- 大体积历史训练 memory 和与当前 Harness 无关的实验产物。

真实视频可以按本文档的命令在本地重新生成；任何远程 clone 都必须重新配置本机 Blender 路径，并自行生成输出目录。
