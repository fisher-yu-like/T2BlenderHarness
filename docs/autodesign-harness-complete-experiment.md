# AutoDesign Meta-Harness for Trajectory-Aware Text-to-Blender Proxy Generation

> Historical `trajectory-v2` report. The current renamed skill and hard generalization experiment are documented in [t2blendercodeharness-hard-experiment.md](t2blendercodeharness-hard-experiment.md).

## 完整实验文档

实验版本：trajectory-v2 / h-trajectory-v2  
Evaluator：deterministic-v1  
状态：数据集、fake benchmark、真实 Blender MCP 抽样验证已完成

---

## 摘要

本实验构建并验证了一个 contract-first Text-to-Blender Meta-Harness。系统把自然语言 prompt 转换为 SceneContract，再生成包含人物/物体状态、运动阶段、附着生命周期和摄像机编排的 TrajectoryPlan，随后通过 Blender MCP/CLI 执行白膜 proxy，经 artifact gate 与 deterministic evaluator 检查，并由 MetaHarnessOptimizer 根据 train/dev 失败证据提出单组件 Harness 更新。

本轮重点是复杂轨迹 prompt，而不是静态场景生成。新建 trajectory-v2 数据集共 80 条样本，覆盖 10 个轨迹家族，显式描述 walk → reach → grasp → lift → carry → place → release，并加入 follow、orbit、dolly、close-up、occlusion/reveal 等摄像机意图。fake benchmark 上 train/dev/test 均为 100 分；真实 Blender MCP 对复杂 dev 样本完成了完整 proxy 渲染、MP4 组装、artifact gate 与 deterministic evaluator 验证。

当前 VLM endpoint 返回 HTTP 403/1010，因此没有伪造 VLM 分数，也没有将 unavailable 结果用于 Harness 训练。本轮 train/dev 没有重复 actionable failure，outer loop 正确输出 no_patch。proposal、train、dev、test、no_patch 的全过程已写入 event-sourced memory。

---

## 1. 实验目标与研究问题

### 1.1 总目标

验证以下闭环是否能够稳定运行：

    自然语言场景
      → SceneContract
      → TrajectoryPlan / CameraPlan
      → Blender proxy 执行
      → artifact gate
      → deterministic evaluator
      → 可选 VLM
      → train/dev 失败聚合
      → 单组件 Harness 更新
      → acceptance gate
      → memory 持久化

### 1.2 研究问题

1. 长 prompt 能否恢复有序动作阶段，而不是只识别单个动作？
2. TrajectoryPlanner 能否生成 frame-indexed states、motion primitives 和 attach/detach 生命周期？
3. CameraPlanner 能否把 follow/orbit/dolly/close-up 意图转换为可验证 shots？
4. Evaluator 能否阻止不完整 artifact、实体语义错误和摄像机错误进入后续阶段？
5. MetaHarness 是否遵守 one-owner、train improvement、dev non-regression 和 test isolation？
6. Harness 更新是否能被作为长期 memory 检索，而不是只保留最终版本号？

### 1.3 实验假设

- H1：显式轨迹 prompt 会产生更丰富的事件序列、状态数、motion primitives 和 camera shots。
- H2：contract/plan 验证可以在 Blender 执行前捕获结构性错误。
- H3：artifact gate + deterministic evaluator 可以阻止 telemetry-only 假通过。
- H4：one-owner gate 可以保持 Harness 更新的因果可归因性。
- H5：append-only memory 可以保存每次 proposal、评估和 acceptance decision。

---

## 2. 系统架构

### 2.1 组件

    Codex Host
    ├── AutoDesign Harness Skill
    ├── MetaHarnessOptimizer
    ├── Dataset + Split Manager
    ├── Harness Memory Store
    └── Evaluator Runner

    DesignHarness
    ├── Scene/Prompt Parser
    ├── Camera Choreography Planner
    ├── Character/Object Trajectory Planner
    ├── Blender MCP/CLI Executor
    ├── Host Video Renderer
    ├── Proxy Validator
    └── Candidate Selection/Fallback

    Evaluator
    ├── Contract/plan predicates
    ├── Artifact Gate
    ├── Telemetry checks
    ├── Camera coverage checks
    ├── Physics/support checks
    ├── Optional VLM judge
    └── Train/dev/test aggregation

### 2.2 内循环

    Prompt
      → SceneContractBuilder
      → TrajectoryPlanner + CameraPlanner
      → Blender MCP/CLI job
      → .blend + PNG + sampled frames + telemetry
      → RealArtifactGate
      → DeterministicEvaluator
      → Optional VLM
      → local repair / candidate promotion

状态机：

    prepared → executing → rendered → artifact_valid → evaluated
        └──────────────────────────────────────────────→ failed

### 2.3 外循环

    train records
      → aggregate repeated failures
      → identify one owner
      → create patch proposal
      → apply at most one Harness component patch
      → rerun train
      → rerun dev
      → accept or reject
      → frozen test

本实验的“训练 Harness”指代码级 Harness evolution，不是模型权重训练。

---

## 3. 数据集设计

### 3.1 统计

| 项目 | 值 |
|---|---:|
| Dataset ID | trajectory-v2 |
| Schema | trajectory-dataset-v2 |
| 样本数 | 80 |
| 轨迹家族 | 10 |
| Train | 50 |
| Dev | 20 |
| Test | 10 |
| Evaluator | deterministic-v1 |
| Fingerprint | 0c4ea75a6823b60ef22dcf1b5f75db6595c1baf951068140b8ef97ec8876800a |

原始 40-case 数据集保留在 dataset/；新数据集独立存放在 dataset/trajectory-v2/，未覆盖旧基线。

### 3.2 轨迹家族

1. approach_lift_carry_release
2. orbit_transfer
3. occlusion_reveal
4. closeup_grasp_release
5. dolly_drop_zone
6. orbit_before_place
7. long_duration_retake
8. support_contact_lifecycle
9. multi_shot_handoff
10. smooth_phase_transfer

每个家族 8 个 variants：5 条 train、2 条 dev、1 条 test。每个 split 都覆盖全部家族。

### 3.3 Prompt 信息层

每条复杂 prompt 同时描述：

- scene establishment：wide/overview；
- character path：walk/reach；
- object interaction：grasp/lift/carry/place/release；
- support relation：object on table/support；
- camera choreography：follow/orbit/dolly/close-up/reveal；
- visibility constraint：grasp 前 target 可见；
- temporal constraint：long shot、ease-in/ease-out、final hold；
- physics constraint：no penetration、support-before-grasp、attachment lifecycle。

代表性 prompt：

    Begin with a wide establishing shot. The character walks to the table,
    reaches for the blue cube, grasps it, lifts it, carries it to the drop zone,
    places it down, and releases it. The camera follows the approach and dollies
    into a close-up of the release while keeping the object visible before grasp.
    Begin from the right side of frame.

对应规划证据：

    Events:      walk → reach → grasp → lift → carry → place → release
    States:      initial → approach → reach → grasp → lift → carry → place → release
    Attachment:  attach at grasp → detach at release
    Camera:      follow overview → dolly action close-up

---

## 4. Contract 与轨迹规划

### 4.1 SceneContract

SceneContract 包含 entities、events、relations、must_show、physics_constraints 和 camera_constraints。

新增动作识别：

    walk, reach, grasp, lift, carry, place, release, reveal

旧的简单 prompt 保持兼容。

### 4.2 典型事件时间轴

以 16 秒场景为例：

| Event | 时间比例 | 含义 |
|---|---:|---|
| walk | 0.00–0.25 | 走向 support |
| reach | 0.25–0.40 | 接近目标 |
| grasp | 0.40–0.55 | 建立 attachment |
| lift | 0.55–0.65 | 抬离表面 |
| carry | 0.65–0.82 | 携带移动 |
| place | 0.82–0.92 | 放到 drop zone |
| release | 0.92–1.00 | 解除 attachment |

事件必须按 start time 排序，且不能超出 scene duration。

### 4.3 TrajectoryPlan

复杂人物轨迹包含：

- one-based frame-indexed EntityState；
- ease_in_out 与 linear motion primitives；
- attach/detach AttachmentEvent；
- phase-specific positions；
- 与 FPS/时间轴一致的状态范围。

代表性人物状态至少包含初始、walk、reach、grasp、lift、carry、place、release 八个阶段。目标物体在 carry/place 阶段同步改变位置，support 和 drop zone 保持环境状态。

### 4.4 CameraPlan

| Shot | 类型 | 用途 |
|---|---|---|
| shot-establish-follow | follow | 建立场景并跟随 walk |
| shot-carry-orbit | orbit | carry 期间保持目标可见 |
| shot-action-dolly | dolly | 进入 grasp/place/release close-up |

所有 must_show 事件必须被至少一个 shot 覆盖，并写入 event observability。

---

## 5. Blender 执行与真实物料

### 5.1 白膜 proxy

- character：sphere proxy；
- support/table/drop zone：cube proxy；
- target object：cylinder proxy；
- material：ProxyWhiteMaterial；
- area light；
- animated camera；
- render engine fallback；
- Blender 输出 PNG sequence；
- host 端组装 MP4。

### 5.2 Artifact contract

成功 real run 必须包含：

    run_manifest.json
    scene_contract.json
    trajectory.json
    camera_plan.json
    blender_job.py
    proxy.blend
    proxy.mp4
    telemetry.json
    frames/index.json
    frames/*.png
    deterministic_report.json

至少 3 张 sampled PNG 必须可读。Telemetry 单独存在不能放行。

### 5.3 运行环境

| 项目 | 值 |
|---|---|
| Blender | 5.1.2 |
| Render engine | BLENDER_EEVEE |
| Resolution | 256 × 256 |
| FPS | 24 |
| Blender FFMPEG | 不可用 |
| MP4 assembly | host-side imageio-ffmpeg |

### 5.4 真实复杂样本

样本路径：

    out/real/trajectory-v2-dev-dry/traj-01-06

结果：

- dev complex job source 生成并编译；
- Blender MCP 执行成功；
- frame range：1–384；
- .blend、PNG animation、sampled frames、telemetry、MP4 均存在；
- artifact gate：complete；
- deterministic evaluator：pass；
- score：100.0；
- active camera：true；
- camera shots：follow + dolly。

Telemetry 记录了真实 proxy entity kind、keyframe count、camera shot trajectory type 和 event observability。

---

## 6. Evaluator

### 6.1 Deterministic evaluator

Evaluator 分为两层：

1. 计划层：event order、required event coverage、camera coverage、support-before-grasp、attachment contact、velocity continuity。
2. 真实层：artifact status、telemetry entities、entity kind、timebase、FPS、active camera、readable sampled frames。

硬失败包括 incomplete real artifacts、missing entity、entity kind mismatch、timebase mismatch、inactive camera、camera event uncovered 和 failed/timeout execution。

### 6.2 真实 bug 与修复

早期 calibration 暴露出：table 在 proxy 生成时被当成普通 prop。计划层可能通过，但真实 sampled frame 不能证明 support 语义正确。

修复为：

1. 生成脚本根据 entity semantic kind 创建 support proxy；
2. telemetry 记录实际生成的 kind；
3. evaluator 对照 SceneContract 做硬校验；
4. 抽查 sampled frame 验证视觉布局。

因此当前流程不会仅依据“报告 100 分”宣称真实视觉正确。

### 6.3 VLM 边界

VLM 只接收 artifact-complete 且 deterministic-pass 的候选。当前 endpoint 返回 HTTP 403/1010，统一记录为 unavailable。

不得将 unavailable 转换为 0 分、负样本、偏好对或 patch 依据。因此本实验的 100 分是 deterministic score，不是完整视频视觉质量已由 VLM 证明。

---

## 7. Harness 训练与 Meta-Evolution

### 7.1 训练定义

    failure evidence
     → owner grouping
     → patch proposal
     → one component patch
     → train rerun
     → dev rerun
     → acceptance / rejection / rollback

### 7.2 One-owner 规则

| Owner | 典型文件 |
|---|---|
| scene_parser | src/videoact/scene_contract.py |
| trajectory_planner | src/videoact/trajectory.py |
| camera_planner | src/videoact/camera.py |
| blender_executor | src/videoact/blender_adapter.py |
| evaluator | evaluator/deterministic.py |

如果 parser、camera、executor 同时失败，不能合并为一个 patch，必须拆分后逐个验收。

### 7.3 Acceptance gate

    train_after > train_before
    dev_after >= dev_before
    hard_dev_regression == false

Test 只做最终盲测，不参与 patch 选择、调参、失败聚合或标签修改。

### 7.4 本轮结果

    status: no_patch
    train_score: 100.0
    dev_score: 100.0
    proposal_case_ids: []

no_patch 表示当前 train 没有重复 actionable failure，不表示训练流程缺失。

---

## 8. Harness Memory

### 8.1 Memory 字段

append-only JSONL 每条记录包含 memory_id、event_index、event、timestamp、parent/current Harness version、owner、dataset/evaluator fingerprints、affected case IDs，以及 score、evidence、reason、hard regression 等 payload。

### 8.2 支持事件

    proposal
    patch_applied
    train_evaluated
    dev_evaluated
    test_evaluated
    accepted
    rejected
    rollback
    no_patch

### 8.3 本轮 memory

Memory ID：

    memory-9b495f68b4271eff

事件序列：

| Index | Event | 结果 |
|---:|---|---|
| 0 | proposal | 建立 h-trajectory-v2 基线记忆 |
| 1 | train_evaluated | 50 cases，mean 100.0 |
| 2 | dev_evaluated | 20 cases，mean 100.0，无 hard regression |
| 3 | test_evaluated | 10 frozen cases，mean 100.0 |
| 4 | no_patch | 无重复 actionable failure |

本轮没有 patch_applied 或 accepted，因为没有合法 patch 候选。该结果仍被保留，防止未来重复评估或误认为尚未训练。被拒绝或 rollback 的更新也必须保留。

---

## 9. 结果与验证

### 9.1 Dataset validation

    cases: 80
    train/dev/test: 50/20/10
    unique prompt hashes: 80
    trajectory metadata drift: 0
    planner validation errors: 0

### 9.2 Fake benchmark

    train: 50/50 pass, mean 100.0
    dev:   20/20 pass, mean 100.0
    test:  10/10 pass, mean 100.0

### 9.3 Real Blender path

    dev jobs generated: 20/20
    job source compiled: 20/20
    real sample executed: 1
    Blender: 5.1.2
    frame count: 384
    artifact gate: complete
    deterministic: pass
    score: 100.0
    VLM: unavailable, HTTP 403/1010

### 9.4 自动化验证

最终项目测试：100 passed。

另外：

- trajectory-v2 validator：80/80；
- prompt hashes：80 unique；
- memory integrity：5 events，index 连续 0–4；
- compileall：通过；
- real complex job source compile：通过。

---

## 10. 复现实验命令

以下使用项目已验证的 bundled Python。

### 10.1 构建与校验数据集

    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\build_trajectory_dataset.py
    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\validate_trajectory_dataset.py --root dataset\trajectory-v2

### 10.2 运行 train/dev/test

    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\train_harness_with_memory.py --dataset-root dataset\trajectory-v2 --out-root out\training\trajectory-v2-final --harness-version h-trajectory-v2

### 10.3 生成真实 jobs

    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\prepare_real_jobs.py --split dev --dataset-root dataset\trajectory-v2 --out-dir out\real\trajectory-v2-dev-dry --harness-version h-trajectory-v2

### 10.4 通过 Blender MCP 执行 job

    from pathlib import Path
    job = Path(r"C:\Users\sy\Desktop\T2BlenderCode\out\real\trajectory-v2-dev-dry\traj-01-06\blender_job.py")
    exec(compile(job.read_text(encoding="utf-8"), str(job), "exec"))

### 10.5 真实 artifact/evaluator 验收

    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -c "import sys,json;sys.path[:0]=['.','src'];from scripts.evaluate_real_runs import evaluate_real_run;print(json.dumps(evaluate_real_run('out/real/trajectory-v2-dev-dry/traj-01-06'),indent=2))"

### 10.6 全套测试

    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q
    & 'C:\Users\sy\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m compileall -q src evaluator blender scripts training tests

---

## 11. 产物索引

设计与协议：

- docs/superpowers/specs/2026-08-25-trajectory-dataset-and-harness-memory-design.md
- docs/superpowers/plans/2026-08-25-trajectory-dataset-and-harness-memory.md
- docs/real-run-protocol.md
- docs/real-run-report.md

Dataset：

- dataset/trajectory-v2/manifest.jsonl
- dataset/trajectory-v2/splits.json
- dataset/trajectory-v2/metadata.json
- scripts/validate_trajectory_dataset.py

Training and memory：

- scripts/train_harness_with_memory.py
- training/harness_memory.py
- out/training/trajectory-v2-final/training_report.json
- out/training/trajectory-v2-final/memory/harness_updates.jsonl

Real sample：

- out/real/trajectory-v2-dev-dry/traj-01-06/run_manifest.json
- out/real/trajectory-v2-dev-dry/traj-01-06/telemetry.json
- out/real/trajectory-v2-dev-dry/traj-01-06/deterministic_report.json
- out/real/trajectory-v2-dev-dry/traj-01-06/frames/frame_000001.png

---

## 12. 局限性与下一步

### 12.1 局限性

1. fake benchmark 的 100 分主要验证 contract/planner/evaluator 确定性闭环，不等同于所有真实视频视觉质量都已证明。
2. deterministic evaluator 仍主要验证结构、telemetry 和计划一致性；复杂遮挡、真实视觉可见性和长时连续性需要更多几何级或 VLM 检查。
3. VLM endpoint 当前不可用，没有可用的视觉偏好分数。
4. labels 仍是 unreviewed 占位，preference-pair export 尚未满足 calibration-ready、accepted candidate 和 reproducible test 三项门槛。
5. 本轮没有 accepted Harness patch，因此 memory 验证了 no-patch 路径；accepted/rejected/rollback 的真实跨版本演化需要注入失败样本或真实失败运行。

### 12.2 下一步

- 配置 tenant-approved VLM endpoint，使用已有 immutable real videos 重新评分；
- 增加人工 calibration labels，覆盖 pass、边界和失败样本；
- 增加 camera coverage、attachment detach、support semantic mismatch 等 injected failures；
- 验证单 owner patch 的真实 train improvement/dev non-regression；
- 扩大真实 MCP 渲染到 trajectory-v2 calibration subset；
- 将 memory 检索接入下一轮 Harness proposal，防止重复回归。

---

## 结论

本实验完成了一个轨迹规划显式化、可复现、可审计的 Text-to-Blender Meta-Harness 闭环。trajectory-v2 证明复杂动作序列、motion primitives、attachment lifecycle 和 camera choreography 可以被结构化保存、规划、执行和评估；真实 Blender MCP 抽样证明这些规划产物能够进入真实 proxy job 并形成完整 .blend、MP4、telemetry 和 deterministic report。

在当前数据和 evaluator 下，train/dev/test 均通过且没有可归因的重复失败，因此系统没有接受 Harness patch，而是将这一结论作为 no_patch memory 保存。这应解释为“当前候选在现有确定性评测上没有暴露可修复失败”，而不是“模型已经完成视觉训练”或“所有视频质量都已被 VLM 证明”。
