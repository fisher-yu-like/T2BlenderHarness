# Codex BlenderCodingAgent 设计

## 目标

把 `T2Blendercodeharness` 的 Blender 代码生成从本地固定 Python 模板改为真实的 Codex coding-agent 调用。每个 case 的 `blender_job.py` 必须由 `codex exec` 根据该 case 的原始 prompt、DirectorPlan 和渲染契约独立生成；本地代码只负责准备上下文、调用 Codex、验证输出和记录审计证据。

## 非目标

- 不删除或重写已经有效的数据集、Evaluator 公式和 Director 事件规划逻辑；
- 不让 Codex 直接修改 Harness 源代码、数据集标签、Evaluator 或 skill；
- 不用固定人物名单、固定物体名单、固定数量、固定动作轨迹或固定几何模板生成 Blender 场景；
- 不把 Codex 生成失败静默降级到本地模板。

## 架构

```text
exact prompt
  → DirectorAgent
  → DirectorPlan + TrajectoryPlan + CameraPlan + proxy policy
  → BlenderCodingAgent
      → coding_agent_request.json
      → codex exec (isolated case directory)
      → case-specific blender_job.py
      → generated_code_manifest.json
  → Python compile check
  → Blender CLI / MCP
  → artifact gate + evaluator
```

`BlenderCodingAgent` 位于 `src/videoact/blender_coding_agent.py`。它不包含任何 Blender 建模代码，不解析 actor/prop token，不生成默认实体。它的职责只有：

1. 将精确 prompt、DirectorPlan、TrajectoryPlan、CameraPlan、proxy geometry policy 和 run manifest 写成只读上下文文件；
2. 调用本机 `codex exec`，工作目录限定为当前 case 目录；
3. 要求 Codex 只写 `blender_job.py`、`generated_code_manifest.json` 和 `codex_agent_last_message.md`；
4. 验证 `blender_job.py` 存在、非空、可由 Python 编译，并且不包含向外写入其他路径的明显路径；
5. Codex 调用失败、超时、没有代码或代码不能编译时最多重试 2 次，并记录每次 attempt；
6. 所有失败都保持 fail-closed，交给上层记录为未生成，绝不回退到本地场景模板。

## 两种 Codex coding mode

### Director mode（正式 Harness）

Codex 输入 exact prompt 和完整 Director 输出。它必须忠实执行 DirectorPlan 的实体、事件依赖、交互生命周期、人物/物体轨迹、摄像机覆盖、时间窗和负约束，同时根据 proxy geometry policy 构建足够有区分度的场景。DirectorPlan 是语义约束，不是 Blender 代码；Codex 仍需自行决定适合每个 prompt 的 mesh、材质、灯光、动作层级和镜头实现。

### Raw prompt mode（诊断用）

如未来需要重新做消融，Codex 只收到 raw prompt 和通用渲染/产物契约，不收到 `event_graph`、`oracle_expectations` 或 DirectorPlan。它也必须独立解析任意数量的实体，不得使用默认 Alice/Bob 或 red cube/blue cup。该模式不是本次正式训练入口，必须单独标记并单独评估。

## Codex 提示词契约

每次调用都要求：

- 只处理当前 case，不修改仓库其他文件；
- 直接创建 `blender_job.py`，而不是返回一段未落盘的 markdown 代码；
- 读取 `coding_agent_context.json`，其中保留 exact prompt 和所有计划 hash；
- 生成真实可运行的 Blender Python，使用 `bpy` 和 `mathutils`；
- 实现动态实体、事件时序、人物/物体轨迹、接触/交接/放置状态、摄像机运动、可见性和真实 proxy geometry；
- 写出 `proxy.blend`、动画 PNG、sample PNG index、telemetry 和 manifest 更新；
- 不使用固定实体名或固定数量作为默认答案；无法解释的隐含细节要记录为 assumption，而不是悄悄改写 prompt；
- 结束时返回生成文件、代码 hash、实现摘要和未解决风险。

## 生成代码验收

本地验收只检查生成物，不重写生成物：

1. 输出路径只在 case 目录内；
2. Python compile 成功；
3. manifest 中 prompt hash、DirectorPlan hash、TrajectoryPlan hash 和 code hash 一致；
4. 代码存在 entity/trajectory/camera/telemetry 输出路径；
5. 真实 Blender CLI 渲染完成并通过 retry/artifact gate；
6. deterministic 和视觉评估分别记录；
7. 代码/计划/视频全部按 case 归档，便于 Harness memory 追溯。

## 删除与保留边界

删除本次已经确认错误的固定 direct ablation adapter、三臂 direct 结果、三臂盲评结果、错误 release bundle 及其专用测试/报告。保留 `trajectory-v4-multi`、`vbench-derived-100-v1`、有效的历史训练结果、Evaluator 校准、Director 计划和训练 memory，并在实验文档中将旧结果标为非 Codex coding-agent 结果。

## 失败策略

- `codex` 不存在或未登录：记录 `coding_agent_unavailable`，不生成视频；
- Codex 返回但没有 `blender_job.py`：重试；
- 代码编译失败：把 stderr/last message 记录后重试；
- Blender 运行失败：沿用现有 CLI renderer 的最多 2 次 render retry；
- 真实视频缺失：不进入视觉打分、训练 proposal 或 patch acceptance；
- 不允许用固定模板补齐任何失败。

## 验证计划

先用 fake runner 测试 request/context、写入边界、重试和 compile gate；再用真实 `codex exec` 对一个复杂 prompt 生成代码并用 `D:\blender\blender.exe` 渲染。只有 smoke 通过，才准备新的训练批次。原三臂 direct 结果不复用。
