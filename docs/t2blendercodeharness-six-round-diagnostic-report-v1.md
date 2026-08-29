# T2Blendercodeharness 六轮真实渲染诊断报告

## 结论

六轮外循环已按 active VBench-2.0 原始 prompt index 完成。每轮使用 10 个新 train + 10 个 paired dev，并在轮末重新跑完整 60 train + 60 dev；每轮只执行 1 次 outer attempt，因为没有出现需要重生成或支持新 Harness patch 的可归因失败。

这是一份 pre-calibration diagnostic，不是正式视觉质量验收：真实 Blender、DirectorPlan、source provenance、deterministic 和 artifact evidence 已完成；人工/独立 VLM 尚未可用，因此 task、人物/物体轨迹、摄像机调度、物理合理性和真实性分数均保持 unavailable。

- 实际协议 case-slot：`840`（attempt 120 + overall 720）。
- 协议最大上限：`1320`；未为凑满上限重复成功样本。
- Blender：`D:\blender\blender.exe`，4 workers，最多 12-case 分组串行。
- 生成模式：`agent` + in-process `codex-local`；没有使用 `template_baseline`。
- 每个 prepared source 必须包含 `CASE_SCENE_PROFILE` 和 `codex-local-case-profile-v2`，否则在 Blender 前 fail-closed。

## 六轮结果

数值列中的 artifact-only realism 是几何/PNG 低层证据，不是视觉真实性；`unavailable` 不是 0。

| Round | Attempt train artifact | Attempt dev artifact | Overall train artifact | Overall dev artifact | Overall train deterministic | Overall dev deterministic | Overall VLM scored |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.8 | 35.6895 | 38.5518 | 38.4716 | 100.0 | 100.0 | 0 / 0 |
| 2 | 39.3893 | 39.4053 | 38.5518 | 38.4716 | 100.0 | 100.0 | 0 / 0 |
| 3 | 39.1186 | 38.9305 | 38.5518 | 38.4716 | 100.0 | 100.0 | 0 / 0 |
| 4 | 38.9838 | 38.9043 | 38.5518 | 38.4716 | 100.0 | 100.0 | 0 / 0 |
| 5 | 38.9829 | 39.1335 | 38.5518 | 38.4716 | 100.0 | 100.0 | 0 / 0 |
| 6 | 39.036 | 38.7664 | 38.5518 | 38.4716 | 100.0 | 100.0 | 0 / 0 |

![六轮诊断曲线](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/director-multi-entity-harness/out/training/diagnostic-six-rounds-v2/six-round-curves.png)

## 真实性与 Harness 判断边界

1. round 1 的单组件升级是 `blender_code_agent`：增加 case-specific visual profile、较高细节的 parametric geometry、connected armature、event-conditioned pose keyframes 和 camera-facing orbit bias。新版 paired attempt 与旧 baseline 的 artifact-only realism 约为 train `31.33 → 35.80`、dev `31.52 → 35.69`；round-end overall 约为 train `33.65 → 38.55`、dev `33.62 → 38.47`。这只是 artifact proxy 改善，不是人工真实性验收。
2. round 2–6 没有再修改 Harness：每轮的 deterministic 失败计数为 0，artifact/prepare 失败为 0，且没有独立视觉证据可以区分穿模、断骨、动作时序或镜头语义错误；继续 patch 会把低层 proxy 当成视觉标签，存在过拟合风险。
3. “不是固定模板”由三层证据约束：训练 CLI 禁止 `template_baseline`，每 case 必须有 `codegen_call_id` 和独立 source hash，并且 source 内容必须含 case-specific profile marker。共享的 Blender runtime library 只是执行库，不是被偷偷替换的模板场景。
4. 训练结束后的语义审查发现 object-only prompt 曾被默认补成 `actor_a=person`；已在 Harness 中删除该默认补人规则。两例真实 smoke 现在只含 prompt-derived props + staging support，说明这次修复改变了计划语义；但画面仍是 proxy 级别，不能宣称已经达到真实人体/物理质量。

## 文件与人工后续

- 逐 case append-only Memory：`docs/t2blendercodeharness-agent-training-memory-v1.md`（当前 980 行）。
- 机器汇总：`C:\Users\sy\Desktop\T2BlenderCode\.worktrees\director-multi-entity-harness\out\training\diagnostic-six-rounds-v2\diagnostic-six-round-summary.json`。
- 曲线：`C:\Users\sy\Desktop\T2BlenderCode\.worktrees\director-multi-entity-harness\out\training\diagnostic-six-rounds-v2\six-round-curves.png`。
- 语义修复 smoke：`out/preflight/director-object-only-v4`（2/2 real video；无 actor 的 DirectorPlan，compound object 和 move 轨迹已保留）。
- 正式训练仍受 `golden_review=pending` 与 `paired_gate=pending` 阻塞；下一步应在升级后的 exact-prompt bundle 上完成人工 blind review，再重跑 readiness。

## 审计计数

| 检查 | 结果 |
|---|---:|
| 六轮 attempt 报告 | 12 split reports |
| 六轮 overall 报告 | 6 reports |
| Attempt 视频 | 120 |
| Overall 视频 | 720 |
| 当前 protocol 视频 | 840 |
| 零字节最终视频 | 0 |
| preparation/artifact failure | 0 |
| source provenance failure | 0 |
| 独立视觉评分 | unavailable（未填 0） |
