# T2BlenderHarness improvement plan 执行状态

依据：`D:/sy/T2BlenderHarness-improvement-plan-zh.md`

更新时间：2026-08-29（工作树 `codex/director-multi-entity-harness`）

## 结论

代码层面的可信边界、评测接口、统计门禁和训练准入已经接入；正式六轮 Harness 自进化尚未放行。当前任何缺少真实 evidence 的地方都保持 `pending`/`blocked`，没有用旧结果、合成分数或模板结果冒充通过。

## Task 状态

| Task | 状态 | 已完成内容 | 尚缺证据/人工工作 |
|---|---|---|---|
| T01 provenance / model-vs-template | 已完成 | `provider_mode=model`、外部 Director + 本地 Codex codegen 的分阶段 provenance、模板只作显式 baseline、失败 fail-closed | 需要重新获得一次真实双调用成功样本 |
| T02 portability | 已完成 | 相对路径、CI、repo portability 检查 | 干净 clone CI 仍应在远程再执行一次 |
| T03 trusted observer | 已完成 | 独立 observer、原始逐帧 telemetry、生成 telemetry 隔离 | 当前 baseline smoke 已通过；正式 model smoke 未通过 |
| T04 blind separation | 已完成 | `primary-blind-v1` 排除 plan、finding、owner、arm | 需要真实独立 judge 调用记录 |
| T05 execution/semantic/quality states | 已完成 | result contract 和阶段状态区分；补上 prop-only 物体事件 participant 规则 | 等正式 model 全流程填充 |
| T06 scoring-v7 / applicability | 已完成 | N/A mask、required-event gate、task/realism 分离 | G1 前冻结配置 |
| T07 evidence/confidence | 已完成 | 逐维 score/confidence/completeness/refs/interval | 需要 golden calibration 校准阈值 |
| T08 bounded inner retry | 已完成 | 最多 3 次 fresh candidate、stage/lineage 记录；无隐藏模板回退 | 需要正式 run 的 retry lineage |
| T09 semantic source reuse | 已完成 | normalized AST、control-flow、scene/animation signature；100 对有标签审计通过 | 正式 run 需保留真实 source pair |
| T10 paired statistics | 已完成 | 固定 seed bootstrap、train gain、dev non-inferiority、安全指标 | 需要 pilot 的真实 paired delta |
| T11 cross-owner | 已完成 | dependency manifest + A-only/B-only/A+B 三臂门禁 | 当前无合法联合 patch 待验收 |
| T12 human calibration | 代码完成/人工待办 | 14 维 MAE、event F1、confidence reliability 报告 | `dataset/golden-review-exact-v2` 当前 0 score rows、0 annotators；必须人工完成双标注 |
| T13 physics oracle | 已完成（smoke 已验证） | OBB/SAT + mesh BVH、contact/ownership/continuity；只读 trusted raw observations | 当前单人物 smoke 没有交互 prop，BVH queries=0；需含真实接触对象的 model case |
| T14 experiment fingerprint | 已完成 | provider/source/blend/observer/telemetry/MP4/evaluator/config/环境全链 hash | 真实 model run 需生成完整 fingerprint |
| T15 frozen/OOD | 已完成 | `frozen-eval-v2` 60 条，Human_Identity、Instance_Preservation、Material、Multi-View_Consistency、Thermotics 各 12 条；四层 collision=0 | 只允许 milestone 评测，不可调参 |
| T16 active sampling | 代码完成/真实回放待办 | train/dev-only sampler、sequential stopping、历史 replay 审计；目标节省≥30%、一致率≥95% | 当前没有把旧实验强行转换为正式 replay；需要真实历史 batch 或 shadow round |
| T17 release gates | 代码完成/准入待办 | sealed G0–G3、嵌套 evidence 重放、readiness 引用 gate hashes | G0/G1/G2/G3 尚未全部通过 |

## 已验证报告

当前正式生成边界已经更新为：外部 Director provider 使用环境变量
`OPENAI_API_KEY`/`OPENAI_BASE_URL`，调用 OpenAI-compatible
`/v1/chat/completions`；本地 Blender code 阶段使用 `codex exec`。两阶段的
provider kind、model id、call id、request/response hash 在
`provider_manifest.json` 中分别记录，readiness 会拒绝把同一个错误 provider
冒充成双阶段成功。

- 当前能力报告：`out/preflight/skill-capability-current.json`，status=`pass`。
- VBench prompt index：140 条，train/dev/test=`60/60/20`，status=`pass`。
- frozen-OOD：60 条，5 个 slice 维度各 12 条，status=`pass`，四层泄漏计数均为 0。
- source fingerprint：100 对，precision=`1.0`、recall=`1.0`、F1=`1.0`。
- 最新全量测试：`out/preflight/pytest-assistant-vbench-final.xml`，`544 passed, 3 skipped, 28 warnings`（JUnit 总数 547）；跳过项是历史 baseline 可选项。compileall、`git diff --check` 和 capability check 均通过。历史回归曾发现并修复了 plan-only oracle、CLI 参数兼容、rerender 状态和 candidate.blend artifact 问题；本次还验证了 prop-only 事件不会被错误判为无 participant。

## 真实 smoke 记录

1. `out/preflight/model-double-call-smoke-current2`：拆分 provider 前的历史 smoke；本地 Codex CLI Director 子进程 180 秒超时，未产生 DirectorPlan，也未进入 Blender。该目录保留为 G0 blocked evidence，不是模型通过。
2. `out/preflight/model-double-call-smoke-current3`：拆分 provider 前的 30 秒真实 smoke；当前 manifest 已保存精确的 `codex_exec_local` Director timeout call 和 `preparation_gate`，没有模板替代。
3. `out/preflight/external-director-smoke-current1`：使用环境变量 `OPENAI_API_KEY`/`OPENAI_BASE_URL` 的 Chat Completions 请求成功到达 provider，但一次返回的 evidence span 越过 prompt 边界，被本地 Director contract 拒绝；因此没有进入 Blender codegen。这证明请求路径和 fail-closed 校验在工作。
4. `out/preflight/external-director-smoke-current2`：外部 `gpt-5.6-luna` Director 调用真实返回并写入 provider provenance，但模型给出 unresolved hard uncertainty，被本地 contract 拒绝；没有模板替代。
5. `out/preflight/hybrid-pair-smoke-current1`：外部 `gpt-5.6-luna` Director 成功返回并通过本地 DirectorPlan contract；随后本地 `codex exec` Blender-code 调用真实超时 180 秒，provider manifest 保存了精确失败 call，未产生 source/视频，也未回退模板。这是当前 G0 blocked evidence。
6. `out/preflight/observer-bvh-smoke-current3`：显式 `rule_template_baseline` 仅用于 observer/Blender artifact smoke。真实 Blender 5.1.2 渲染成功，120 帧、MP4 可播放、trusted observer 成功、每帧 512 mesh triangles、candidate/proxy hash 相同。该目录不能用于证明 model-driven 生成或正式训练。
7. `out/preflight/observer-bvh-smoke-current2`：发现并保留了旧 artifact contract 失败（candidate.blend 缺失），对应生成器修复前的真实失败记忆。
8. `out/preflight/assistant-full-chain-vbench-06-15-v2`：当前 Codex 直接生成的首个球体 case-specific Blender 源码在真实 Blender 5.1.2 中暴露 `scene.world is None`，在 candidate 保存前失败；该目录保留为真实运行时失败证据。
9. `out/preflight/assistant-full-chain-vbench-06-15-v3`：针对上述失败的精确修复 provider 因目标行缩进不匹配而 fail-closed，未生成源码；这是拒绝不确定改写的记录。
10. `out/preflight/assistant-full-chain-vbench-06-15-v4`：修复 `World` 后暴露 Blender 5.1 slot-based `Action` 没有直接 `fcurves` 的第二个真实运行时错误，仍未生成视频；失败和修复前源码保留。
11. `out/preflight/assistant-full-chain-vbench-06-15-v5`：不调用 `codex exec`，由当前 Codex 生成 `assistant_generated` Director/code 两阶段产物；真实 Blender 返回 0，trusted observer 成功，120 帧和 MP4 可播放，动态 source audit 通过，deterministic=100，physics oracle=pass（penetration=0、teleport=0、BVH intersection=0）。随后对 8 个真实帧做了带 evidence refs 的 Codex-local review：task=88.8873、realism=68.9466、realism_vlm=82.6785；这些是诊断分数，不解锁 formal training。

## Gate 状态

| Gate | 当前状态 | 原因 |
|---|---|---|
| G0 | blocked | 已有一次外部 Director 成功并通过 contract，但本地 Codex Blender-code 调用超时；尚无成功双调用 smoke |
| G1 | blocked/pending | golden review 没有 30–50 case 的双标注完整记录；evaluator 代码已测试但不能替代人工校准 |
| G2 | not started | 必须先有 sealed G0/G1，再做精确 10 train + 10 dev paired pilot |
| G3 | not started | 必须先有 G2，再做不改 Harness 的 60 train + 60 dev shadow |
| G4 | blocked | `train_real_harness.py` formal mode 需要 readiness 和 sealed G0–G3；六轮没有启动 |

## 下一步唯一放行顺序

1. 正式 `provider_mode=model` 仍需重新获得一份真实双调用成功样本：外部 Director 返回合规 DirectorPlan，随后本地 `codex exec` 生成 Blender code，并完成 candidate/source/artifact/fingerprint 审查；v5 的当前 Codex `assistant_diagnostic` 结果不能替代该 gate。
2. 人工在 golden review 前端对 exact prompt 的 90 个匿名真实 MP4 完成两名独立评分者的 14 维评分、证据和置信度。
3. 用人工结果运行 calibration，冻结 scoring-v7、evidence threshold、judge model 和 paired margins。
4. 生成并封存 G0/G1；完成 10 train + 10 dev paired pilot（G2），再执行不改 Harness 的 60+60 shadow（G3）。
5. 只有 `training_allowed=true` 且 release report hash 可重验时，才开始六轮；每轮最多 5 次 attempt，每次 10 train + 10 dev，轮末完整 60 train + 60 dev，所有 memory append-only。
