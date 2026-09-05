---
record_id: harness-rsi-continuation-2026-09-03
project: T2BlenderCode
worktree: harness-rsi
record_date: 2026-09-03
protocol: t2blendercodeharness-v5-executable-director
status: blocked_before_formal_training
training_allowed: false
formal_rounds_completed: 0
---

# harness-rsi 续训记录

## 1. 记录范围

本记录汇总 `harness-rsi` 工作树截至 2026-09-03 的续训执行结果。这里的“训练”是 code-level Harness evolution：固定 VBench prompt 和评估协议，改进 Director、Blender codegen、执行器或 evaluator owner；不是神经网络梯度训练。

正式六轮训练尚未开始。当前状态为 `training_allowed=false`，阻断发生在正式训练之前的人工 golden review 和 paired gate。

## 2. 训练协议

| 项目 | 固定值 |
|---|---|
| active dataset | `dataset/vbench2-agent-training-index-v1` |
| active split | train 60 / dev 60 / test 20 |
| frozen eval | `dataset/frozen-eval-v2`，60 cases |
| training rounds | 6 |
| 每次 attempt | 10 train + 10 paired dev |
| 每轮上限 | 5 attempts |
| G2 | 20-case paired pilot |
| G3 | 60 train + 60 dev shadow round |
| 正式生成路径 | external Director + local Codex codegen，`provider_mode=model` |
| Director | `external_openai_compatible:gpt-5.6-luna` |
| Blender codegen | `codex_exec_local:codex-cli` |
| Blender | 5.1.2 |

## 3. 当前门禁结果

| Gate | 状态 | 证据或原因 |
|---|---|---|
| full test | pass | `733 passed, 3 skipped, 1 deselected, 35 warnings` |
| capability | pass | required components、imports、contract、evaluator、fail-closed 均通过 |
| dataset | pass | 140 cases，train/dev/test = 60/60/20；fingerprint `196f71f7...43332` |
| frozen eval | pass | 60 cases；5 个 OOD 维度各 12 cases；collision = 0 |
| source fingerprint | pass | 100 pairs；precision/recall/F1 = 1.0 |
| owner challenges | pass | 70 fixtures、7 owners、35 pairs |
| closed-loop liveness | pass | 7/7 detected；recall = 1.0；owner accuracy = 1.0 |
| real Blender smoke | pass | G0 r7；candidate、trusted observer、120 帧和 MP4 完整 |
| dynamic provider | pass | provider 身份与正式配置一致 |
| golden review | pending | 90 个视频仍为 0/180 条人工评分，ICC 未生成 |
| paired gate | pending | `paired_gate_report_missing` |
| formal six-round training | not started | 被上述两个 gate 阻止 |

最新综合 readiness：

```text
status          = blocked
training_allowed = false
blocking_gates  = golden_review, paired_gate
```

## 4. G0 真实链路执行

### 4.1 r6 基线问题

首个成功的双 provider smoke 使用真实 external Director 和 local Codex codegen，生成、渲染和 artifact contract 均可通过；但生成的 Blender source 没有创建灯光。

独立视觉 judge 对 r6 的判断是：画面严重欠曝，8 个抽帧几乎只有低对比度轮廓，视觉主分约 8.94/100。该结果被记录为失败证据，没有被包装成视觉通过。

### 4.2 r7 修复结果

新增正式 agent codegen 的可见灯光约束，并将其加入：

- `BlenderCodeAgent` 的可选静态 gate；
- `provider_mode=model` 的正式 wiring；
- local Codex codegen prompt；
- 对应回归测试。

G0 r7 使用 case `vbench2-train-01-01`，结果如下：

| 项目 | 结果 |
|---|---:|
| Director 调用 | 1 次，`gpt-5.6-luna` |
| Blender codegen 调用 | 1 次，`codex-cli` |
| Blender 执行 | pass，5.1.2 |
| trusted observer | pass |
| 动画帧 | 120 |
| 视频 | 10 秒、12 fps、可播放 |
| deterministic/artifact score | 100.0 |
| physics oracle | pass；penetration/teleport/BVH intersection = 0 |
| 独立视觉 judge confidence | 0.96 |
| independent visual overall | 76.277 |
| visual task score | 83.9958 |
| realism VLM score | 58.2666 |
| fused realism score | 51.1351 |

r7 说明灯光修复有效，画面已可辨识；但内容仍是低多边形代理场景，真实度和事件可见证据仍不足以作为最终质量结论。

## 5. 已完成的代码与边界修复

本次续训期间完成并验证了以下问题：

1. local Codex strict schema 对嵌套 open map 的处理改为递归闭合；外部 provider 仍保留必要的 map 行为。
2. local codegen 默认使用低 reasoning effort，并允许一次有界的静态 gate repair。
3. codegen prompt 强制要求 `DIRECTOR_PLAN`、`OUTPUT_DIR`、单次 `candidate.blend` 保存、实体 observer metadata 和正确的 `CameraKeyframe` 字段。
4. 静态 gate 捕获 camera primitive 返回对象误用 `.rotation` 的问题；正确字段为 `.frame`、`.location`、`.target`。
5. 正式 agent codegen 增加可见灯光要求，避免生成可执行但不可见的欠曝视频。
6. formal evaluator 配置从旧的 GLM 身份更新为实际已跑通的 external Director + local Codex 身份；dynamic provider gate 已通过。
7. golden bundle 的 manifest 原先指向兄弟工作树；已验证 720 张抽帧逐张 SHA-256 一致，并将 90 个 MP4 归位到当前 bundle，改为 bundle-relative 路径。

相关代码包括：

```text
config/formal-evaluator-v1.json
src/videoact/codex_exec_provider.py
src/videoact/blender_code_agent.py
scripts/train_real_harness.py
scripts/prepare_real_jobs.py
tests/test_blender_code_agent.py
tests/test_codex_exec_provider.py
tests/test_provider_identity_gate.py
```

## 6. Golden review 状态

当前 bundle 已具备完整媒体：

```text
cases       = 30
videos      = 90
sampled png = 720
score rows  = 0
```

人工评分必须由两位独立 annotator 完成，每个视频填写 14 个维度，因此需要 `90 × 2 = 180` 条评分记录。不得读取或修改 `blind_manifest.json` 来推断隐藏实验臂，也不得用模型分数、frame statistics 或合成数据替代人工记录。

盲评页面已启动：

```text
http://127.0.0.1:8765/
```

人工完成后执行：

```powershell
uv run python scripts/validate_golden_review_set.py --root dataset/golden-review-exact-v2
uv run python scripts/finalize_golden_review.py --root dataset/golden-review-exact-v2
```

只有 validator 通过并生成全部 14 维 ICC 后，才能进入 evaluator calibration、G2 paired pilot 和 G3 shadow round。

## 7. 证据索引

| 证据 | 路径 |
|---|---|
| 最新 readiness | `out/preflight/continuation-readiness-20260903.json` |
| 最新全量测试 JUnit | `out/preflight/continuation-full-test-post-lighting-20260903.xml` |
| 最新 capability | `out/preflight/continuation-capability-post-lighting-20260903.json` |
| source fingerprint | `out/preflight/continuation-source-fingerprint-post-lighting-20260903.json` |
| liveness | `out/preflight/continuation-liveness-post-lighting-20260903.json` |
| G0 r7 artifact | `out/preflight/continuation-g0-model-20260903-r7/vbench2-train-01-01/` |
| G0 r7 visual judge | `out/preflight/continuation-g0-model-20260903-r7/vbench2-train-01-01/vlm_report.json` |
| golden validator snapshot | `out/preflight/continuation-golden-validator-20260902.json` |
| golden bundle | `dataset/golden-review-exact-v2/` |

## 8. 后续执行顺序

1. 两位独立 annotator 完成 180 条 golden score rows。
2. 运行 golden validator 和 finalizer，确认 14 个维度的 ICC。
3. 重新生成 readiness，并封存 G0/G1 evidence hashes。
4. 运行 20-case paired pilot（G2）；失败时只允许按协议选择一个 Harness owner。
5. 在不修改 Harness 的前提下运行 60 train + 60 dev shadow round（G3）。
6. 通过 formal release gate 后，才启动六轮 code-level Harness training；每轮最多 5 个 attempt，并保留 append-only training memory。

截至本记录时间点，没有启动 G2、G3 或正式六轮，也没有写入任何人工评分或虚构的 paired 结果。
