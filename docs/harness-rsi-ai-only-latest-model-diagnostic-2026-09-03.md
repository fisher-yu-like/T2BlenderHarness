---
report_id: harness-rsi-ai-only-latest-model-diagnostic-2026-09-03
project: T2BlenderCode
worktree: harness-rsi
report_date: 2026-09-03
execution_mode: ai_only_diagnostic
provider_mode: model
generation_mode: agent
template_used: false
formal_training_allowed: false
human_calibration_bypassed: false
status: partial_external_vlm_diagnostic
---

# harness-rsi 最新模型 AI-only 诊断记录

## 结论

已按“最新版、不要模板”的要求执行 AI-only 诊断通道。生成链使用：

- Director：`external_openai_compatible:gpt-5.6-luna`
- Blender codegen：`codex_exec_local:codex-cli`
- `generation_mode=agent`
- `provider_mode=model`
- `template_backed=false`
- `fallback_used=false`
- Blender：5.1.2
- 视觉评估策略：`visual-primary-v6`

本报告不是正式人工校准结果，也没有把 AI 分数写入 `human_scores.jsonl`，没有生成虚构的第二位标注者或 paired gate 结果。

## 覆盖范围与结果

历史 golden 清单共 30 个案例（90 个视频），本次先对其中 16 个 train 案例做最新版模型诊断；dev 子集未在本批次进入生成阶段。

| 阶段 | 结果 |
|---|---:|
| train 案例尝试数 | 16 |
| 成功生成最新版模型源的唯一案例 | 12 |
| 生成失败的唯一案例 | 4 |
| 进入视觉评估 staging 的唯一案例 | 10 |
| 首次成功返回 AI 视觉分数的案例 | 2 |
| 模板回退次数 | 0 |
| 写入人工评分行数 | 0 |

4 个生成失败案例均保留 fail-closed 证据，原因包括外部服务 `HTTP 429` 限流、Director 硬不确定性未消除或响应契约校验失败；没有用模板补位。

## AI 视觉评分快照

以下两项是视觉服务在受控重试前成功返回的数值快照，来源为外部 `gpt-5.6-luna`，置信度均为 0.96：

| 案例 | overall VLM | task | realism | 备注 |
|---|---:|---:|---:|---|
| `vbench2-train-01-04` | 36.2534 | 37.2976 | 33.9474 | 外部视觉评分成功 |
| `vbench2-train-01-09` | 47.5239 | 51.0698 | 37.3909 | 外部视觉评分成功 |

随后视觉服务持续返回 `transport_error`。为避免把传输失败误判为低分，后续案例均记为不可用或未完成，不做数值填充；没有重新推导或伪造分数。

## 证据边界

- 最新模型生成证据：`out/diagnostics/ai-only-latest-model-20260903/`
- 视觉评估 staging：`out/diagnostics/ai-only-latest-model-20260903/eval-latest/`
- 人工评分文件：`dataset/golden-review-exact-v2/human_scores.jsonl`
- 正式 readiness：`out/preflight/continuation-readiness-20260903.json`

当前人工评分文件仍为 0 字节；正式 readiness 仍为 `blocked`，`training_allowed=false`，阻塞项为 `golden_review` 和 `paired_gate`。因此本次 AI-only 分数仅用于诊断和问题定位，不用于 patch 选择、训练记忆更新、G2/G3 放行或正式六轮训练。

## 后续断点

外部视觉服务恢复后，可以从 `eval-latest` 的可播放、硬门禁通过案例继续 AI-only 评分；继续执行时仍固定使用 `provider_mode=model`，禁止 `rule_template_baseline` 和任何模板回退。正式训练仍需独立人工校准门禁通过后才能启动。
