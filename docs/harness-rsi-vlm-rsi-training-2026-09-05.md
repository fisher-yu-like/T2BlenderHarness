---
record_id: harness-rsi-vlm-rsi-2026-09-05
project: T2BlenderCode
worktree: harness-rsi
execution_mode: ai_only_vlm_rsi
training_status: prepared_before_launch
formal_training_allowed: false
visual_scores_permitted: true
vlm_required: true
test_schedule: baseline_final_only
---

# Harness-RSI VLM 驱动六轮训练记录与运行手册

## 1. 这次训练要解决什么问题

此前的六轮运行实际上是 `diagnostic_precalibration`：它生成了真实 Blender artifact 和 deterministic/proxy 指标，但关闭了视觉 provider，并使用了“无 Harness patch 控制器”的 round transition。因此它不能回答“VLM 发现了什么问题、Harness 改了什么、修改后 dev 是否变好”。

本次运行改为 `ai_only_vlm_rsi`。VLM 将读取每个合格 case 的真实 `proxy.mp4` 和按时间排序的采样帧，对 prompt compliance、event timing、camera、trajectory、physical realism、motion naturalness、visual presentation 等 14 个维度给出结构化结果和证据帧。train 结果只用于提出 Harness patch；dev 用于非回归；frozen test 只做 round 0 baseline 和 round 6 final 对比。

这是一轮 AI-only research training，不是完成 release gate 的正式人工校准训练。`formal_training_allowed=false` 会保留在协议和总报告中；这不会阻止本次 Harness 研究性演化，但禁止把 VLM 结果冒充 golden-review/paired-gate 结果。

## 2. 固定输入与禁止事项

| 项目 | 本次固定值 |
|---|---|
| train/dev | `dataset/vbench2-agent-training-index-v1` |
| 每轮样本 | 10 train + 10 paired dev |
| 轮数 | 6 |
| frozen test | `dataset/vbench2-agent-test-100-v1`，100 cases |
| test 运行 | round 0 baseline、round 6 final；round 1–5 不运行 |
| Director/codegen | external Director + local Codex codegen，`provider_mode=model` |
| VLM provider | `CodexVisualReviewProvider` |
| 请求模型 | `gpt-5.6-luna` |
| 人工评分 | 不调用 |
| 模板/fallback | 禁止进入训练和评分 |
| patch 目标 | 仅 Harness code，原则上限定 `src/videoact` |

禁止把以下内容作为 VLM 或训练分数：

- `deterministic_video_proxy_metrics`；
- geometry/PNG artifact-only proxy；
- Director plan、Blender source、telemetry 推断出的“视觉成功”；
- 缺少真实视频/采样帧的默认分；
- 模板场景、缓存 source 或 fallback response；
- 任何人工评分文件的空值填充或模型模拟。

## 3. 产物清理范围

旧训练生成的视频已从以下目录清理：

```text
D:\harness-rsi-training\**\*.mp4
```

清理结果：删除 107 个生成 MP4，剩余 0 个。删除范围不包含：

- `dataset/` 中的原始输入视频；
- 仓库源代码；
- 已有 JSON、Markdown、transition 和失败日志；
- 新运行启动后产生的文件。

新运行使用独立目录，不复用旧 round/case artifact：

```text
D:\harness-rsi-training\vlm-rsi-six-rounds-20260905
```

## 4. 真实评分链路

```text
case prompt
  -> external Director
  -> local Codex Blender codegen
  -> Blender 5.1.2
  -> candidate.blend / trusted observer / runtime evidence
  -> proxy.mp4 + chronological PNG frames
  -> CodexVisualReviewProvider
  -> VLMJudgeResponse
  -> real_unified_score.json
```

VLM 请求只包含 exact prompt、采样帧路径、帧时间线和需要观察的事件区间。它不能读取 Blender source、Director plan 或 deterministic result 来代替视觉观察。每个维度都需要 `evidence_refs` 指向实际检查过的 frame filename；低置信度或不完整 evidence 不会成为 patch acceptance 的正向证据。

每个 case 的 `vlm_report.json` 必须记录：

```text
status
review_source=codex_local_visual_review
vlm_model/model_alias
confidence
dimension_evidence
event_scores
sampled_frames
raw response/call provenance
```

## 5. 六轮执行协议

### Round 0：VLM baseline test

在任何 patch 前运行 100 个 frozen test cases，记录 baseline task/visual/realism 和 VLM availability。baseline 只用于最终对照，`patch_selection_excluded=true`。

### Round 1–6：VLM-RSI train/dev 循环

每轮执行：

1. 生成并评估 10 个 train case。
2. 生成并评估 10 个 paired dev case。
3. 从 train 中提取有 VLM frame evidence 的低分维度和 deterministic failures。
4. 归并重复 root cause，选出一个 Harness owner。
5. 产生一个 bounded patch proposal；只能修改允许的 Harness owner 文件。
6. 用 `PatchExecutor` 保存 parent hash、diff hash、patch manifest，并支持 rollback。
7. 重新检查 train/dev 的 VLM score、deterministic gate、provenance 和失败率。
8. dev 不回归才接受 patch；否则拒绝/回滚并记录原因。

第 1–5 轮不跑 test。第 6 轮 patch gate 结束后运行 100 个 final test。test 绝不进入 patch proposal、owner attribution 或 patch acceptance。

## 6. 训练中的分数含义

| 字段 | 作用 |
|---|---|
| `deterministic_score` | artifact/runtime 硬约束；不是视觉分 |
| `task_final_score` | VLM 对 prompt/event 完成度的视觉判断 |
| `visual_score` | VLM 对画面清晰度、细节、呈现的判断 |
| `realism_score` | 有独立视觉 review 时的融合分 |
| `realism_score_kind` | 必须区分 `independent_review_fused` 与 `artifact_only_proxy` |
| `vlm_scored_count` | 实际返回合格 VLMJudgeResponse 的 case 数 |
| `review_source` | 必须可区分 VLM、proxy、human 和 unavailable |

如果 VLM provider 不可用，case 标记为 `unavailable`，该 case 不进入均值、不产生接受 patch。不能用 local frame statistics 伪装成 VLM 分数。

## 7. Patch 闭环

```text
train VLM evidence
  -> FailureExtractor / owner attribution
  -> OuterTransitionController
  -> one-owner proposal
  -> PatchExecutor
  -> source diff/hash/manifest
  -> train + dev re-evaluation
  -> accept or rollback
```

允许的 owner 包括 Director contract、Blender codegen、inner-loop recovery、outer controller、patch impact、case coverage、liveness 和 failure extraction。禁止改变 evaluator、observer、dataset、test policy 和 VLM judge 来“修正分数”。

每次 patch 必须满足：

- 来源是 train，而不是 dev/test；
- owner 唯一且能被 cross-owner 校验；
- affected files 通过 Harness path gate；
- 有可复现的 failure evidence；
- 有 parent/diff hash；
- patch 后 dev 不低于非回归阈值；
- 失败时保留 rollback evidence。

## 8. 运行命令

代码发布后使用以下入口启动：

```powershell
Set-Location C:\Users\sy\Desktop\T2BlenderCode\.worktrees\harness-rsi
uv run python scripts/run_vlm_rsi_six_rounds.py
```

可覆盖的关键参数：

```powershell
uv run python scripts/run_vlm_rsi_six_rounds.py `
  --output-root D:\harness-rsi-training\vlm-rsi-six-rounds-20260905 `
  --vlm-model gpt-5.6-luna `
  --provider-mode model `
  --test-schedule baseline_final_only `
  --workers 2
```

启动前必须先写入 `six_round_protocol.json`；如果 VLM provider 构造失败，入口非零退出，不进入无 VLM 的训练。

## 9. 关键记录文件

```text
six_round_protocol.json
round-00/test_report.json
round-01..06/round_report.json
round-06/test_report.json
memory/harness_updates.jsonl
outer_transitions.jsonl
patch_manifest.json
six_round_training_report.json
```

最终文档要汇总每轮 train/dev 的 VLM scored count、task/visual/realism 均值、unavailable 数量、patch 状态、dev gate、rollback 和 test exclusion。

## 10. 验收清单

- [ ] `visual_scores_permitted=true`。
- [ ] `vlm_required=true`。
- [ ] 至少一个真实 case 的 `review_source=codex_local_visual_review`。
- [ ] `vlm_scored_count > 0`。
- [ ] train VLM evidence 进入 outer controller。
- [ ] 至少有一次真实 patch proposal，或有明确、可审计的 `no_patch` 原因。
- [ ] patch 只修改 Harness owner 路径。
- [ ] dev non-regression 执行并记录。
- [ ] 只有 round 0/6 有 test report。
- [ ] 无模板/fallback 伪造视频或评分。
- [ ] formal release gate 仍保持未通过状态，未冒充人工校准结果。

## 11. 发布顺序

当前远程：

```text
origin = git@github.com:fisher-yu-like/T2BlenderHarness.git
```

发布步骤：

1. 完成 VLM-RSI 文档、代码和回归测试。
2. 排除运行输出、pytest 临时目录和日志，核对 staged file list。
3. 运行 targeted tests、full unit tests、`git diff --check`。
4. 提交最新代码。
5. 使用 `git push --force-with-lease origin HEAD:main` 覆盖远端 main。
6. 用 `git ls-remote` 核对远端 main SHA。
7. 仅在推送成功后启动新训练。

