# Harness-RSI VLM 驱动训练设计

**日期：** 2026-09-05  
**工作树：** `harness-rsi`   
**远程仓库：** `git@github.com:fisher-yu-like/T2BlenderHarness.git`  
**目标分支：** `main`

## 1. 目标

把当前只生成 deterministic/proxy 证据的诊断运行，改成真正的 VLM 驱动 Harness-RSI：VLM 读取真实 Blender 生成的 MP4 和采样帧，给出结构化视觉评分与证据；train 评分进入失败归因和 patch 选择；dev 负责非回归；只有通过约束、测试和 dev gate 的 Harness patch 才能进入下一轮。

本次运行是 AI-only research run。它允许 Harness 代码演化，但不宣称通过正式 release gate，因为 golden review/paired gate 仍需要独立人工校准。任何缺少 VLM 证据的 case 都保持 `unavailable`，不会转成 0 分、proxy 分或模板分。

## 2. 当前问题与修复目标

当前 `diagnostic-six-rounds` 路径存在两个使 RSI 失效的开关：

1. `diagnostic_only=True` 时把 `effective_visual_provider` 设为 `None`，并把 `assistant_local` 设为 `True`。
2. 协议写入 `visual_scores_permitted=false`，并使用 `continue_scheduled_round_without_patch`，每轮只记录 evidence，不执行 Harness patch。

新路径必须显式写入：

```text
execution_mode          = ai_only_vlm_rsi
visual_scores_permitted = true
vlm_required            = true
patch_controller        = outer_transition_controller
patch_executor          = patch_executor
test_schedule           = baseline_final_only
```

`formal_training_allowed` 仍为 `false`，以区分 AI-only 研究训练和完成正式人工校准后的 release training。

## 3. 数据与隔离边界

| 项目 | 固定值 |
|---|---|
| train/dev 数据集 | `dataset/vbench2-agent-training-index-v1` |
| 每轮新增批次 | 10 train + 10 paired dev |
| 训练轮数 | 6 |
| frozen test 数据集 | `dataset/vbench2-agent-test-100-v1` |
| baseline test | round 0 |
| final test | round 6 |
| 中间轮 test | 不运行 |
| test 是否可进入 patch selection | 永远否 |
| Director/codegen | external Director + local Codex codegen，`provider_mode=model` |
| VLM | `CodexVisualReviewProvider`，请求模型 `gpt-5.6-luna` |
| 模板 | 禁止；`template_backed=false`、`llm_generated=true` 必须通过 provenance gate |
| 人工评分 | 不调用；VLM 评分不是人工评分的替代 release 证据 |

生成结果必须写入全新的运行根目录，例如：

```text
D:\harness-rsi-training\vlm-rsi-six-rounds-20260905
```

训练重置只删除 `D:\harness-rsi-training` 内的生成 `.mp4`。不删除数据集原始视频、仓库源代码、JSON evidence 或训练协议记录。

## 4. 评分链路

```text
Director
  -> case-specific Blender codegen
  -> Blender render / trusted observer / runtime evidence
  -> proxy.mp4 + sampled PNG frames
  -> CodexVisualReviewProvider
  -> VLMJudgeResponse (14 dimensions + evidence refs)
  -> real_unified_score.json
  -> train failure extraction / patch proposal
```

VLM 只能读取真实视频和采样帧，不能读取 source、Director plan、telemetry 或 deterministic score 来推断视觉成功。每个维度都必须包含可回溯的 frame filename evidence；VLM 不可用、低置信度、schema 错误或 artifact 不完整时，case 标记为 `unavailable`/`needs_human_review`，不进入可接受 patch 的正向分数。

保留并区分以下通道：

- `deterministic_score`：运行时和 artifact 硬约束。
- `task_final_score`：VLM 对 prompt/event 的语义视觉判断。
- `visual_score`：VLM 视觉呈现通道。
- `realism_score`：有独立视觉 review 时的融合分；仅 geometry/PNG 时只能是 `artifact_only_proxy`。
- `vlm_scored_count`、`review_source`、`confidence`：必须出现在汇总报告中。

`deterministic_video_proxy_metrics` 只用于诊断，不能被转换为 `VLMJudgeResponse`，不能写入 `task_final_score`，也不能单独驱动 patch acceptance。

## 5. 六轮 RSI 状态机

### Round 0：baseline

在任何 Harness patch 前，对 100 个 frozen test case 运行 VLM baseline。baseline 只记录结果，不能进入 train patch selection。

### Round 1–6：训练循环

每一轮执行以下顺序：

1. 运行本轮 10 个 train case 的真实 Director → codegen → Blender → observer → VLM 链路。
2. 运行本轮 10 个 paired dev case 的同一链路。
3. 只把 train evidence 输入 `OuterTransitionController`。
4. 聚合重复 failure，选择唯一 owner 和 root cause；VLM 维度低分只有在具备 frame evidence 且能映射到 Harness owner 时才可成为 actionable finding。
5. 生成一个最多一个 owner 的 Harness patch proposal；禁止修改 dataset、evaluator、observer、test policy 和生成视频。
6. 由 `PatchExecutor` 应用 patch，并保存 parent hash、diff hash、patch manifest 和 rollback evidence。
7. 运行 patch 后的 train/dev non-regression gate：deterministic hard gate、VLM availability、task/visual score、失败率和 template/provenance gate 均不得越过安全阈值。
8. 通过才把 patch 作为下一轮 parent；失败则 rollback，并保留失败证据。

第 1–5 轮不运行 frozen test；第 6 轮完成 patch gate 后运行 final test。test 始终设置 `patch_selection_excluded=true`。

## 6. Harness patch 范围

允许的 patch owner 仅限 `src/videoact` 内已经定义的 Harness 组件，例如：

- `director.py` / `director_contracts.py`：prompt/event/plan 结构约束；
- `blender_code_agent.py` / `codex_exec_provider.py`：case-specific codegen、schema 和 bounded repair；
- `real_inner_loop.py` / `outer_loop.py` / `outer_controller.py`：重试、失败归因和状态转换；
- `patch_executor.py` / `patch_impact.py`：patch 作用域、回滚和影响验证；
- `case_coverage.py` / `liveness.py` / `failure_extractor.py`：证据覆盖和失败抽取。

禁止 patch：`dataset/`、`evaluator/`、`blender/trusted_observer.py`、`tests/`、frozen test 输入、人工评分、VLM 原始响应和 generated output。测试代码可以由开发者增加回归测试，但不能作为训练 patch 的目标文件。

## 7. 运行记录

每个 case 至少保留：

```text
run_manifest.json
deterministic_report.json
observer_report.json
geometry_report.json
proxy.mp4
sampled frame PNGs
vlm_report.json
real_unified_score.json
```

每轮保留：

```text
round_report.json
attempt_report.json
outer_transitions.jsonl
patch_manifest.json (若发生 patch)
memory/harness_updates.jsonl
```

训练总报告必须汇总每轮 train/dev 的：case count、VLM scored count、mean task score、mean visual score、mean realism score、unavailable count、patch status、rollback status、dev gate 和 test exclusion。

## 8. 失败处理

- VLM provider 不可用：停止 patch acceptance，case 为 `unavailable`；不退回 assistant-local proxy 评分。
- MP4/采样帧不完整：不调用 VLM或将结果标为 unavailable；不猜分。
- VLM schema 不完整：记录原始错误，不进入 score aggregation。
- 只有 artifact proxy 分数：可用于诊断，但不能作为 semantic task score。
- patch 触碰冻结路径：`OuterTransitionController` 和 `PatchExecutor` 双重拒绝。
- train 有改善但 dev 退化：拒绝 patch 或 rollback。
- 连续两次无效 patch：写入 stagnation evidence，停止该轮，不重复生成相同 patch。

## 9. 验收标准

新运行必须同时满足：

1. `six_round_protocol.json` 的 `visual_scores_permitted=true`、`vlm_required=true`。
2. 至少一个成功 case 的 `vlm_report.json` 中 `review_source=codex_local_visual_review`，并且 `vlm_scored_count > 0`。
3. train VLM evidence 能进入 outer controller；不是仅写报告。
4. 至少产生一次真实 Harness patch proposal；若无满足证据的可接受 patch，必须明确记录 `no_patch` 原因。
5. patch 只修改允许的 Harness owner 路径，并具有 diff hash/rollback 记录。
6. dev non-regression gate 在 patch 后执行。
7. round 0 和 round 6 有 test；round 1–5 没有 test 目录或 test report。
8. 所有视频来自 case-specific LLM codegen，`template_backed=false` 且没有 fallback 伪造结果。
9. 正式 release gate 仍清楚标记为未通过，不把 AI-only 结果伪装成人工校准结果。

## 10. Git 发布顺序

1. 在 `harness-rsi` 工作树完成文档、代码和测试。
2. 排除训练输出、`.pytest-*`、日志和临时目录，检查 staged file list。
3. 运行 targeted tests、全量 unit tests、`git diff --check`。
4. 创建提交，记录 VLM-RSI 模式、测试结果和运行文档。
5. 用 `git push --force-with-lease origin HEAD:main` 覆盖远端 `main`；若 lease 因远端变化失败，先重新 fetch/核对，不使用盲目 force。
6. 推送成功后才启动新训练，避免训练运行在未发布代码上。

