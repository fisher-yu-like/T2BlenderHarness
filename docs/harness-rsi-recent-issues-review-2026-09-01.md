# harness-rsi 最近问题复盘

更新时间：2026-09-01  
分支：`harness-rsi`  
基线：`2087d7a`（`baseline-glm-5.3-flash-v1`）

## 1. 结论摘要

最近一次从 GLM-5.3-flash 训练实验继续推进时，问题并不集中在单一模型调用，而是分布在完整链路的多个边界：

~~~text
VBench prompt
  -> Director 解释
  -> Blender codegen / cache
  -> Blender CLI + trusted observer
  -> artifact / hash / geometry gate
  -> real video VLM review
  -> RSI proposal / readiness gate
~~~

已确认并修复的主要代码缺陷包括：

- Windows 默认 `cp936/GBK` 解码 Blender UTF-8 输出，触发 `UnicodeDecodeError`，随后导致 reader thread 的 `IndexError`。
- Blender 版本字段直接截取 stdout 前缀，结果把启动 warning 误记成版本号。
- Director、codegen、provenance 和 artifact gate 对非 ASCII prompt 使用了不一致的 JSON canonicalization，导致哈希互相不一致。
- handoff 轨迹缺少完整的 `attach -> transfer -> detach` 生命周期。
- 内循环评估遗漏 dataset fingerprint，导致 experiment fingerprint 不完整。
- 外层 attempt 和同计划重生成共用冻结 cache，导致合法的重生成被误判为 source conflict。
- GLM Director 的默认 reasoning effort 被误设为 `low`；codegen 应为 `low`，Director 应保持 `high`。
- 视觉评审 fallback 没有继承主模型的 `visual_frame_budget`，导致同一个 case 的评审证据预算不一致。

仍未解决、且会阻止正式六轮训练的主要问题是：

- Codex CLI 视觉评审通道持续超时，当前 batch 没有 video-scored case。
- 正式 readiness report 的 `golden_review` 和 `paired_gate` 仍为 pending，因此 `training_allowed=false`。
- 已生成的历史 experiment artifacts 保留了旧代码产生的错误版本字段和旧评审状态，不能仅靠修改代码自动变成新证据。

## 2. 当前可验证状态

### 2.1 测试和能力检查

2026-09-01 最后一次验证结果：

~~~text
618 passed, 3 skipped, 34 warnings
capability_check: pass
git diff --check: pass
~~~

Pillow 的 34 条 warning 来自 `Image.Image.getdata` 弃用提示，不是测试失败。

### 2.2 真实 Blender case

对已有 case `vbench2-train-01-01` 重新运行几何审计后：

- `geometry_raw.audit_available=true`
- mesh 数量：4
- `geometry_report.hard_gate_failed=false`
- Blender 本地可执行文件版本：`5.1.2`

这证明 Blender 几何审计路径现在可以正常执行，但它不能替代视觉 VLM 评审。

### 2.3 当前训练门禁

`out/training_readiness_report.json` 当前状态：

| Gate | 状态 | 原因 |
|---|---|---|
| `full_test` | pass | 全量测试通过 |
| `capability` | pass | Harness 能力检查通过 |
| `dataset` | pass | benchmark prompt index 合格 |
| `frozen_eval` | pass | frozen evaluation 边界可用 |
| `real_blender_smoke` | pass | agent 真实 Blender smoke 有完整 artifact |
| `dynamic_agent_provider` | pass | Director/codegen provider 身份和 fail-closed 边界通过 |
| `golden_review` | pending | `golden_review_bundle_missing` |
| `paired_gate` | pending | `paired_gate_report_missing` |

所以当前不能宣称正式训练已获准，即使自动化测试全部通过。

## 3. 已遇到并修复的问题

### 3.1 Blender 输出按 GBK 解码

症状：

- Windows Python 默认编码为 `cp936`。
- Blender 输出中包含 UTF-8 字节，例如 prompt 或路径中的非 ASCII 内容。
- `subprocess.run(..., text=True)` 使用系统默认编码读取时出现：

~~~text
UnicodeDecodeError: 'gbk' codec can't decode byte 0x82 ...
~~~

- 由于 Python subprocess reader thread 已经失败，随后还会出现 `IndexError`，让表面症状看起来像渲染器或 worker 崩溃。

受影响的路径：

- `scripts/render_proxy_jobs_parallel.py` 的 trusted observer 子进程。
- `scripts/render_proxy_jobs_parallel.py` 的实际 Blender 渲染子进程。
- `src/videoact/blender_adapter.py` 的 CLI backend。
- `scripts/evaluate_proxy_realism.py` 的 Blender 几何审计子进程。

修复：所有需要把 Blender stdout/stderr 当文本读取的调用显式使用：

~~~python
encoding="utf-8",
errors="replace",
~~~

并增加了 subprocess 参数回归测试。历史 `runner.stderr.log` 仍然会保留旧运行产生的 GBK traceback；旧日志不会被修复代码改写。

相关文件：[render_proxy_jobs_parallel.py](../scripts/render_proxy_jobs_parallel.py)、[blender_adapter.py](../src/videoact/blender_adapter.py)、[evaluate_proxy_realism.py](../scripts/evaluate_proxy_realism.py)。

### 3.2 Blender 版本字段被启动 warning 污染

症状：旧的 `run_manifest.json` 中，`blender_version` 不是 `5.1.2`，而是类似下面的 stdout 开头：

~~~text
00:00.953 reports | WARNING Path 'D:\blender\5.1\datafiles\...'
~~~

根因是渲染成功分支直接保存了 `completed.stdout[:120]`，假定 stdout 前缀一定以 Blender 版本开头。实际 Blender 启动时可能先打印资产路径 warning。

影响：

- 运行清单中的环境证据不可靠。
- 依赖 Blender version 的 experiment fingerprint 被污染。
- 旧 artifact 不能直接作为干净的正式实验环境证据。

修复：新增 `_extract_blender_version()`，按行在 stdout、stderr 和 observer tail 中提取 `Blender x.y.z`，现在应记录 `5.1.2`。旧 manifest 仍需重新渲染或单独重建证据，不能静默覆盖。

相关文件：[render_proxy_jobs_parallel.py](../scripts/render_proxy_jobs_parallel.py)、[test_real_batch_discovery.py](../tests/test_real_batch_discovery.py)。

### 3.3 非 ASCII prompt 导致 plan hash 分歧

症状：包含弯引号、中文或其他非 ASCII 字符的 prompt，在不同阶段得到不同的 plan hash。典型错误包括：

- `case_profile_not_bound_to_director_plan`
- `director_plan_hash_mismatch`

根因是不同组件使用了不同 JSON 序列化方式：

- `DirectorPlan.content_hash()` 使用了默认 ASCII escaping。
- codegen payload 和 provenance canonical hash 使用 `ensure_ascii=False`。
- Director contract 修复后，artifact gate 仍然沿用旧的默认 escaping，于是再次拒绝同一份 plan。

影响：Round-2 的两个 dev case（`02-14 fills cup`、`02-19 brushes jacket`）因 prompt 含弯引号而失败；修复发生在这两个 case 的内循环预算消耗之后，预算无法回收。

修复：

- `a412c9f`：统一 Director plan content hash 的 canonicalization。
- `1a89e25`：让 artifact gate 使用同一 canonicalization。
- 增加包含非 ASCII 内容的 hash 回归测试。

### 3.4 handoff 轨迹缺少 detach

症状：handoff-only prompt 在 deterministic interaction evaluator 中失败，即使场景看起来已经完成交接。

根因：轨迹只记录了接收者 attach/transfer，没有记录交付者在边界处释放对象的 detach。评估契约要求完整生命周期：

~~~text
giver attach -> transfer -> giver detach -> receiver attach/continue
~~~

修复：`9c53d1c` 在 handoff 边界补齐 release/detach 事件，并增加轨迹回归测试。

### 3.5 内循环丢失 dataset fingerprint

症状：真实评估的 experiment fingerprint 缺少 dataset fingerprint，最终退化成 hard artifact failure。这个问题会让每个 case 都带着不完整的环境绑定信息，即使 Blender 本身执行成功也不能形成可信证据。

根因：`train_real_harness.py` 的 inner-loop evaluate callback 调用 `evaluate_real_run()` 时没有传入 dataset fingerprint。

修复：`9c53d1c` 补齐传递，并通过 fingerprint/real evaluator 相关测试。

### 3.6 外层 attempt 共用 code cache

症状：Attempt-2 对同一个 plan 重新生成 source 时，被 frozen-source guard 判定为跨 case/source reuse conflict，导致 prepare 阶段批量失败。

根因：外层 attempt 之间复用了同一个冻结 cache namespace。模型对相同 plan 重新生成不同 source 是允许的，但 cache guard 把不同 attempt 的 source 当成了同一槽位的非法覆盖。

修复：`aac67f7` 让 cache namespace 包含 outer attempt；累计 overall evaluation 也使用独立 namespace。

### 3.7 同计划重生成仍会撞冻结槽位

症状：即使 outer attempt 已隔离，inner loop 对同一 plan 的合法重生成仍可能命中旧冻结槽位，导致所有后续 prepare 失败。

根因：`store()` 的契约要求调用者在 source 变化时版本化 slot，但 prepare path 没有执行版本化。

修复：`1276a10` 在检测到冲突时版本化 code-cache slot，并保留每一次候选的 provenance。

### 3.8 GLM Director 的 reasoning effort 回归

症状：全套测试中出现唯一失败：Director provider 的默认 reasoning effort 为 `low`，而现有延迟/质量契约要求 Director 为 `high`。

根因：为了降低 codegen 延迟新增的 `low` 默认值错误地落到了 `GLMStructuredProvider.for_director()`；正确拆分应为：

- Director：`high`，负责复杂 prompt 解释和事件链推理。
- Blender codegen：`low`，负责按已经冻结的 DirectorPlan 生成受约束 source。

修复：Director 恢复 `high`，codegen 保留 `low`，并保留对应测试。

### 3.9 visual fallback 没有保持帧预算

症状：主视觉模型使用自定义 `visual_frame_budget`（例如 3 或 6）失败后，fallback provider 会退回默认预算 8。这样同一个 case 的主模型和 fallback 实际看到的证据集合可能不同。

影响：评审结果不可直接比较，且增加了 fallback 隐式改变证据边界的风险。

修复：fallback 构造时显式传递 `self.visual_frame_budget`，并增加“主模型失败后 fallback 仍使用同一预算”的回归测试。

相关文件：[codex_visual.py](../evaluator/codex_visual.py)、[test_harness_rsi_glm_codex.py](../tests/test_harness_rsi_glm_codex.py)。

## 4. 尚未解决的运行与环境问题

### 4.1 Codex CLI 视觉评审持续超时

当前 real batch 已完成真实 Blender 视频生成，但没有完成视觉评分。对一个已有 case 的局部复核中：

- primary model `gpt-5.6-terra` 在 90 秒内超时。
- fallback model `gpt-5.6-luna` 在 45 秒内超时。
- 独立的 `codex exec ... Return exactly OK.` 短探测在 30 秒内也超时。

对应 `vlm_report.json` 状态为：

~~~text
status: unavailable
review_source: codex_local_visual_review
provider_kind: codex_exec_visual_review
~~~

后果：

- official-v2 Round-1 已生成 10 个真实视频，但 `video-scored cases=0`。
- `mean final score=None`，不能把 artifact-only realism 均值 `32.4141` 当成视觉分数。
- batch 状态保持 `incomplete_local_visual_review`，不能进入正式 RSI patch selection。

处理原则：VLM unavailable 必须保持 unavailable；deterministic score、geometry score、frame statistics 或默认分数均不能复制成 VLM 分数。

### 4.2 两种本地视觉评审路径都有人工/环境依赖

当前存在两条合法路径，但都不是自动无条件可用：

1. `assistant_local_review`：为每个 case 生成 review request，等待明确的 Codex/人工 JSON 评审。
2. `codex-cli`：由本地 Codex CLI 读取 exact sampled frames 并返回 schema-constrained JSON。

之前的 assistant-local runner 在等待 review 文件，当前 Codex CLI runner 又遇到进程超时。因此不能通过改变 status 文本或补写数值来“完成”评审。

### 4.3 外部 VLM endpoint 的合规/可用性问题

项目记录显示当前外部 VLM 环境指向未获批准的 proxy/tenant 配置。按照 real-run protocol，这类调用只能记录为 unavailable，不能作为正式评审证据。后续需要配置合规且可验证的 endpoint，或者完成 Codex local visual review。

### 4.4 Blender 视频编码依赖 host-side

当前连接的 Blender 环境缺少可直接使用的 FFmpeg video encoding 能力，因此视频编码由 host-side 流程处理。这个不是单个 case 的逻辑错误，但会增加：

- Blender render 与 MP4 编码之间的边界。
- MP4 playable probe 的必要性。
- 真实视频 artifact 不完整时的 fail-closed 分支。

这条环境约束必须继续写入 run manifest 和 release evidence，不能把“渲染成功”直接等同于“可评估视频成功”。

### 4.5 历史 artifact 与当前代码不一致

已有实验目录包含旧代码生成的内容：

- 旧 `run_manifest.json` 中的错误 Blender version 前缀。
- 旧的 `UnicodeDecodeError` runner log。
- `awaiting_assistant_review` 或 `unavailable` 的旧 VLM report。

这些文件对历史追踪有价值，但不能与修复后的新运行混为同一正式 evidence set。要形成新实验，需使用新 output root 或显式重新渲染并重建所有依赖 artifact 的报告。

## 5. 训练协议和工程治理问题

### 5.1 readiness 不是分数

曾经容易把“有一个 numeric score”误认为 gate 已通过。当前 gate 明确要求来源、身份、完整 artifact 和独立报告：

- 数值不能替代 provenance report。
- Boolean 不能替代 gate evidence。
- VLM unavailable 不能替代 review score。
- template baseline 不能替代 agent smoke。

这也是为什么当前测试通过仍不能启动正式训练：`golden_review` 和 `paired_gate` 证据尚未建立。

### 5.2 test100 必须与 patch selection 隔离

六轮协议要求每轮后运行 frozen test100，但 test100 只能用于 round-end evaluation，不能进入：

- issue localization；
- owner choice；
- Harness patch proposal；
- patch acceptance。

这条边界已在 protocol 和测试中固化，后续恢复训练时不能为了“提高分数”把 test evidence 传回 transition callback。

### 5.3 动态 agent 失败必须 fail closed

旧的 table-driven/template 路径容易让生成失败被一个看似可运行的通用场景掩盖。当前约束是：

- 没有结构化 provider 时返回 `hard_uncertainty`。
- codegen 失败时不偷偷回退到固定模板。
- 每个 source 必须绑定 DirectorPlan、provider provenance 和 code hash。
- 真实 Blender/observer/artifact 证据缺失时，case 不得进入正式评分。

这会让失败数量变多，但能避免把“模板能运行”误报为“agent 学会了”。

### 5.4 工作树和实验目录污染

当前 `harness-rsi` worktree 同时包含大量既有 tracked modifications、实验报告、`.pytest-tmp*` 临时目录和未跟踪脚本。它们属于不同阶段的工作产物，若不区分会带来：

- 无法准确判断某个修改属于哪个实验。
- 误把临时报告或旧 output 当成正式 evidence。
- 重新运行时误复用历史 cache、review 或 round report。

本次没有执行 destructive cleanup，也没有 reset/checkout 用户已有修改。后续应为每次正式训练使用独立 output root，并在提交前单独整理临时测试目录和实验产物。

## 6. 问题与修复索引

| 问题 | 主要根因 | 状态 | 依据 |
|---|---|---|---|
| Blender stdout GBK 解码 | Windows 默认编码与 Blender UTF-8 输出不一致 | 已修复 | renderer/adapter/geometry tests |
| Blender version 被 warning 污染 | 直接截取 stdout 前缀 | 已修复代码；旧 artifact 待重建 | `_extract_blender_version` test |
| 非 ASCII plan hash mismatch | `ensure_ascii` canonicalization 不一致 | 已修复 | `a412c9f`, `1a89e25` |
| handoff 缺 detach | interaction lifecycle 不完整 | 已修复 | `9c53d1c` |
| dataset fingerprint 丢失 | inner-loop callback 未传递 | 已修复 | `9c53d1c` |
| outer attempt cache collision | cache namespace 未隔离 | 已修复 | `aac67f7` |
| 同 plan 重生成冲突 | frozen slot 未版本化 | 已修复 | `1276a10` |
| Director reasoning 默认值错误 | Director/codegen 延迟策略混用 | 已修复 | GLM provider tests |
| VLM fallback 帧预算漂移 | fallback 使用默认 budget | 已修复 | fallback budget test |
| Codex CLI 视觉评审超时 | 本地 CLI/endpoint 当前不可用 | 未解决 | primary/fallback/short probe timeout |
| golden review 缺失 | readiness evidence 未建立 | 未解决 | `golden_review_bundle_missing` |
| paired gate 缺失 | paired report 未建立 | 未解决 | `paired_gate_report_missing` |
| 历史 artifacts 与新代码不一致 | 旧运行不会自动重写 | 未解决 | official-v2 old reports/logs |
| worktree 临时产物混杂 | 多轮实验共用工作目录 | 需治理 | `git status --short` |

## 7. 建议的后续顺序

1. 先修复或确认 Codex CLI 的认证、endpoint、model 和 local image-inspection 可用性，用单 case 完成一次真实 JSON visual review。
2. 重新运行 readiness，补齐 `golden_review` 和 `paired_gate`，直到 `training_allowed=true`。
3. 使用新的 experiment root 重新生成受旧解码/版本字段影响的 Round-1 evidence，不直接覆盖历史目录。
4. 确认 train/dev 的 visual review 全部完成后，再启动 Round-2 及后续六轮；每轮 test100 仍只做独立评估。
5. 完成一轮后检查 `run_manifest`、provider provenance、plan/code/blend/MP4 hash 和 geometry report，再生成可提交的训练记忆文档。
6. 正式提交前清点 `.pytest-tmp*`、临时报告和未跟踪脚本，明确哪些是源码、哪些是实验 evidence、哪些应被忽略。

## 8. 关键参考

- [六轮训练计划](six-round-training-plan-v7.md)
- [真实运行协议](real-run-protocol.md)
- [GLM 训练记录 Round-1/2](glm-flash-training-report-round1.md)
- [最近一次 official-v2 训练报告](../out/training/glm-flash-six-rounds-official-v2/GLM-5.3-flash-Harness-RSI-training.md)
- [readiness report](../out/training_readiness_report.json)
- [Blender 渲染器](../scripts/render_proxy_jobs_parallel.py)
- [Codex 视觉 provider](../evaluator/codex_visual.py)

