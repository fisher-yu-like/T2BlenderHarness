# 两份方案融合审计 v1

> **Current handoff (2026-08-28):** automated implementation is advanced, but
> training remains blocked. `dataset/golden-review-v1` is now a 30-case/90-real-
> MP4 blind bundle served by `scripts/golden_review_app.py`; it still requires
> two independent annotators and `scripts/finalize_golden_review.py`. The active
> training dataset is the exact VBench index
> `dataset/vbench2-agent-training-index-v1`, not the historical augmented
> comparison dataset. Real provider, agent smoke, and paired-gate evidence are
> also still external pending gates.

审计日期：2026-08-28  
工作树：`codex/director-multi-entity-harness`  
来源方案：

- `E:/2026-08-27-harness-quality-remediation.md`
- `E:/2026-08-27-agent-codegen-layered-evolution.md`
- 合并执行版：`docs/superpowers/plans/2026-08-27-harness-remediation-agent-codegen-unified.md`

## 结论

两份方案的生产架构已经融合：唯一生产路径是

```text
exact prompt
→ DirectorAgent
→ DirectorPlan / evidence / event order / trajectory / camera cue
→ case coverage gate
→ BlenderCodeAgent
→ frozen per-case source
→ real Blender CLI
→ artifact + deterministic + independent oracle
→ one shared visual review
→ separate task / realism channels
→ outer Harness evolution
```

代码、数据契约、fail-closed、source freeze、artifact hash、独立 oracle、评分通道和六轮协议已经有自动化实现。方案尚未达到“所有实验结果已完成”的状态，原因不是用模板代替了 agent，而是以下外部证据必须真实存在：双人 blind golden review、可返回结构化结果的真实 Codex Director/BlenderCodeAgent 成对调用、真实 VLM 或可审计人工视觉审查，以及 20-case agent/template paired gate。

状态定义：

| 状态 | 含义 |
|---|---|
| `complete` | 代码和自动化测试/验证证据均已存在 |
| `partial` | 核心代码存在，但原方案要求的真实 smoke、人工视觉或完整实验尚未完成 |
| `pending_human` | 必须由人工标注/审阅，不能由代码推断 |
| `pending_external` | 依赖真实 provider、VLM 或外部数据，当前证据不可用 |
| `not_applicable` | 原方案中的可选扩展不属于当前 proxy Harness 主线 |

## Remediation 方案对账

| 原方案部分 | 当前实现/证据 | 状态 | 未完成事项 |
|---|---|---|---|
| Phase 1 Task 1 真实 VLM dispatch | `evaluator/vlm_providers.py`、`evaluator/visual_primary.py`、lowercase model policy、unavailable 语义和相关测试 | `pending_external` | 真实 `gpt-5.6-luna`/`gpt-5.6-terra` 返回一次可审计多帧结果 |
| Phase 1 Task 2 双人 golden review | `scripts/build_golden_review_set.py`、`scripts/validate_golden_review_set.py`、review UI | `pending_human` | 30–50 case、每 case 两名标注者、14 维评分、ICC/α |
| Phase 1 Task 3 frame-only fallback | `evaluator/visual_evidence.py`、`evaluator/realism.py`、patch gate 边界测试 | `complete` | 历史报告的 supersede 标记持续保留 |
| Phase 1 Task 4 calibration | `scripts/calibrate_evaluator.py`、`docs/evaluator-v5-calibration.md` | `pending_human` | golden bundle 完成后计算相关系数、bootstrap CI、agreement |
| Phase 2 Task 5 falsifiable proposal | `PatchProposal` prediction fields、proposal tests、capability check | `complete` | 真实 accepted candidate 仍需训练实验产生 |
| Phase 2 Task 6 attribution/rollback | `src/videoact/patch_attribution.py`、outer/meta harness、rollback tests | `complete` | 真实 refuted patch 证据仍待训练产生 |
| Phase 2 Task 7 k-rollout | seed fingerprint、same-source retry/cache contracts、multi-rollout tests | `complete` | 主实验仍固定 k=1；k=2 是独立稳定性实验 |
| Phase 3 Task 8 skeleton/skin | `blender/character_rig.py`、`blender/lib/rigging.py`、rigging tests | `partial` | walk/reach/grasp/release 的真实视频和人工确认 |
| Phase 3 Task 9 attachment/handoff | `blender/lib/constraints.py`、interaction lifecycle、oracle 和 handoff tests | `partial` | 真实手骨 Child Of influence 交叉与 carry 无独立 location keyframe的视觉确认 |
| Phase 3 Task 10 CameraPlan | `blender/camera_dsl.py`、camera library、orbit/follow/reveal/visibility tests | `partial` | 真实复杂 case 中完整 cue、occlusion、continuity 的视觉证据 |
| Phase 3 Task 11 render quality | Principled/material/light/detail audit、render-quality tests、真实 baseline artifact | `partial` | 三 case 的 512×512/16–32 samples 成本冻结和人工画质确认 |
| Phase 3 Task 12 text-to-motion | 未接入外部 motion provider | `not_applicable` | 当前主线使用可审计程序化动作；若启用需单独批准新 provider |
| Phase 3 Task 13 asset retrieval | 未接入 Objaverse/AssetRetrieval3D | `not_applicable` | 当前目标是 proxy Harness，不把资产检索混入本轮训练 |
| Phase 4 Task 14 dynamic prompt parsing | `DirectorAgent.from_provider`、`CodexExecProvider`、strict schema、evidence/uncertainty gate | `pending_external` | 真实 provider 成功返回并通过一例复杂 prompt；deterministic parser 只做 explicit baseline |
| Phase 4 Task 15 frozen third-party set | `dataset/frozen-eval-v1`、validator、proposal-independent tests | `complete` | 已对 `vbench-derived-100-v1`、`trajectory-v4-multi` 和 active training set 完成 case/prompt/source/semantic 四类零泄漏比较 |
| Phase 4 Task 16 three-arm/four-arm ablation | 历史产物、ablation scripts/UI、报告 schema | `pending_external` | 可信视觉 review 下重新评分并报告旧/新 evaluator 差异 |
| Phase 5 Task 17 reusable function library | `blender/lib/*`、signatures、library tests | `complete` | 真实 agent examples 进入后再增加 usage evidence |
| Phase 5 Task 18 executable skill evolution | skill update proposal、owner mapping、memory and skill tests | `complete` | 只能从真实 accepted evidence 产生后续 skill proposal |

## Agent Codegen 方案对账

| 原方案部分 | 当前实现/证据 | 状态 | 未完成事项 |
|---|---|---|---|
| Phase 0 evaluator/执行前置 | artifact/deterministic/oracle、真实 Blender baseline smoke | `partial` | golden、provider、paired gate 未全部通过 |
| L2 Task 1 geometry/rigging | `blender/lib/geometry.py`、`rigging.py`、contracts/tests | `complete` | 逐函数真实 Blender smoke 已覆盖的范围要继续记录 |
| L2 Task 2 camera primitives | `blender/lib/camera.py`、camera tests | `complete` | 真实复杂镜头视觉 gate 仍属 Phase 3 partial |
| L2 Task 3 layout/interaction | `blender/lib/layout.py`、constraints、layout tests | `complete` | handoff/penetration 的人工视频确认仍未完成 |
| L2 Task 4 metadata/signatures/usage | `__meta__.py`、`signatures.json`、export tests | `complete` | usage count 需要真实 codegen run 才有生产数据 |
| L3 Task 5 schema contract | `codegen_contracts.py`、strict provider schema normalization | `complete` | 真实 provider 返回仍需 external gate |
| L3 Task 6 per-case codegen | `blender_code_agent.py`、AST/runtime/coverage gates | `pending_external` | 至少一个真实 Codex 生成的 source 需要成功冻结并渲染 |
| L3 Task 7 few-shot examples | `validate_codegen_examples.py`、`codegen_context.py`、真实 reviewed bundle 目录 | `pending_external` | 当前没有真实 reviewed plan+MP4 bundle；需 provider 生成、真实渲染和人工/合规视觉 review 后再加载 |
| L3 Task 8 generate-once-freeze | `code_cache.py`、manifest code hash、retry mutation gate | `complete` | 真实 agent source 进入 cache 后再做 paired evidence |
| L4 Task 9 fallback gate | `fallback_codegen.py`、strict escalation and fail-closed tests | `complete` | 未有真实 `library_insufficient` case 的外部 provider 证据 |
| L4 Task 10 primitive promotion | 原方案要求统计 ≥3 次后候选晋升；`promote_fallback_primitives.py` 只读统计工具 | `partial` | 真实案例达到阈值后仍需人工审批、签名/docstring/测试/副作用审查，当前不得改 L2 |
| Phase 4 Task 11 agent/template paired | template arm 与 agent path 隔离，protocol/report schema 已有 | `pending_external` | 20 个 train case 的真实 agent source、相同 seed、可信视觉评分 |
| Phase 4 Task 12 L4 coverage experiment | L4 contract/test scaffolding exists | `pending_external` | 10 个模板无路径 prompt 的真实 agent/L4 执行与 ≥50% gate evidence |
| Phase 4 Task 13 regen | version/cache/hash tests | `complete` | 真实 library version bump paired run仍需外部 provider |
| Phase 5 Task 14 owner taxonomy | `MetaHarnessOptimizer`、owner mapping、one-owner tests | `complete` | 真实 candidate 才能填入 accepted/refuted history |
| Phase 5 Task 15 docs/skills | Harness、Director、BlenderCodeAgent、training skills and architecture docs | `complete` | 本审计与 readiness report同步加入 |

## 不可由自动化替代的人工/外部门禁

这些项目故意保留为 `pending`，不能用模板 baseline、artifact-only realism、frame statistics 或 VLM unavailable 替代：

1. **Golden review：** 人工看匿名真实视频/帧，不看 arm、commit、Harness 版本或候选分数；两人独立完成 14 维评分。
2. **真实 provider pair gate：** `CodexExecProvider` 必须成功返回 Director interpretation 和 Blender code response；失败时只留下 `director_failed`/`codegen_failed`。
3. **真实视觉 review：** 只能使用外部 `gpt-5.6-luna`、`gpt-5.6-terra`、可审计 `codex_local_visual_review` 或 `human_review`。
4. **Agent/template paired gate：** 不能因为模板 baseline 可渲染就宣称 agent 能力通过。
5. **L4 primitive promotion：** ≥3 次出现只产生候选报告，签名、docstring、测试、副作用和人工审核通过后才允许进 L2。

## 当前自动化证据

- 全量 pytest：`395 passed, 28 warnings`，JUnit 证据为
  `out/two-plan-convergence/full-test.xml`。
- capability check：`pass`。
- active dataset：`dataset/vbench2-agent-training-index-v1`，60 train / 60 dev / 20 frozen test；140 条 prompt 均来自 VBench-2.0 原始 `prompt_en`，不含本地编写的实体、事件或 oracle 标签。
- benchmark index fingerprint：`196f71f7bf9deed54b2c4cd0b0f943e3122fff01114b3a84b2c1b400e0b43332`；与 frozen 和历史参考集的 prompt/source identity 重叠均为 0。
- `trajectory-v5-agent-codegen`、`vbench-derived-100-v1` 与 `trajectory-v4-multi` 仅作历史/对照数据；readiness 和训练入口均拒绝它们。
- frozen set：20 个唯一 prompt、semantic signature 和 source identity；对三个参考集
  的四类重叠均为 0；proposal/patch selection 禁止；当前 fingerprint 为
  `8a7210bef81945439ac593713276523520b0d1abe6079b2a8568faa8da3b1b4e`。
- Blender：`D:\blender\blender.exe`，版本 5.1.2。
- 真实 template baseline smoke：MP4、480 帧、`.blend`、telemetry、3 张 PNG、artifact hash 完整；它只证明执行层，不证明 agent 能力。
- 真实动态 Codex 预检：schema 规范化与 UTF-8 读取问题已修复，但本机 provider 会话未在时限内返回，故保留 `director_failed`，未生成 code、未换模板。

## 本审计新增的自动化收口（2026-08-28）

- `scripts/validate_frozen_eval_set.py` 现在接受重复的 `--reference-root`，并对四类身份（case ID、prompt hash、source identity、semantic signature）逐类报告重叠；旧 frozen set 与 VBench-derived 集合的实际重叠已通过重建清除。
- `scripts/validate_codegen_examples.py` 现在要求真实 artifact、`director_plan.json`、与计划内容一致的 `plan_hash`、与源文件一致的 `code_hash` 以及合规 review provenance；hash 不匹配、模板、artifact-only review 均拒绝。
- `scripts/prepare_real_jobs.py` 只加载 validator 通过的 few-shot context；manifest 缺失记录 `context_status=none`，manifest 存在但无效则在 provider 调用前写 `codegen_failed/invalid_codegen_context`，绝不使用旧模板；cache/job manifest 保留 context ID/status。
- `scripts/check_training_readiness.py` 将 full test、capability、dataset、frozen-eval、agent smoke、golden、dynamic provider、paired gate 分开裁决。只有全部为 `pass` 才能 `training_allowed=true`，数字、模板 smoke、VLM/provider unavailable 都不能替代证据。
- `scripts/validate_benchmark_prompt_index.py` 校验 source SHA、逐条 prompt 等于 raw `prompt_en`、benchmark provenance、split/fingerprint、唯一 source identity 和参考集重叠；`scripts/train_real_harness.py` 在读取 splits 之前执行同一类 benchmark-only 门禁。

## 融合判定

融合判定为：**架构融合完成，自动化修复大部分完成，实验完成条件尚未满足。**

在 golden/provider/paired gates 通过之前，训练 readiness 必须为 `training_allowed=false`。只有这些证据出现后，才可以执行六轮协议：每轮最多 5 次，每次 10 train + 10 paired dev，每轮结束完整 60 train + 60 dev，理论最大 1320 次真实视频执行；每条记录写入 append-only Markdown/JSONL Memory。

## Final verification snapshot (2026-08-28)

`395 passed, 28 warnings` is recorded in the current JUnit report. The active
VBench index is valid (`140` unique verbatim prompts; `60/60/20` splits) and
the frozen set is valid (`20` cases, no reference overlap). The Chinese review
page is available at `http://127.0.0.1:8766/`; its three real MP4 controls were
verified loaded in an isolated browser session. The page warns that the
current historical comparison bundle is `仅比较用途` because its render
prompt differs from the displayed source prompt.

The remaining gates are intentionally not hidden: exact-prompt agent rerender,
two-person visual calibration, successful dynamic provider pair, agent-only
Blender smoke, and the 20-case paired gate. Readiness is
`training_allowed=false`; no training result is claimed.
