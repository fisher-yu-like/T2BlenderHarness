# T2Blendercodeharness Skill 自进化实验记录 v1

## 结论

本轮完成了“最新 Harness → 可复用 skill → 历史真实证据驱动的 skill 自进化”闭环。

最终保留的 skill 更新不是运行时自动改写，而是由上一轮真实 Blender 外循环训练记录生成 proposal、检查重复 failure 后，在用户明确要求下应用的最小文档更新。更新没有修改 Harness 源码、数据集标签、evaluator 公式或既有 Blender 计划。

核心新增规则是：Director 必须先构建 evidence-backed event order，再应用 generic carry/handoff 匹配；必须保留 reveal、subjectless clause、pause、handoff 和 return 的顺序。若 prompt 含 reveal 但生成的 plan 没有显式 reveal event，必须在 Blender 编译前产生 hard Director finding 并停止。

## 1. 固定输入

| 项目 | 固定值 |
|---|---|
| 当前 Harness | `h-t2-hard-v4-director-prompt-elliptical-return-order-v1` |
| 当前 commit | `306e4d2` |
| 训练前基线 | `h-t2-hard-v4-pretraining-baseline` / `7fe017a` |
| 历史数据集 | `dataset/trajectory-v4-multi` |
| 数据集划分 | 50 train / 60 dev / 30 frozen test |
| 历史真实产物 | `out/training/multi-five-rounds-v1` |
| Blender | `D:\blender\blender.exe` |
| 外部 VLM | 本轮未调用；保留 assistant-local review 的来源标记 |

数据集 validator 结果为 `pass`，共 140 条，fingerprint 为
`4d0abd0eb387bc58c4b4cd03259874670a5c7010ecbc484f056cbf3b255a0f41`。本轮没有重写或重新生成该数据集。

## 2. 执行流程

```text
历史 round patch_manifest.json + attempt_report.json
 → 只读归一化为 historical_records.jsonl
 → repeated-failure proposal
 → 单一 owner 审核
 → 更新最小 skill 章节
 → capability check + skill validator + pytest
 → 原数据集 artifact 回归
 → Director 原 prompt/variation forward-test
 → 汇总文档与曲线
```

### 2.1 RED 阶段

先运行旧 proposal 工具的压力测试。旧行为把 proposal 版本标为
`t2blendercodeharness-v1`，并不能将 `director_prompt_interpreter` 映射到明确章节；测试按预期失败。

随后补充了两个回归测试：

- Director owner 必须落到 `Director prompt interpretation and event scheduling`；
- 双语 skill 必须声明 evidence-backed event order、generic carry 和 subjectless 约束。

### 2.2 Green 阶段：工具和 skill 更新

新增只读工具：

`skills/t2blendercodeharness/scripts/build_self_evolution_records.py`

该工具读取历史 `round-*/patch_manifest.json` 与 `attempt_report.json`，保留 round、train case、evidence path 和 patch provenance，过滤 evaluator pretraining correction 与 VLM unavailable，不修改任何源文件。

更新：

- `skills/t2blendercodeharness/scripts/propose_skill_update.py`
  - 新增 Director owner 映射；
  - proposal version 更新为 `t2blendercodeharness-v4-director`。
- `skills/t2blendercodeharness/SKILL.md`
  - 同步当前 Harness 版本、Director 唯一入口和自进化流程；
  - 增加 event-order hard guard。
- `skills/t2blendercodeharness-zh/SKILL.md`
  - 同步最新架构、评分边界、VLM fallback、retry 和 self-evolution 规则；
  - 清除了旧的 parser/evaluator 入口描述。
- `skills/director-agent/SKILL.md`
  - 同步相同的 event-order hard guard，使 proposal 的 owner 与实际组件 skill 一致。

## 3. 历史证据与 proposal

从上一轮五轮训练产物中归一化出 20 条记录，来源是第 4、5 轮已接受的 Director patch：

| 字段 | 结果 |
|---|---|
| normalized failure | `implicit_event_order_not_preserved` |
| owner | `director_prompt_interpreter` |
| category | `event_order` |
| severity | `error` |
| affected train cases | 20 个不同 case：`multi-train-031` 至 `multi-train-050` |
| proposal 数量 | 1 |
| mutation policy | proposal-only；生成期间未修改 skill/source/evaluator/labels |
| target section | Director prompt interpretation and event scheduling |

历史 patch 的原始 manifest 中部分自然语言字段存在旧编码损坏，因此归一化工具使用稳定 failure ID 和可读的规范化 message；原始 manifest 仍作为 evidence path 保留，没有把损坏文本伪装成新观察结果。

## 4. 应用记录

应用记录位于：

`out/skill-self-evolution-v1/application.json`

| 项目 | 结果 |
|---|---|
| 是否应用 | `true` |
| 应用依据 | 用户明确要求更新 skill 并执行自进化 |
| 修改 owner | `director_prompt_interpreter` |
| English skill SHA-256 | `b51f5db6f6d0ddcdaf653edf4e24eddf31cc9eb809046754606e38362679ba42` → `37e9728d07dacae67cd75d8ea8fdab4b04a0c3427e51c42f1adc29509377207e` |
| Chinese skill SHA-256 | `63f4295df7c472e8fda71d2ac4ccfe77139810d462cc5c3fa1c0b0419796bb16` → `1ac22e14b988cf583b43b3fe1acd6bfa3cc649176690a12ff7502a2510f9eaa1` |
| DirectorAgent skill SHA-256 | 当前值 `f0dfb4f61acbaf2177379e82f9ac1843661a179480c9739d1d1596735c03142b` |
| Harness source 是否修改 | 否 |
| evaluator 是否修改 | 否 |
| dataset 是否修改 | 否 |
| generated plan 是否修改 | 否 |

## 5. 回归验证

| 验证项 | 结果 |
|---|---|
| `quick_validate.py skills/t2blendercodeharness` | pass |
| `quick_validate.py skills/t2blendercodeharness-zh` | pass |
| `capability_check.py --project-root .` | pass，全部检查通过 |
| 全量 pytest | `205 passed` |
| `trajectory-v4-multi` validator | pass，50/60/30，无 leakage |
| 历史 round-05 train+dev artifact | 110/110 有 `proxy.mp4`、manifest、deterministic、geometry、realism、visual evidence |
| 历史 deterministic | 110/110 pass |
| 历史 render attempt | 110/110 成功，0 retry |
| proposal postflight | 1 个单 owner proposal，版本 `t2blendercodeharness-v4-director` |

历史真实 artifact 的审计范围是 `round-05/overall/real`，只做读取和一致性检查，不重新覆盖旧视频。

## 6. Director forward-test

### 正常 prompt

输入：

`Alice carries the red cube while Bob carries the blue cup, then Alice hands the red cube to Bob.`

结果：4 entities、3 events、4 trajectories、2 shots、7 evidence、1 uncertainty，plan hash：
`e7b764cc7607a78c06d15bf05f80ecf9faddcbbda365a0ef8e1e147ec19dfea1`。

### subjectless variation

输入使用省略主语的 handoff/pause/return 顺序：

`Alice carries the red cube to Bob; hands the red cube to Bob. Bob pauses; returns the red cube to Alice, and Alice places the red cube.`

结果：3 entities、5 events、3 trajectories、5 shots、8 evidence、1 uncertainty，plan hash：
`8c00abccd023cac1f7df186b582235077b9322d43512fa9a1b4d44cbab911b1a`。该 variation 通过。

### reveal precedence probe

输入同时包含 reveal 和 carry：

`Alice reveals the red cube, then carries the red cube to Bob; hands the red cube to Bob. Bob pauses; returns the red cube to Alice, and Alice places the red cube.`

当前 deterministic parser 的实际输出为 3 entities、5 events，事件动作是
`carry → handoff → pause → return → place`，缺少显式 `reveal`。新的 skill guard 正确触发 `hard_guard_triggered=true`，因此处理为“停止 Blender 编译并路由到 `director_prompt_interpreter`”，而不是把不完整 plan 当成成功。

这项结果揭示的是 Harness runtime 的待修复问题；按照本轮范围没有修改 `src/videoact`，避免把“skill 规则更新”与“Harness 代码修复”混在一起。

## 7. 训练曲线

![T2Blendercodeharness skill self-evolution curves](C:/Users/sy/Desktop/T2BlenderCode/.worktrees/director-multi-entity-harness/out/skill-self-evolution-v1/skill-self-evolution-curves.png)

曲线来自上一轮真实五轮训练的 cumulative report。R1–R3 的旧 report 没有暴露 Director plan 均值，因此图中对应 Director 曲线保持空值，没有用 0 填充；R4–R5 暴露为 100。Task、realism 和 artifact completion 均按已有报告读取，不重新合成缺失值。

| Round | train task | dev task | train realism | dev realism | artifact completion |
|---:|---:|---:|---:|---:|---:|
| 1 | 65.6393 | 62.8292 | 59.2108 | 56.6084 | 100% |
| 2 | 65.6393 | 62.8292 | 59.2157 | 56.6084 | 100% |
| 3 | 66.2084 | 62.8292 | 58.9371 | 56.6084 | 100% |
| 4 | 67.8926 | 65.7122 | 58.7411 | 57.6930 | 100% |
| 5 | 67.9486 | 65.7122 | 58.0018 | 57.6992 | 100% |

这张图反映的是 Harness 历史训练曲线，不是 skill 文档修改本身产生了新的视频分数；本轮 skill 更新的验收重点是规则完整性、proposal 可审计性和 forward-test 的 fail-closed 行为。

## 8. 产物索引

- Skill：`skills/t2blendercodeharness/SKILL.md`
- 中文 Skill：`skills/t2blendercodeharness-zh/SKILL.md`
- Director skill：`skills/director-agent/SKILL.md`
- 计划：`docs/superpowers/plans/2026-08-27-t2blendercodeharness-skill-self-evolution.md`
- 历史 records：`out/skill-self-evolution-v1/historical_records.jsonl`
- proposal：`out/skill-self-evolution-v1/proposal.json`
- proposal pre-application manifest：`out/skill-self-evolution-v1/proposal_pre_application_manifest.json`
- application：`out/skill-self-evolution-v1/application.json`
- validation：`out/skill-self-evolution-v1/validation.json`
- 曲线：`out/skill-self-evolution-v1/skill-self-evolution-curves.png`
