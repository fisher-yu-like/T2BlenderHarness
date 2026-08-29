# T2Blendercodeharness Skill Self-Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前已训练的 Director Harness 状态同步到可复用 skill，并用 `trajectory-v4-multi` 的真实历史证据完成一次可审计、单 owner、无数据泄漏的 skill 自进化。

**Architecture:** 保持 Harness 源码、dataset labels、evaluator 公式和既有 Blender artifacts 不变。新增只读的历史记录归一化入口，将已接受的外循环 patch manifest 转换为 proposal JSONL；依据重复 failure 更新最小 skill 章节，再用 capability check、全量 pytest、原数据集回归和 prompt variation 验证。

**Tech Stack:** Markdown/YAML skills, Python 3.11+, JSON/JSONL, existing `trajectory-v4-multi` artifacts, `pytest`, Blender CLI evidence already produced by `D:\blender\blender.exe`.

---

## Scope and frozen inputs

- Current Harness: `h-t2-hard-v4-director-prompt-elliptical-return-order-v1`, commit `306e4d2`.
- Historical baseline: `h-t2-hard-v4-pretraining-baseline`, commit `7fe017a`.
- Dataset: `dataset/trajectory-v4-multi`, validator contract 50 train / 60 dev / 30 frozen test.
- Prior real artifacts: `out/training/multi-five-rounds-v1`.
- External VLM is not required for this skill-only evolution; unavailable results remain unavailable. Existing `assistant_local_review` evidence is not converted into synthetic VLM evidence.
- No changes to `src/videoact`, `evaluator`, dataset manifests/labels, generated Blender plans, or prior output directories.

## Task 1: Establish RED evidence and immutable fingerprints

**Files:**
- Create: `out/skill-self-evolution-v1/preflight.json`
- Test: `tests/test_autodesign_skill_tools.py`

- [x] Run the existing capability check and record its JSON output before the skill update.
- [x] Add a failing test showing that a Director owner is currently mapped to an old/generic skill version or section.
- [x] Run that test and capture the expected failure before editing the proposal tool.
- [x] Hash the current English/Chinese skill files, dataset metadata, evaluator policy, and Harness commit in the final report.

Expected RED evidence: the proposal tool emits `t2blendercodeharness-v1` and does not target a Director-specific section.

## Task 2: Add deterministic historical-record normalization

**Files:**
- Create: `skills/t2blendercodeharness/scripts/build_self_evolution_records.py`
- Modify: `skills/t2blendercodeharness/scripts/propose_skill_update.py`
- Test: `tests/test_autodesign_skill_tools.py`

- [x] Add a test for preserving round number, train case IDs, patch manifest evidence, and normalized Director failure ID.
- [x] Implement the read-only converter over `round-*/patch_manifest.json` and `attempt_report.json`.
- [x] Ignore non-accepted patches, evaluator pretraining corrections, and VLM-unavailable records.
- [x] Add Director owners (`director_prompt_interpreter`, `director_event_scheduler`, `director_trajectory`, `director_camera`, `blender_code_agent`) to proposal section mapping.
- [x] Set proposal metadata to `t2blendercodeharness-v4-director`.
- [x] Run focused tests and then the full suite.

## Task 3: Generate proposal from the previous real training dataset

**Files:**
- Create: `out/skill-self-evolution-v1/historical_records.jsonl`
- Create: `out/skill-self-evolution-v1/proposal.json`

- [x] Convert accepted Director patch history from `multi-five-rounds-v1` into JSONL with source paths.
- [x] Require at least two distinct train cases for one normalized failure.
- [x] Emit a proposal without mutating any skill/source/evaluator/dataset file.
- [x] Inspect that the proposal has one owner, affected cases, evidence paths, current skill hash, and required regression checks.

## Task 4: Apply the smallest approved skill update

**Files:**
- Modify: `skills/t2blendercodeharness/SKILL.md`
- Modify: `skills/t2blendercodeharness-zh/SKILL.md`
- Create: `out/skill-self-evolution-v1/application.json`

- [x] Add the current Harness snapshot and unified Director entry point.
- [x] Add the historical-evidence self-evolution protocol, proposal-only boundary, Director owner taxonomy, and read-only converter command.
- [x] Synchronize the Chinese skill so it no longer advertises the stale parser/evaluator path or stops solely because external VLM is unavailable.
- [x] Record pre/post hashes, proposal ID, applied section, approval source, and the fact that Harness implementation code was untouched.

## Task 5: Regress the skill against the same dataset and a variation

**Files:**
- Create: `out/skill-self-evolution-v1/validation.json`
- Create: `docs/t2blendercodeharness-skill-self-evolution-v1.md`

- [x] Run the skill validator with the available Python runtime.
- [x] Run capability check and full `pytest` with a workspace-local temp directory.
- [x] Validate `trajectory-v4-multi` without changing it.
- [x] Re-audit prior train/dev real artifacts: artifact completeness, deterministic reports, separate task/realism channels, review-source provenance, and retry logs.
- [x] Forward-test Director planning on the original multi-entity prompt and an implicit-intent variation.
- [x] Generate a report containing before/after skill hashes, proposal evidence, regression status, dataset counts, score channels, and a curve of prior Harness rounds.

## Acceptance gates

The skill update is accepted only if:

1. proposal generation is read-only and exactly one owner is targeted;
2. the normalized failure affects at least two distinct train cases;
3. capability check, skill validation, and full tests pass;
4. dataset remains unchanged and train/dev/test boundaries are preserved;
5. artifact gate, Director evidence, VLM-unavailable handling, render retry, and one-owner acceptance rules remain present;
6. prior real artifacts remain complete and no score is fabricated or copied from plan/telemetry;
7. the forward variation still produces a DirectorPlan with evidence, trajectories, interaction lifecycle, and camera visibility obligations.

## Deliverables

- Updated reusable skills: `skills/t2blendercodeharness/SKILL.md` and `skills/t2blendercodeharness-zh/SKILL.md`.
- Reusable scripts: `build_self_evolution_records.py` and updated proposal mapper.
- Proposal and application audit under `out/skill-self-evolution-v1`.
- Chinese final report with tables and curve at `docs/t2blendercodeharness-skill-self-evolution-v1.md`.
