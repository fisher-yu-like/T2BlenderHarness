# Two-Plan Convergence and Completion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对 `2026-08-27-harness-quality-remediation.md` 与 `2026-08-27-agent-codegen-layered-evolution.md` 做逐 task 对账，完成可自动化的缺口，并把必须由人工或本地 Codex 运行证据完成的门禁明确保留为 pending。

**Architecture:** 保持唯一生产链 `exact prompt → DirectorAgent → DirectorPlan → case coverage → BlenderCodeAgent → frozen source → real Blender CLI → artifact/deterministic/oracle → one shared visual review → separate task/realism → outer Harness evolution`。模板只作为显式 baseline；训练只修改一个 Harness owner，不能修改 evaluator、dataset、Blender 模板或已生成 plan。

**Tech Stack:** Python 3.11+, Pydantic 2, pytest, Blender 5.1.2 CLI, Codex structured output, JSONL/Markdown experiment memory, SHA-256 fingerprints。

**Spec:** `E:/2026-08-27-harness-quality-remediation.md`, `E:/2026-08-27-agent-codegen-layered-evolution.md`, and the merged execution plan `docs/superpowers/plans/2026-08-27-harness-remediation-agent-codegen-unified.md`.

## Global Constraints

- Production generation is provider-assisted and fail-closed; deterministic parsing and `template_baseline` are explicit comparison paths only.
- A missing provider, invalid plan, schema/static/coverage failure, missing artifact, or unavailable visual review cannot become a template video or synthetic score.
- The official local inner loop regenerates a complete plan/code/render candidate at most three times for a case; same-source render retry is zero and every failed candidate is retained as evidence.
- Each outer candidate changes exactly one Harness owner; evaluator, dataset labels, Blender template implementation, and frozen test are immutable.
- The active dataset is the verbatim VBench-2.0 index `dataset/vbench2-agent-training-index-v1`: exactly 60 train / 60 dev / 20 frozen test, unique source identities, and no split leakage. `prompt` must equal raw `prompt_en`; self-built or augmented prompt datasets are not training inputs.
- Every implementation task ends with focused tests, cross-case adaptation tests, and the relevant real Blender or local-Codex gate.
- Human golden review and local Codex provider availability are recorded as gates, not silently inferred; the official training path does not call an external endpoint.

### Local-provider and bounded-inner-loop amendment (2026-08-28)

The official execution policy is now `codex-local`: DirectorAgent and
BlenderCodeAgent run through the in-process Codex environment. There is no external endpoint.
A plan/non-compliance, Blender render, or artifact failure creates a
fresh candidate, at most three times (`at most three`); exhausted cases are
`NOT_RENDERED` with every attempt retained. This inner loop is execution
recovery only, not scene repair or Harness evolution. Human visual calibration
(`人工视觉校准`) and the paired agent/template gate remain independent pending
gates because local generation cannot supply independent ground-truth review.

---

### Task 1: Produce the two-plan completion matrix and isolate human gates

**Files:**
- Create: `docs/two-plan-convergence-audit-v1.md`
- Modify: `docs/superpowers/plans/2026-08-27-harness-remediation-agent-codegen-unified.md`
- Test: `tests/test_skill_contracts.py`

**Interfaces:**
- Consumes: both source plans, current repository files, validators, and current test output.
- Produces: a status matrix marking every required task `complete`, `partial`, `pending_human`, `pending_external`, or `not_applicable`; no pending item may be represented by a baseline score.

- [x] **Step 1: Write the failing audit assertion**

Add a test that requires the audit document to contain the two source paths, every status vocabulary value, the active dataset counts, the 1320 upper-bound protocol, and explicit human/provider gates.

- [x] **Step 2: Run the audit test and verify it fails**

Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-audit-red tests/test_skill_contracts.py -k two_plan`.
Expected: FAIL because the audit document does not yet exist.

- [x] **Step 3: Write the matrix**

Map remediation Phase 1–5 and codegen Phase 0–5 to concrete files, tests, and evidence. Mark the double-annotator golden review, real provider pair gate, real VLM smoke, and paired agent/template experiment as pending gates; mark optional text-to-motion and Objaverse work as `not_applicable` to the current proxy objective.

- [x] **Step 4: Run the audit test and adaptive checks**

Run the focused test, the benchmark-only prompt-index validator, and
`uv run python scripts/validate_frozen_eval_set.py --root dataset/frozen-eval-v1`.
All must pass without adding generated scores.

### Task 2: Close third-party frozen-set leakage validation

**Files:**
- Modify: `scripts/validate_frozen_eval_set.py`
- Modify: `tests/test_golden_review_set.py` or create `tests/test_frozen_eval_isolation.py`
- Modify: `docs/two-plan-convergence-audit-v1.md`

**Interfaces:**
- Consumes: `dataset/frozen-eval-v1`, `dataset/vbench-derived-100-v1`, and any historical `trajectory-v4-multi` manifest.
- Produces: a validator that rejects case ID, prompt hash, source identity, and semantic signature overlap with every named training/ablation set.

- [x] **Step 1: Add a failing overlap test**

Copy one frozen record into a temporary historical-set manifest and assert the validator returns a nonzero status with the exact overlap category.

- [x] **Step 2: Run the test and verify the expected failure**

Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-isolation-red tests/test_frozen_eval_isolation.py`.
Expected: FAIL because the validator currently checks only the frozen set's internal uniqueness.

- [x] **Step 3: Implement explicit reference-set arguments**

Add `--reference-root` as a repeatable option, load each reference manifest/metadata, compare all four identity fields, and report `reference_dataset`, `overlap_kind`, and IDs. Keep proposal generation disabled for the frozen set.

- [x] **Step 4: Run cross-case adaptation tests**

Run the new overlap test with prompt-only, case-ID-only, source-only, and semantic-signature-only collisions, then validate the real frozen set against `vbench-derived-100-v1` and `trajectory-v4-multi` when present.

### Task 3: Add an auditable real-example and L4 promotion boundary

**Files:**
- Create: `scripts/validate_codegen_examples.py`
- Create: `scripts/promote_fallback_primitives.py`
- Create: `tests/test_codegen_examples_and_promotion.py`
- Modify: `skills/blender-code-agent/SKILL.md`

**Interfaces:**
- Consumes: `dataset/codegen-examples-v1` records and `code_manifest.jsonl` records from real runs.
- Produces: a validator requiring real artifact paths, code/plan hashes, review provenance, and deterministic/artifact evidence; a promotion report that proposes but never auto-adds L4 primitives.

- [x] **Step 1: Write failing validator/promotion tests**

Require a missing MP4, an artifact-only review, and an unreviewed `new_primitive` occurrence to be rejected; require three distinct real occurrences to produce `promotion_candidate` without editing `blender/lib`.

- [x] **Step 2: Run RED**

Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-examples-red tests/test_codegen_examples_and_promotion.py`.
Expected: FAIL because the validator and promotion report do not exist.

- [x] **Step 3: Implement minimal read-only tooling**

Validate every example's frozen source and required artifacts, reject synthetic/unavailable reviews, count normalized primitive names by distinct case, and emit a JSON report with `promotion_candidate`, evidence paths, required human review, and no source mutation.

- [x] **Step 4: Run adaptation tests**

Exercise simple single-entity, multi-actor handoff, reveal/occlusion, and unknown-primitive manifests. Run the tool against real existing outputs; an empty valid example set must report `pending_external_examples`, not success.

### Task 4: Make codegen context and cache evidence explicit

**Files:**
- Modify: `src/videoact/blender_code_agent.py`
- Modify: `src/videoact/code_cache.py`
- Modify: `scripts/prepare_real_jobs.py`
- Create: `tests/test_codegen_context_and_cache_boundary.py`

**Interfaces:**
- Consumes: DirectorPlan, verified library signatures, and only validated real codegen examples.
- Produces: a codegen request with `context_examples`, a manifest that records example IDs and validation status, and fail-closed behavior when examples are absent or invalid.

- [x] **Step 1: Write the failing context-boundary tests**

Assert invalid examples cannot enter the provider request, a case-specific example with a mismatched plan hash is rejected, and a cache hit preserves the exact source while retaining context provenance.

- [x] **Step 2: Run RED**

Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-context-red tests/test_codegen_context_and_cache_boundary.py`.
Expected: FAIL because context examples are currently optional without validation evidence.

- [x] **Step 3: Implement the boundary**

Load only validator-approved examples, include their IDs/hashes in the request and code manifest, and make missing examples an explicit `context_status=none` rather than silently treating historical templates as few-shot examples.

- [x] **Step 4: Run cross-prompt tests**

Verify two prompts with identical action vocabulary but different entities/layout produce distinct request fingerprints and cannot share a source unless their plan hash and Harness version match.

### Task 5: Add a provider/evaluator readiness report without fabricating scores

**Files:**
- Create: `scripts/check_training_readiness.py`
- Create: `tests/test_training_readiness.py`
- Modify: `skills/t2blendercodeharness-training/SKILL.md`
- Modify: `docs/t2blendercodeharness-agent-training-memory-v1.md`

**Interfaces:**
- Consumes: full test result marker, capability report, dataset validators, golden-review metadata, dynamic-provider preflight, Blender smoke manifest.
- Produces: machine-readable readiness with independent statuses and a human-readable gate summary.

- [x] **Step 1: Write failing readiness tests**

Require readiness to remain blocked when golden review or dynamic provider is pending, even when all automated tests and a template baseline smoke pass.

- [x] **Step 2: Run RED**

Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-readiness-red tests/test_training_readiness.py`.
Expected: FAIL because no readiness report exists.

- [x] **Step 3: Implement the report**

Expose `automated_checks`, `real_blender_smoke`, `golden_review`, `dynamic_agent_provider`, `paired_gate`, and `training_allowed`; use `pending`/`blocked` for unavailable evidence and never write numeric substitutes.

- [x] **Step 4: Run matrix tests**

Test all combinations of one missing gate, all automated gates, and a template-only smoke. Only an explicit all-required-pass fixture may set `training_allowed=true`; current repository state must be false.

### Task 6: Final per-task verification and documentation synchronization

**Files:**
- Modify: `docs/superpowers/plans/2026-08-27-harness-remediation-agent-codegen-unified.md`
- Modify: `docs/harness-architecture-v2.md`
- Modify: `docs/evaluator-v5-calibration.md`
- Modify: `docs/two-plan-convergence-audit-v1.md`

**Interfaces:**
- Consumes: all task reports and test outputs.
- Produces: synchronized status, exact remaining human actions, and a final architecture/experiment handoff; no six-round training invocation.

- [x] **Step 1: Run all focused and adaptation tests**

Run the new tests plus the Director, codegen, coverage, artifact, oracle, evaluator, dataset, and skill-contract suites.

- [x] **Step 2: Run real checks**

Run `D:\blender\blender.exe --version`, evaluate the existing real baseline artifact, and attempt at most one dynamic provider preflight. Record failures as evidence.

- [x] **Step 3: Run final verification**

Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-two-plan-final`, compileall, both dataset validators, capability check, readiness report, and `git diff --check`.

- [x] **Step 4: Update the handoff**

Mark each matrix row from evidence, keep human golden/provider/paired tasks pending, and present the latest Harness flow before any training approval.

## Execution result (2026-08-28)

### Annotation frontend completion amendment (2026-08-28)

The automatic portion of the human-review handoff is now implemented:

- `scripts/golden_review_ui/index.html` is a Chinese blind-review page with
  separate original English prompt and Chinese helper translation, metadata,
  unsaved-change protection, and “保存并下一个”.
- `scripts/golden_review_app.py` exposes only safe metadata/media URLs and
  atomically upserts one case/sample/annotator row.
- `scripts/build_golden_review_set.py` preserves `source_prompt` verbatim and
  records benchmark provenance in the public manifest without exposing arm
  mappings.
- `scripts/finalize_golden_review.py` computes ICC(2,1) and revalidates the
  bundle atomically.

The real bundle is `dataset/golden-review-v1` (30 cases, 90 MP4s), but its
validator intentionally reports `awaiting_human_annotations` until two
independent reviewers create 180 score rows. This task remains
`pending_human`; real provider, agent smoke, and paired-gate tasks remain
`pending_external`. The exact human steps are in
`docs/golden-review-human-handoff-zh.md`.

Tasks 1–6 are complete as an audit/automation pass. The two source plans are
architecturally converged, but the plans' external acceptance gates remain
open by design: double-annotator golden review, a successful real dynamic
provider pair, an agent (not template) Blender smoke, and the 20-case paired
agent/template experiment. The final readiness report is
`out/training_readiness_report.json` with `training_allowed=false`; no
six-round training invocation was made.

## Latest amendment: benchmark-only training input (2026-08-28)

The requirement that training use VBench or another benchmark supersedes the
historical self-authored trajectory dataset as an active training input. The
reproducible active index is `dataset/vbench2-agent-training-index-v1`. It
selects raw records from `data/vbench-source/VBench2_full_info.json`, copies
`prompt_en` exactly, and writes no local event/entity/oracle labels. The
DirectorAgent therefore remains responsible for deriving the executable plan
at runtime.

### Final verification snapshot (2026-08-28)

The fresh full suite reports `395 passed, 28 warnings`; compileall and the
capability check pass. The active benchmark index reports 140 unique records
with `train/dev/test = 60/60/20`, zero prompt/source overlap with all listed
references, and the frozen evaluator reports 20 unique cases with proposal
generation disabled. The review UI was opened through a separate browser
session; all three real MP4 elements reached `readyState=4` with six-second
duration.

The production review bundle is deliberately not marked calibrated:
`dataset/golden-review-v1` has 30 cases and 90 real MP4 files, but all 30 cases
have `render_prompt_mismatch_count=30` because those historical videos were
rendered from the augmented comparison prompt. Readiness therefore reports
`golden_review=pending` with
`golden_bundle_requires_exact_prompt_rerender`; it also reports
`real_blender_smoke=blocked` for template-only evidence,
`dynamic_agent_provider=fail`, and `paired_gate=pending`. The final value is
`training_allowed=false`, so no six-round training has been started.

- [x] Fix per-record split and case-id construction so benchmark dimensions
  cannot inherit split state from a previous family.
- [x] Add benchmark source SHA, exact-prompt, provenance, uniqueness,
  fingerprint, split-count, and reference-overlap validation.
- [x] Make readiness reject every non-benchmark dataset, even when the
  historical self-built validator passes.
- [x] Make the training entry point fail closed before reading splits unless
  the benchmark index validator passes.
- [x] Make every real-training CLI mode require a readiness report with
  `training_allowed=true`; a blocked report creates no round or render output.
- [x] Make prompt-only coverage use the generated DirectorPlan as an internal
  reachability contract without inventing benchmark semantic labels.
- [x] Synchronize the reusable Harness/training skills and architecture
  references with the benchmark-only policy.
- [ ] Keep six-round training blocked until the independent readiness matrix
  is fully passing; benchmark provenance alone is not evidence of a successful
  real Blender/agent/VLM experiment.

## Preparation checkpoint (2026-08-29)

- [x] Render a fresh exact-prompt calibration bundle from the active VBench
  index, restricted to `train` and `dev`; no frozen test prompt is included.
- [x] Render three real comparison arms with `D:\blender\blender.exe`:
  `codex-local`, explicit `template_baseline`, and `direct_code`.
- [x] Verify 30 cases / 90 MP4s and require every available run manifest's
  `prompt_hash` to equal the displayed prompt hash. The fresh bundle reports
  `render_prompt_mismatch_count=0`.
- [x] Add the bounded `run_bounded_outer_attempts` state machine. It accepts
  only `patch`, `accept`, or `stop`; without an explicit one-owner patch it
  stops at `awaiting_harness_patch` instead of repeating attempt 1.
- [ ] Human gate: two independent reviewers must score all 90 blind videos on
  all 14 dimensions and produce ICC evidence. The handoff is
  `docs/golden-review-exact-v2-handoff-zh.md` and the bundle is
  `dataset/golden-review-exact-v2`.
- [ ] After human calibration, rerun the readiness matrix; do not start the
  six-round outer training while `training_allowed` is false.

## User-directed pre-calibration diagnostic amendment (2026-08-29)

The latest instruction is to run the real training evidence first because the
current proxy videos are visibly coarse, and to defer human annotation until
after a Harness upgrade. This does not unlock formal training or permit
numeric visual substitutions. The implementation therefore adds an explicit
diagnostic path rather than weakening the production gate:

- [x] Add `require_diagnostic_training_readiness`: it accepts only a report
  whose automated, benchmark, frozen-eval, real-Blender, and dynamic-agent
  gates pass; only pending `golden_review` and `paired_gate` may remain.
- [x] Add `diagnostic-attempt`, `diagnostic-overall`, and
  `diagnostic-six-rounds` modes. Every diagnostic manifest records
  `diagnostic_precalibration`, `formal_training_allowed=false`, and
  `visual_scores_permitted=false`.
- [x] Add `audit_dynamic_agent_index` before rendering. An explicit
  `template_baseline`, missing per-case codegen provenance, missing source, or
  `all_cases_reuse_one_generated_source` is fail-closed; it cannot produce a
  diagnostic video.
- [x] Keep the existing outer contract: at most five attempts per round,
  10 train + 10 paired dev per attempt, and a separate 60 train + 60 dev
  overall evaluation at round end. Inner plan/code/render regeneration remains
  bounded at three candidates.
- [x] Run the diagnostic rounds with the active verbatim VBench index and
  real `D:\blender\blender.exe`; append every case and missing visual review
  channel to the local Memory table while the run progresses.
- [x] Inspect repeated physical/rig/trajectory evidence and apply at most one
  Harness-owner patch per outer attempt. Do not edit dataset, evaluator,
  Blender library/template, or generated plan artifacts during this phase.
- [ ] After the upgraded Harness produces a new exact-prompt bundle, resume
  the two-annotator human review and rerun formal readiness before claiming
  training acceptance.

## Diagnostic round-1 execution record (2026-08-29)

- [x] Completed the pre-upgrade baseline attempt with 10 train + 10 paired
  dev cases using real `D:\blender\blender.exe`; all 20 MP4s were playable and
  all source hashes were unique, but the visual channel remained unavailable.
- [x] Applied exactly one Harness-owner patch (`blender_code_agent`) in
  `src/videoact/codex_self_provider.py`: case-specific visual profiles,
  higher-detail parametric geometry, a connected armature, event-conditioned
  pose keyframes, and a camera-facing orbit bias. No evaluator, dataset,
  Blender library, or generated plan was changed for the patch.
- [x] Re-ran the paired 10 + 10 cases and the round-end 60 + 60 overall set
  with the upgraded Harness. The overall artifact-only realism proxy moved
  about `33.65 -> 38.55` on train and `33.62 -> 38.47` on dev; geometry moved
  about `70.10 -> 80.32` and `70.04 -> 80.15`, respectively. Deterministic
  pass counts stayed `60/60` on both splits, prompt hashes stayed aligned,
  and the upgraded dev batch had no sampled black frames.
- [x] Added a hard source-content gate: a dynamic source must contain
  `CASE_SCENE_PROFILE` and `codex-local-case-profile-v2`; unique hashes without
  those markers are rejected before Blender. This is provenance evidence, not
  a fabricated visual score.
- [x] Strengthened the source gate so the profile signature must bind to the
  DirectorPlan hash prefix and the recorded source hash must match the actual
  `blender_job.py` bytes. A copied generic scaffold with a fake marker now
  fails before render.
- [x] Continue rounds 2-6 with the upgraded Harness, at most five attempts per
  round and one owner per accepted candidate. Human/VLM visual scores remain
  pending and must not be synthesized from artifact evidence.

## Post-diagnostic semantic audit (2026-08-29)

- [x] Inspect representative real frames and their exact DirectorPlans rather
  than trusting the 100-point deterministic gate. Object-only prompts were
  found to contain an invented `actor_a=person`, which explained the recurring
  generic actor/table appearance.
- [x] Patch the local Director interpretation rule in
  `src/videoact/codex_self_provider.py` so object-only prompts no longer add a
  default person. The patch is Harness-only; it does not modify the evaluator,
  dataset, Blender library, or any historical training artifact.
- [x] Preserve compound object names and map `slide/slides/sliding` into a
  prompt-evidenced executable movement event, so an object-only droplet case is
  not reduced to a static generic `water` prop.
- [x] Add a regression test and run a real Blender two-case smoke for the
  balloon/surface and water-droplet prompts. Both plans now contain only
  prompt-derived props plus the declared support surface, provenance is 2/2,
  both MP4s are non-empty/playable, and no fallback source is used.
- [ ] Before human review, run the exact-prompt review bundle with this latest
  semantic rule and inspect whether the remaining low-detail proxy still shows
  penetration, disconnected-looking limbs, or physically incorrect motion.
