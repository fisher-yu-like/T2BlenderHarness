---
name: t2blendercodeharness
description: Use when a user asks to turn text into Blender proxy scenes or videos, run scene/trajectory/camera planning, execute Blender MCP or CLI, evaluate artifacts or VLM frames, repair failures, evolve a Harness from train/dev evidence, or package and validate the T2Blendercodeharness workflow as a reusable skill.
---

# T2Blendercodeharness

Apply this skill to operate a contract-first Text-to-Blender proxy Harness. Preserve the separation between Codex Host, MetaHarnessOptimizer, DesignHarness components, Dataset, and Evaluator.

**REQUIRED TRAINING SUB-SKILL:** When the request includes Harness training, fixed train/dev rounds, full-train scoring, real Blender video evaluation, VLM scoring, or Memory, use `t2blendercodeharness-training` and its `scripts/train_real_harness.py` entry point.
**REQUIRED DIRECTOR SUB-SKILL:** For prompts with implicit intent, multiple actors/props, handoffs, concurrency, or camera choreography, use `director-agent` first. DirectorAgent is the only external planning entry point; legacy SceneContract/TrajectoryPlan objects are compatibility projections.

**Core principle:** evidence gates every promotion. A plan is not a render, telemetry is not a complete artifact, a deterministic pass is not a VLM score, and a proposal is not an accepted Harness patch.

### Human review UI and dataset audit

The Chinese blind-review frontend is `scripts/golden_review_ui/index.html`,
served by `scripts/golden_review_app.py`. It displays the verbatim VBench
prompt separately from its Chinese helper translation, anonymized real MP4s,
and the 14 visual dimensions. The active production bundle is `dataset/golden-review-exact-v2`; it needs two distinct annotator IDs for every
one of its 90 videos. The older `golden-review-v1` is historical
comparison-only.
Run
`scripts/finalize_golden_review.py` to calculate ICC(2,1)
and then `scripts/validate_golden_review_set.py`. This calibration bundle is
not the training dataset: training remains gated to
`dataset/vbench2-agent-training-index-v1`, whose prompt/source/fingerprint
validator must pass first.

The primary blind payload is `primary-blind-v1`: it contains only the exact
prompt, chronological real-video frames/timecodes and the response schema. It
excludes the DirectorPlan, scene contract, deterministic findings, owner, arm
and Harness version. A local Codex visual review is an explicit diagnostic
path with no external endpoint; deterministic frame statistics can never be
relabeled as that review.

## Current implementation snapshot

Use the latest tested Harness contract as the reference implementation:

- Harness version: `t2blendercodeharness-v5-executable-director` (working tree;
  executable rig/camera/evaluator boundary is tested in the current branch).
- The pre-training comparison point is `h-t2-hard-v4-pretraining-baseline` at
  commit `7fe017a`.
- `DirectorAgent` is the only planning entry point. It combines prompt/entity
  interpretation, event scheduling, per-entity trajectories, and multi-target
  camera choreography, then projects compatibility `SceneContract` and
  `TrajectoryPlan` objects.
- The current accepted Director fixes preserve evidence-backed reveal,
  handoff, pause, and return ordering. Do not reintroduce a separate legacy
  parser/planner path or bypass the Director evidence fields.

### Current agent-codegen boundary

The formal model path is `DirectorAgent -> BlenderCodeAgent -> real Blender
CLI`, selected explicitly with `provider_mode=model`. Each case records an
external structured Director call and a separate local `CodexExecProvider`
Blender-code call in `provider_manifest.json`. `CodexLocalProvider`/`codex-local` is retained as the
explicit `rule_template_baseline` diagnostic arm; it cannot satisfy readiness.
The external Director boundary reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`
and calls the OpenAI-compatible `/v1/chat/completions` path. The Blender-code
boundary invokes the local `codex exec` command. They are separate provider
instances and their call records must never be merged. Both boundaries normalize Pydantic schemas for strict output (all
object properties required, nullable optional values, tuple arrays expressed
through `items`, and closed objects). Provider, schema, JSON, or session
errors remain hard failures. The `BlenderCodeAgent` composes only the
verified `blender/lib` signatures, runs static and case-coverage gates, then
freezes source by `plan_hash + harness_version`. Within a passing candidate,
`generate-once-freeze` keeps render execution tied to that exact source.

Few-shot codegen context is a separate audited input. Use
`scripts/validate_codegen_examples.py` before loading
`dataset/codegen-examples-v1`; each example must bind a real MP4/artifact,
`director_plan.json`, `plan_hash`, `code_hash`, deterministic evidence, and an
eligible review provenance. A missing manifest is `context_status=none`; a
present invalid manifest fails the case before the Blender code provider is
called. Context IDs and status are retained in the job/cache manifests.

`rule_template_baseline`/`template_baseline` is an explicit historical
comparison arm. It is never an agent fallback. A missing provider, invalid
DirectorPlan, codegen/schema error,
coverage failure, or source mutation during retry is fail-closed and produces
no agent video. In the official local path, `max_inner_attempts=3` means that
a plan/code/render/artifact failure creates a fresh case candidate; after
three failures the case is `NOT_RENDERED`. Same-source render retry is zero,
so a failed candidate cannot silently become a different provenance class.

The Director must not add a default human to an object-only prompt. If the
exact prompt describes only a droplet, balloon, liquid, surface, container, or
other non-human subject, the generated DirectorPlan and runtime telemetry must
contain no invented actor; a neutral support surface is allowed only as an
explicit staging assumption. This is checked before code generation and is a
Harness semantic gate, not a visual-quality guess.

The active six-round training data is the verbatim benchmark index
`dataset/vbench2-agent-training-index-v1` (60 train, 60 dev, 20 frozen test).
Its `prompt` is byte-for-byte the local VBench-2.0 `prompt_en`; it contains no
locally authored event/entity/oracle labels. `DirectorAgent` creates the
executable plan at run time. The historical `trajectory-v5-agent-codegen`,
`vbench-derived-100-v1`, and `trajectory-v4-multi` datasets remain useful only
for comparison/validator regression tests and are ineligible for training.

### Director event-order invariant

Build the evidence-backed event order graph before applying generic carry or handoff
matching. Preserve prompt order across `then`, `after`, `while`, reveal,
subjectless clauses, pauses, and return actions; a generic `carry` match must
not erase an already-established reveal, handoff, or return event. Validate
this with at least one subjectless handoff and one reveal-to-return variation
before compiling Blender code. If the prompt contains a reveal but the plan
has no explicit reveal event, emit a hard Director finding and stop before
Blender compilation.

## Component handoff

Use the existing component skills when installed:

- **REQUIRED SUB-SKILL:** Use `director-agent` for evidence-backed prompt interpretation, event scheduling, multi-entity trajectories, and multi-target camera choreography.
- **REQUIRED SUB-SKILL:** Use `scene-contract` for validating the Director projection and legacy single-entity compatibility.
- **REQUIRED SUB-SKILL:** Use `trajectory-planner` for validating projected entity states, camera shots, and event observability.
- **REQUIRED SUB-SKILL:** Use `blender-proxy-executor` for controlled MCP/CLI execution and run manifests.
- **REQUIRED SUB-SKILL:** Use `harness-evolution` for failure aggregation and train/dev acceptance.

If a component skill is unavailable, use the project modules with the same responsibility. Never bypass a missing contract or evaluator by writing ad-hoc Blender code.

For the active scoring contract, read `docs/evaluator-v5-calibration.md` and
`docs/harness-architecture-v2.md`; older evaluator/calibration documents are
historical snapshots only. Formal provider/judge identity is frozen in
`config/formal-evaluator-v1.json`, and the primary blind payload is
`primary-blind-v1`. Formal release additionally requires sealed G0--G3 reports
from `scripts/check_formal_release_gates.py`; a bare `training_allowed=true`
value without those report hashes is not sufficient.

## Run the pipeline

1. Discover the project root, Python runtime, dataset split, available Blender backend, and existing run state. Prefer the project's Python 3.11+ runtime; do not assume the shell's `python` is compatible.
2. Run `DirectorAgent.plan` on the exact prompt. Reject empty prompts, unknown entity references, unsupported assumptions, unresolved hard uncertainty, invalid timing, broken relations, and contradictory event order.
3. Validate the projected `SceneContract`, multi-entity `TrajectoryPlan`, interaction lifecycle, one-based frame bounds, state continuity, target visibility, shot coverage, and event observability before execution.
4. Execute through the controlled adapter or Blender MCP. Persist prompt/plan/Harness/evaluator fingerprints, MCP response, state transitions, and immutable artifacts.
5. Apply the real artifact gate. Require manifest, contract, trajectory, camera plan, frozen job source, `candidate.blend`, observer-bound `.blend`, host-assembled `.mp4`, trusted telemetry manifest, telemetry, index, and at least three readable sampled PNGs. Generated telemetry is quarantined and never trusted.
6. Run deterministic evaluation first. Hard failures block VLM, training records, and patch selection. Inspect a sampled frame when a semantic or visibility failure is plausible.
7. Run the blind primary VLM evaluation only for artifact-complete deterministic-pass runs and only through a compliant endpoint. Its payload contains the exact prompt and chronological frames/timecodes, not DirectorPlan, scene contract, deterministic findings, arm, or Harness version. Record `unavailable` for network, policy, or schema failures; never convert it to zero or a synthetic preference. Formal configuration uses `gpt-5.6-luna` for primary and `gpt-5.6-terra` for audit.
8. Use only the bounded execution-recovery loop: a plan/non-compliance or
   Blender/artifact failure may regenerate the whole case candidate at most
   three times (`max_inner_attempts=3`). Do not repair a scene in place, retry
   the same source secretly, or switch to a template. Route repeated findings
   to one component in the outer loop, keep the evaluator and dataset fixed,
   and stop after the project's configured attempt limit.
9. For the outer loop, aggregate repeated train failures by failure ID, owner, category, severity, and root cause. A proposal requires the same failure to affect two distinct train cases and exactly one owner.
10. Re-run paired train/dev with the same dataset, evaluator, backend, and fingerprints. Accept only strict paired train improvement, paired and overall dev non-regression, zero hard regressions, and non-regressing artifact completion. For realism/renderer patches, realism must improve and task score must not fall. Keep test split frozen and use it only for final blind verification.

## Project commands

When the repository contains the current implementation, prefer these entry points:

```text
scripts/prepare_real_jobs.py       # immutable jobs
scripts/run_real_pipeline.py       # dry-run/evaluate host stages
scripts/evaluate_real_runs.py      # artifact gate + deterministic reports
scripts/evaluate_real_videos.py    # eligible sampled-frame VLM stage
scripts/run_real_outer_loop.py     # train/dev failure aggregation
scripts/build_agent_training_dataset.py    # reproducible trajectory-v5 agent-codegen builder
scripts/validate_agent_training_dataset.py # 60/60/20 leakage and contract validator
scripts/build_benchmark_prompt_index.py    # verbatim VBench-2.0 execution index
scripts/validate_benchmark_prompt_index.py # benchmark provenance and no-mutation gate
scripts/validate_frozen_eval_set.py        # held-out 20-case blind boundary
scripts/validate_codegen_examples.py       # reviewed real few-shot context gate
scripts/check_training_readiness.py        # independent pre-training gate matrix
skills/t2blendercodeharness/scripts/build_self_evolution_records.py # historical evidence -> JSONL
```

For the active six-round protocol, use
`dataset/vbench2-agent-training-index-v1`, `dataset/frozen-eval-v2`,
`scripts/train_real_harness.py --mode six-rounds --provider-mode model`, and the canonical memory at
`docs/t2blendercodeharness-agent-training-memory-v1.md`. The historical
trajectory datasets and augmented VBench-derived set must never be passed to
the training entry point; `train_real_harness.py` and readiness reject them.
The real-training modes also refuse to prepare/render any case unless the
specified readiness report contains `training_allowed=true` and the formal
release report verifies G0--G3.

The current integrity modules are `source-fingerprint-v1` (semantic code
reuse), `paired-statistics-v1` (fixed-seed CI and safety invariants),
`physics-oracle-v2-obb-bvh-contact-ownership` (trusted raw contact/ownership),
`experiment-fingerprint-v1` (full provenance), and `active-sampling-v2-replay`
(train/dev-only failure sampling). The frozen/OOD boundary is
`dataset/frozen-eval-v2`.

Run `python skills/t2blendercodeharness/scripts/capability_check.py --project-root .` before claiming the skill works in a new project. Read `references/real-pipeline.md` for state and artifact details.

## Skill self-evolution

Treat skill self-evolution as a reviewed documentation change driven by prior
real-run evidence, not as runtime self-editing. Keep the Harness implementation,
dataset labels, evaluator formulas, and generated plans immutable while
producing a proposal.

For an existing run, use this order:

1. Hash the current `SKILL.md`, dataset metadata, evaluator policy, and Harness
   commit. Run the capability check and project tests before changing the skill.
2. Use `build_self_evolution_records.py` to normalize accepted round
   `patch_manifest.json` files and their new train batches into JSONL. The
   converter is read-only and records source paths; it must not infer a failure
   from a VLM-unavailable result.
3. Run `propose_skill_update.py` on that JSONL. Require one normalized failure
   to affect at least two distinct cases and map the proposal to exactly one
   owner/section. A proposal is not an applied change.
4. After explicit human approval, edit the smallest skill section, preserve the
   proposal and pre/post hashes, then rerun capability check, full tests, and a
   forward test on the original dataset plus a prompt variation.
5. Accept the skill update only if the new instructions preserve artifact
   gates, Director evidence, one-owner routing, unavailable-VLM handling,
   train/dev/test isolation, and the real-video retry rule. Otherwise record
   `rejected` and retain the previous skill text.

For the current project, the reusable commands are:

```powershell
uv run python skills/t2blendercodeharness/scripts/build_self_evolution_records.py `
  --round-root out/training/multi-five-rounds-v1 `
  --out out/skill-self-evolution-v1/historical_records.jsonl
uv run python skills/t2blendercodeharness/scripts/propose_skill_update.py `
  --records out/skill-self-evolution-v1/historical_records.jsonl `
  --out out/skill-self-evolution-v1/proposal.json
```

## Realism evaluator boundary

For realism probes, use the project training sub-skill and the independent-review boundary in `evaluator/realism.py`. The Blender geometry audit is an eligibility gate, not a realism oracle: it may reach 100 only for structural compliance. One shared visual-review call using lowercase `gpt-5.6-luna` or `gpt-5.6-terra` returns separate task and realism dimensions; the scores are never added. An `assistant_local_review` is valid only when a human has supplied an auditable payload with frame-grounded evidence. Realism uses `.15` geometry, `.15` rendered-frame evidence, and `.70` independent visual review. If no review is available, retain only the capped `artifact_only_proxy` evidence score and mark it `not_established`; never copy it into a VLM score. Trusted observer output is the only telemetry accepted by a formal run; `candidate.blend` is the generator/observer boundary.

Use `python skills/t2blendercodeharness/scripts/propose_skill_update.py --records <records.jsonl> --out <proposal.json>` after a real evaluation batch. The script may group repeated failures and propose a single-owner section update, but it must not edit `SKILL.md`, source code, evaluator code, or dataset labels. Require human review, capability checks, project tests, and a forward-test before applying a proposal. Read `references/self-evolution.md`.

## Stop conditions

Stop and report a blocked stage when any of these occurs:

- contract or trajectory validation fails;
- MCP/CLI execution is failed or still running;
- required artifacts are missing, stale, or unreadable;
- deterministic hard gate fails;
- VLM endpoint is unavailable or non-compliant;
- train evidence has no repeated actionable failure;
- a proposal has more than one owner;
- no repeated failure affects two distinct train cases;
- dev regresses or test data would influence patch selection.

Do not claim "trained", "accepted", or "video-evaluated" unless the corresponding report and evidence artifact exists.

## Common mistakes

| Temptation | Required correction |
|---|---|
| Telemetry exists, so evaluate | Wait for the full artifact gate and readable samples. |
| VLM returned 403, so use score 0 | Record `unavailable`; exclude it from numeric training data. |
| Several owners failed, so patch all files | Split proposals by owner and rerun the acceptance gate. |
| Train passed, so accept the patch | Require strict train improvement plus non-regressing dev. |
| Test reveals the fix | Keep test frozen until final blind verification. |
| Skill can edit itself immediately | Emit a reviewed proposal; never self-apply. |
| Frame statistics produced 14 numbers | Keep semantic dimensions `None`; retain only `artifact_health`. |
| A prose skill edit raised confidence | Count only a tested runtime patch as Harness evolution. |
| An orbit has two endpoints | Require a sampled arc and `continuity_group` check. |
| A prop is near a hand | Require a handoff constraint and penetration check. |

## Evaluator v5 boundary and executable evolution

`evaluator/visual_evidence.py` and any `frame_statistics_only-v1` fallback may
measure only artifact health and low-level observations. Its semantic fields
(`prompt_compliance`, event timing, physical plausibility, and character/object
trajectory) must be `None`; its `score` must be `None`; and a record with
`review_source=frame_statistics` must never enter realism fusion or the patch
acceptance gate. A real VLM or a validated human review is required for
semantic/video quality claims. Use lowercase model IDs `gpt-5.6-luna` and
`gpt-5.6-terra`; unavailable transport or schema results remain unavailable.

Every patch proposal is a falsifiable record with `predicted_fixes`,
`predicted_regressions`, and `prediction_rationale`. Attribution runs before
root-cause distillation; an unpredicted negative case delta produces a
`refuted` verdict and file-granularity rollback. A proposal still needs one
owner and at least two distinct train cases.

Prefer executable `function_library`, `owner_mapping`, and append-only
`memory_entry` updates over prose-only guidance. A `prose_guidance` change is
not a training gain unless its runtime effect is separately demonstrated.
Keep orbit/occlusion/continuity, handoff attachment, and penetration checks as
runtime evidence, not as promises in documentation.

## Improvement-plan release boundary

The plan is implemented as five explicit gates. G0 covers model provenance,
portable benchmark inputs, the independent trusted observer and the blind
generator/judge boundary. G1 freezes result states, scoring-v7 applicability,
evidence/confidence thresholds, paired-statistics policy, golden calibration
and the complete experiment fingerprint. G2 is an exact 10-train + 10-dev
paired pilot against `rule_template_baseline` and `model_driven_candidate`.
G3 is a no-patch 60-train + 60-dev shadow run that verifies resume, cost,
memory and fingerprint stability. G4 is the formal six-round Harness run.

Use `scripts/check_formal_release_gates.py` to create the sealed
`out/formal_release_gate_report.json`. Formal modes of
`scripts/train_real_harness.py` require both that report and the ordinary
readiness report; a manually edited boolean is rejected. The required
versions are `source-fingerprint-v1`, `paired-statistics-v1`,
`physics-oracle-v2-obb-bvh-contact-ownership`, `experiment-fingerprint-v1`, and
`active-sampling-v2-replay`. Frozen evaluation is `dataset/frozen-eval-v2` and is
never read for patch selection or threshold tuning.
