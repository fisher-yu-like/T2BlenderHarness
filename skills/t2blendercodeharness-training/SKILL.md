---
name: t2blendercodeharness-training
description: Use when running real Blender outer-loop Harness training with fixed train/dev rounds, per-case Director and code generation, visual review, anti-overfit gates, and append-only experiment memory.
---

# T2Blendercodeharness training

## Human visual-calibration handoff

The review UI and calibration bundle are part of the training gate, but they
are not a substitute for real agent/provider evidence. The current official
training path uses the in-process `codex-local` provider; there is no external
endpoint in the official path. Use
`dataset/golden-review-exact-v2` as the active blind comparison/calibration bundle:
30 cases, 90 real MP4 files, and 180 required score rows from two independent
annotators. Launch the Chinese UI with:

```powershell
uv run python scripts/golden_review_app.py --bundle dataset/golden-review-exact-v2 --host 127.0.0.1 --port 8765
```

Each reviewer must use a distinct ID and score every sample independently.
Afterwards run `uv run python scripts/finalize_golden_review.py --root
dataset/golden-review-exact-v2` and the strict validator. The finalizer computes
ICC(2,1) for all 14 dimensions; missing or low-coverage annotations remain
`pending` and cannot be replaced by a numeric score.

This protocol trains the Harness, not the dataset, evaluator, Blender binary,
or generated plans. The active dataset is the verbatim VBench-2.0 execution
index `dataset/vbench2-agent-training-index-v1`; every `prompt` is copied
exactly from the local benchmark source. It carries no authored event/entity
oracle labels: DirectorAgent must derive the executable plan at runtime. The
historical `trajectory-v5-agent-codegen`, `vbench-derived-100-v1`, and
`trajectory-v4-multi` datasets remain read-only comparison fixtures and are
ineligible for training.

The historical trajectory index is self-built and the VBench-derived set is
augmented; neither is a valid benchmark training input.

The benchmark index contains exactly 60 train, 60 dev, and 20 frozen test
cases.
Its runner name is `multi-five-rounds-v1`; that legacy protocol is reproducible
only as historical evidence and is not the active six-round training command.

## Approval gates before real training

Do not start the six-round run until all are recorded as passing:

1. Full pytest and the capability check pass.
2. `scripts/validate_benchmark_prompt_index.py` passes: exactly 60 train, 60
   dev, and 20 frozen test cases; six train and six dev ten-case families;
   unique source identities; every prompt equals raw VBench-2.0 `prompt_en`;
   no locally authored prompt, event, entity, proxy, or oracle fields; and no
   overlap with the frozen/reference datasets.
3. `scripts/validate_frozen_eval_set.py --root dataset/frozen-eval-v1` passes.
4. A real `D:\blender\blender.exe` smoke produces a playable MP4, sampled PNGs,
   `proxy.blend`, telemetry, complete manifest, and inspectable camera/hand
   evidence.
5. The visual-review source is an auditable human review payload or the
   in-process local Codex visual reviewer. The report metadata may retain the
   canonical lowercase labels `gpt-5.6-luna` and `gpt-5.6-terra`, but the
   current run has no external endpoint. Unavailable review stays
   `unavailable`; it is never a numeric zero or a plan-derived score. Human
   golden calibration must be complete before accepting a patch from visual
   quality evidence.
6. `scripts/check_training_readiness.py` must report
   `training_allowed=true`. It keeps full tests, capability, dataset,
   frozen-eval, agent Blender smoke, golden review, dynamic provider, and
   paired agent/template evidence as independent gates. The dynamic-provider
   gate is satisfied by a provenance-bearing `codex-local` Director + code
   pair, not by the historical external `codex exec` diagnostic. A
   template-only smoke, numeric placeholder, or unavailable provider cannot
   satisfy a gate.
   The real-training modes of `scripts/train_real_harness.py` enforce this
   report again before preparing or rendering a case.

## Explicit pre-calibration diagnostic run

The user may deliberately run the real pipeline before human visual
calibration when the generated videos are visibly coarse. This is a separate
diagnostic mode, not a relaxation of the formal gate:

```powershell
uv run python scripts/train_real_harness.py --mode diagnostic-attempt `
  --round 1 --attempt 1 `
  --dataset-root dataset/vbench2-agent-training-index-v1 `
  --readiness-report out/preflight/readiness-exact-v2.json `
  --round-root out/training/diagnostic-six-rounds-v1 `
  --blender-bin D:\blender\blender.exe --workers 4 `
  --vlm-model gpt-5.6-luna
```

`diagnostic-six-rounds` is available for a complete six-round run with the
same at-most-five-attempt, 10-train+10-dev, and round-end 120-case accounting.
The diagnostic manifest must explicitly contain
`diagnostic_precalibration`, `formal_training_allowed=false`, and
`visual_scores_permitted=false`. It may record deterministic, artifact,
geometry, and render-failure evidence, but it must not turn missing human
visual review into a number, accept a Harness patch on that missing channel,
or claim that the formal six-round experiment passed. Human annotation starts
after the Harness upgrade and uses the exact-prompt bundle separately.

Before any case is rendered, `audit_dynamic_agent_index` must verify
`generation_mode=agent`, a per-case codegen call ID, a real generated source,
an embedded `CASE_SCENE_PROFILE` marked
`codex-local-case-profile-v2`, and case-specific source hashes. A unique hash
alone is insufficient: a generic scaffold stamped with a case id is rejected
as `missing_case_specific_generation_profile`; the profile signature must also
bind to the DirectorPlan hash prefix and the recorded code hash must match the
source bytes. If all cases reuse one source,
`all_cases_reuse_one_generated_source` is a hard diagnostic failure. The
explicit `template_baseline` arm is never accepted by this audit and never
enters training evidence.

## Fixed six-round protocol

- Six rounds, at most five outer attempts per round.
- Every attempt: 10 new train cases + 10 paired dev cases.
- The fixed per-attempt accounting is exactly `10 train + 10 paired dev`.
- Every round end: a separate overall evaluation of all 60 train + all 60
  dev cases (120 videos), not only the newly sampled batch.
- Upper-bound accounting: `6 × (5 × 20 + 120) = 1320` real video executions.
  The runner may stop an outer round early when there is no new actionable
  repeated failure; 1320 is a maximum, not an instruction to waste retries.
- Each case has an inner candidate loop with `max_inner_attempts=3`: an
  invalid/non-compliant DirectorPlan or a Blender render/artifact failure
  causes a fresh local Director + BlenderCode candidate. After three failed
  candidates the case is `NOT_RENDERED` and all attempt evidence is retained.
  There is no hidden same-source render retry in the official local path.
  Therefore the worst-case candidate-generation budget is `1320 × 3 = 3960`;
  1320 remains the committed protocol case-slot count.
- Render in bounded groups of at most 12 cases. The safe default is 4 Blender
  workers per group, with a hard ceiling of 12 workers; never launch the
  six-round corpus as one unbounded pool.
  Record Blender path, group size, worker count, resolution, samples, source
  hash, retry count, and artifact status.
- Keep groups serial so disk/CPU contention is bounded. A single evaluator
  worker may evaluate the previously rendered group while the next group is
  rendering; it must not start a second evaluator for the same artifacts.
- The frozen test set is milestone-only: it never creates a proposal or
  acceptance decision.

The attempt and overall paths use the same shared code cache for a protocol
root. A cache key is `plan_hash + harness_version`; a cache hit reuses the exact
source. Any L2 or L3 change must increment the Harness version or explicitly
request regeneration.

The canonical Markdown path is append-only across diagnostic roots. The writer
must aggregate sibling `out/training/diagnostic-*` roots and deduplicate only
the same `(round, attempt, split, case_id, proxy_video)` evidence row; it must
never replace the table with a single-run summary. The current six-round
diagnostic has 840 protocol video slots (120 attempt + 720 overall), while the
formal maximum remains 1320 because each round may stop before five attempts
when no new actionable failure exists.

## Per-case execution

```text
exact prompt
  -> dynamic DirectorAgent via codex-local (entities, evidence, event graph, trajectories, camera cues)
  -> DirectorPlan obligations/evidence/ordering gate
  -> BlenderCodeAgent per-case source generation from blender.lib signatures
  -> static source + case coverage gate
  -> frozen job source and hash chain
  -> D:\blender\blender.exe, isolated case directory
  -> proxy.mp4 + proxy.blend + frames + telemetry
  -> artifact/deterministic/independent-oracle checks
  -> decode proxy.mp4 + read Blender runtime_observations
  -> one shared local-Codex or explicitly configured VLM review on chronological frames
  -> separate visual, physical, trajectory, camera, task, and realism channels
  -> append Memory and aggregate repeated train failures
```

The inner loop is execution recovery, not Harness evolution: it may regenerate
the failed case up to three times, but it cannot patch a scene in place or
change the evaluator. The outer loop still changes at most one Harness owner
per candidate and records every failed candidate as evidence.

`template_baseline` is an explicit paired-ablation arm only. It is never an
agent fallback. Missing Director provider, code provider, schema, coverage,
artifact, or visual review fails closed and writes the failure reason; no
template video is substituted.

The dynamic Director must not invent a human for an object-only benchmark
prompt. A prompt about a droplet, balloon, liquid, surface, or other non-human
subject must produce only the prompt-derived prop(s) plus any explicitly
declared staging support. An actor in that DirectorPlan without prompt evidence
is a semantic Harness failure and must be fixed before human visual review.

## Score channels

### Real-video visual/physics/trajectory review is active

The local path no longer stops at `frame_statistics`. After a real Blender
render, `evaluate_real_video` decodes the actual `proxy.mp4` (not just source
PNGs) and requires `telemetry.runtime_observations` for every rendered frame.
The Blender job records actual entity transforms, world-space bounds,
screen-space bounds, camera transforms, actor pose points, and connected-rig
evidence. The in-process Codex visual reviewer consumes those observations and
the decoded MP4 frames through the shared VLM response schema:

```text
visual_score     = mean(visual_clarity, appearance_detail, visual_presentation)
physical_score   = actual collision/ground/rig/smoothness evidence
trajectory_score = mean(object_trajectory, character_trajectory,
                        event_timing, temporal_smoothness)
camera_score     = mean(camera_coverage, camera_innovation)

semantic = GM(prompt_compliance, physical_plausibility,
              object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_vlm = .45 * semantic + .45 * choreography + .10 * visual_clarity
realism_vlm = GM(appearance_detail, physical_realism,
                 spatial_consistency, motion_naturalness,
                 visual_presentation)
reviewed_realism = .15 * geometry_score + .15 * rendered_frame_evidence
                 + .70 * realism_vlm
overall_vlm = .70 * task_vlm + .30 * realism_vlm
```

The deterministic score remains a gate/diagnostic channel and is not added to
the task or realism result. The local evidence scorer caps observed scores at
95, caps semantic compliance at 90, and returns `unavailable` when the MP4 or
runtime observations are missing. A valid plan or complete file list therefore
cannot create a visual/physics/trajectory score. `codex_local_visual_review`
is the actual in-process review source here; it does not call an external key.
External labels remain lowercase `gpt-5.6-luna` and `gpt-5.6-terra` only when
an external provider is explicitly enabled.

The deterministic result is a gate and diagnostic channel, not the main visual
score. For an eligible review:

```text
semantic = GM(prompt_compliance, physical_plausibility,
              object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_score = .45 * semantic + .45 * choreography + .10 * visual_clarity
task_final_score = task_score
realism_vlm = GM(appearance_detail, physical_realism,
                 spatial_consistency, motion_naturalness,
                 visual_presentation)
reviewed_realism = .15 * geometry_score + .15 * frame_evidence_score + .70 * realism_vlm
overall_vlm_score = .70 * task_score + .30 * realism_vlm
```

`task_score` and `realism_score` are never added together. Geometry and frame
statistics may report `artifact_health`/`artifact_only_proxy`, but cannot claim
semantic, trajectory, camera, or realism quality. A missing review is
`unavailable` or `needs_human_review`, not zero.
The `frame_statistics` source is artifact-health-only and is excluded from
semantic training decisions.

## Outer-loop evolution

1. Start with attempt 1 and record every rendered or `NOT_RENDERED` case.
2. Aggregate repeated train findings by failure ID, root cause, owner, severity,
   and evidence path. Require the same actionable failure in at least two
   distinct train cases.
3. Propose exactly one Harness owner/component. A proposal must include
   `predicted_fixes`, `predicted_regressions`, `prediction_rationale`, rerun
   command, and rollback path. Attribution runs before root-cause distillation.
   Prefer an executable `function_library`, `owner_mapping`, or append-only
   `memory_entry`; a prose-only change is not a measured training gain.
4. Regenerate DirectorPlan and Blender source for the candidate Harness version,
   then rerun the same paired train/dev cases. Do not perform an inner-loop
   scene repair.
5. Accept only strict train improvement, paired and full-dev non-regression,
   no new hard regressions, no artifact-completion regression, and—when the
   owner is visual/renderer—realism improvement without task loss. A refuted
   proposal rolls back only its owner files; evidence and Memory remain.
6. Stop before five attempts if no repeated actionable failure remains. Never
   use frozen test results to choose a patch.

## Mandatory append-only Memory

Update the Markdown table after each split and each round; do not wait until
the experiment ends. The canonical file is
`docs/t2blendercodeharness-agent-training-memory-v1.md`. Every row must retain
the exact prompt and an absolute proxy video address, or
`NOT_RENDERED: <reason>`, plus natural-language problem and handling:

| Round | Attempt | Split | Case ID | Prompt | Proxy video address | Director plan score | Task score | Realism score | Review source/confidence | Harness problem | Harness fix location/method | Before→after delta | Natural-language handling |
|---:|---:|---|---|---|---|---:|---:|---:|---|---|---|---:|---|

Also preserve machine-readable evidence under:

```text
round-XX/attempt-YY/real/{train,dev}/
round-XX/overall/real/{train,dev}/
round-XX/attempt_report.json
round-XX/overall_report.json
round-XX/patch_manifest.json
memory/harness_updates.jsonl
```

The machine-facing row keys are `director_plan_score`, `task_score`,
`realism_score`, and `proxy_video`; the human table keeps the same meanings
under readable column labels.

Plot task, realism, Director plan, artifact completion, hard-failure rate,
train/dev gap, and findings by owner as separate series. Unavailable visual
review is a missing point, never an imputed score.

## Commands

```powershell
uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-training
uv run python scripts/validate_benchmark_prompt_index.py `
  --root dataset/vbench2-agent-training-index-v1 `
  --source data/vbench-source/VBench2_full_info.json `
  --reference-root dataset/frozen-eval-v1 `
  --reference-root dataset/vbench-derived-100-v1 `
  --reference-root dataset/trajectory-v4-multi `
  --reference-root dataset/trajectory-v5-agent-codegen
uv run python scripts/validate_frozen_eval_set.py --root dataset/frozen-eval-v1
uv run python scripts/check_training_readiness.py --project-root . `
  --dataset-root dataset/vbench2-agent-training-index-v1 `
  --frozen-reference-root dataset/vbench2-agent-training-index-v1 `
  --frozen-reference-root dataset/vbench-derived-100-v1 `
  --frozen-reference-root dataset/trajectory-v4-multi `
  --blender-smoke-root out/preflight/<agent-smoke> `
  --dynamic-provider-root out/preflight/<dynamic-provider> `
  --full-test-report out/two-plan-convergence/full-test.xml
uv run python skills/t2blendercodeharness/scripts/capability_check.py --project-root .
uv run python scripts/render_evaluate_groups.py `
  --run-root out/preflight/<prepared-train> `
  --run-root out/preflight/<prepared-dev> `
  --dataset-root dataset/vbench2-agent-training-index-v1 `
  --blender-bin D:\blender\blender.exe `
  --group-size 12 --workers 4 --timeout-s 900 --max-render-retries 2 `
  --output out/preflight/grouped-pipeline-report.json
uv run python scripts/train_real_harness.py --mode protocol --dataset-root dataset/vbench2-agent-training-index-v1 --round-root out/training/agent-six-rounds-v1
uv run python scripts/train_real_harness.py --mode six-rounds --dataset-root dataset/vbench2-agent-training-index-v1 --readiness-report out/training_readiness_report.json --round-root out/training/agent-six-rounds-v1 --blender-bin D:\blender\blender.exe --workers 4 --vlm-model gpt-5.6-luna --markdown-path docs/t2blendercodeharness-agent-training-memory-v1.md
```

Before claiming training complete, rerun full tests, capability check, both
dataset validators, final artifact audit, visual-review provenance audit, and
frozen blind evaluation. A readiness report with any non-pass gate keeps the
six-round run blocked; a partial run is documented as partial.

## Exact-prompt human gate and outer-loop state machine

The current calibration bundle is `dataset/golden-review-exact-v2`. It was
built only from the active VBench index with `--include-split train
--include-split dev`; the frozen test split is excluded. It contains 30 cases
and 90 real MP4 files from three hidden arms (`codex-local`, explicit
`template_baseline`, and `direct_code`). The builder checks every available
run `prompt_hash` against the displayed prompt. A non-zero mismatch makes the
bundle comparison-only and blocks readiness.

Serve it with `scripts/golden_review_app.py --bundle
dataset/golden-review-exact-v2`. Two independent annotators must score every
blind video on all 14 dimensions and provide visible evidence; missing rows or
ICC evidence are pending, never zero. The bundle is for evaluator calibration
and arm comparison only. It cannot itself create a Harness patch.

The top-level runner uses `run_bounded_outer_attempts`: one attempt produces
evidence, then an explicit Codex Host transition must record one Harness-owner
patch before the next attempt. `patch`, `accept`, and `stop` are the only
legal transitions. With no transition it stops as `awaiting_harness_patch`;
it never repeats the same Harness automatically. The hard bound is at most
five attempts per round. Each patch must pass `validate_harness_patch_paths`;
generated Blender jobs, plans, dataset labels, and evaluator code are not
patch targets.

## Current rerun rules (real-video retraining)

The six-round rerun uses only `dataset/vbench2-agent-training-index-v1`, whose
prompts are exact raw `prompt_en` values from the local VBench-2.0 source. It
does not create prompts or add authored entity/event labels. Each round uses
at most five outer attempts; each attempt renders ten train and ten paired dev
cases; after the chosen attempt, the round evaluates all 60 train and all 60
dev cases. The frozen test split is milestone-only.

One outer patch may modify one Harness owner only. The first attempt is the
baseline for that round. A patch is eligible only when the same actionable
failure appears in at least two distinct train cases and the candidate is
regenerated through DirectorAgent and BlenderCodeAgent. Accept only strict
train task-VLM improvement, paired-dev non-regression, full-dev
non-regression, no new hard failures, and no regression in the relevant
visual/physical/trajectory/camera channel. A rejected candidate is retained
in Memory and its generated video and scores are never deleted.

The inner loop is execution recovery only: a non-compliant plan, failed
Blender render, or incomplete artifact gets at most three fresh local
Director+code candidates. It cannot edit a rendered scene and cannot switch
to a template. Rendering is serial groups of at most 12 cases, default four
Blender workers, hard ceiling twelve. The canonical Markdown memory is
updated after every split and every round with the separate visual, physical,
trajectory, and camera table.

After the six-round rerun, the decision is data-driven: if task/realism
channels or cross-split behavior remain below the documented target, select a
non-overlapping extension from unused raw VBench prompts and retrain with the
same rules; if they are satisfactory, use unused raw VBench prompts only for
an evaluation set. No self-authored replacement dataset is permitted in
either branch.
