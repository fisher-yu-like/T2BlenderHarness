---
name: t2blendercodeharness-training
description: Train T2Blendercodeharness with real Blender CLI proxy videos on trajectory-v4-multi using five outer-loop rounds, paired train/dev gates, separate Director/task/realism scores, and append-only Markdown memory.
---

# T2Blendercodeharness Training

This is the real-video outer-loop protocol for `T2Blendercodeharness`. The
object being optimized is the Harness. Freeze the dataset, evaluator policy,
Blender binary, render settings, Director provider, and model naming during a
run. Do not edit dataset labels, evaluator thresholds, Blender source, or
generated plan files to obtain a better score.

## Approval gate

Batch training must not start until all of the following are true and the user
has explicitly approved the run:

1. Full test suite passes.
2. Capability check passes.
3. `trajectory-v4-multi` validator passes with 50/60/30 and no leakage.
4. A real two-actor/two-prop Blender smoke has a complete artifact gate,
   playable MP4, Director/interaction deterministic pass, geometry report,
   and inspected handoff frame.
5. The visual-review mode is declared as `external_vlm` or
   `assistant_local_review`.

The current smoke is complete. The endpoint model listing worked, but the
image request returned HTTP 403/error 1010, so the current approved default is
to record external review as `unavailable` until a compliant endpoint or an
auditable local review payload is available. Never replace unavailable with a
zero, 100, or a plan-derived score.

## Frozen protocol

- Dataset: `dataset/trajectory-v4-multi`; exactly 50 train, 60 dev, 30 frozen test.
- Output: `out/training/multi-five-rounds-v1`.
- Memory: `docs/t2blendercodeharness-multi-training-memory-v1.md`.
- Report: `docs/t2blendercodeharness-multi-training-report-v1.md`.
- Blender: `D:\blender\blender.exe`.
- Render workers: 12 workers; each case owns an isolated immutable directory.
- Blender render retries: at most 2 per case; outer-loop attempts are different
  from render retries.
- External endpoint model IDs: lowercase `gpt-5.6-luna` or `gpt-5.6-terra`.
- Canonical report names: `gpt-5.6-Luna` and `gpt-5.6-Terra`.
- Rounds: exactly 5; each round has at most five attempts (5 outer-loop attempts).
- Every real proxy video gets an explicit absolute proxy video address or
  `NOT_RENDERED` reason in Memory.

The five-round protocol has one attempt batch of 10 train + 10 paired dev
videos. At round end it evaluates cumulative train cases plus all 60 dev:

| Round | New train | Paired dev | Round-end train + all dev | Maximum videos including attempts |
|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 10 + 60 = 70 | 170 |
| 2 | 10 | 10 | 20 + 60 = 80 | 180 |
| 3 | 10 | 10 | 30 + 60 = 90 | 190 |
| 4 | 10 | 10 | 40 + 60 = 100 | 200 |
| 5 | 10 | 10 | 50 + 60 = 110 | 210 |
| **Total** | **50** | **50 paired** | **450 overall evaluations** | **950 maximum video evaluations** |

The 10 dev cases in the sixth dev family are overall-only in this protocol.
The 30 test cases are run exactly once after round five and cannot create a
proposal.

## Per-case pipeline

```text
exact prompt
  -> DirectorAgent: entities, evidence, uncertainty, event graph
  -> interaction lifecycle and dependency schedule
  -> multi-entity trajectories and current-owner states
  -> multi-target camera choreography and visibility predicates
  -> projected SceneContract/TrajectoryPlan
  -> generated Blender CLI job
  -> real Blender frames + proxy.blend + host MP4 + telemetry
  -> render retry if Blender/artifacts fail (maximum 2 retries)
  -> RealArtifactGate
  -> deterministic task checks + independent Director evaluator + interaction evaluator
  -> geometry/frame realism audit
  -> one shared visual review for eligible real videos
  -> separate Director plan, task, and realism channels
```

The Director evaluator is independent from the authored dataset oracle for
multi-entity runs. It checks evidence, assumptions, stable identity,
dependencies, missing trajectories, path collisions, camera target coverage,
and telemetry identity. Interaction evaluation separately checks attach,
transfer, detach, handoff window, contact, final owner, and final support.

## Visual review branch

Use exactly one branch per eligible real video:

- `external_vlm`: call lower-case `gpt-5.6-luna` or `gpt-5.6-terra` on the
  sampled real PNG frames; report the canonical mixed-case name.
- `assistant_local_review`: create an auditable review request, inspect the
  listed frames in chronological order, and write an explicit review payload
  with visible evidence and confidence. The local reviewer must not infer
  unseen motion from the plan or telemetry.
- If neither branch yields a valid review with confidence at least 0.6, write
  `unavailable` or `needs_human_review` and exclude the numeric task review
  score from aggregates. Artifact-only realism remains separately available.

## Score channels and formulas

The recorded fields are `director_plan_score`, `task_score`/`task_final_score`,
and `realism_score`; each is a separate channel. `deterministic_score` is a
gate and an independent task-contract score. It
checks artifacts, entities/kinds, timebase, telemetry, camera activation,
event coverage, phase alignment, support/contact, velocity continuity, and
ordinary findings. `director_plan_score` is stored separately and is never
folded into the deterministic score.

For a valid visual review, grouped task VLM scoring is:

```text
semantic = GM(prompt_compliance, physical_plausibility,
              object_trajectory, event_timing)
choreography = GM(camera_coverage, camera_innovation,
                  character_trajectory, temporal_smoothness)
task_vlm = 0.45 * semantic + 0.45 * choreography + 0.10 * visual_clarity
task_final = 0.20 * deterministic_score + 0.80 * task_vlm
```

If deterministic has a hard gate failure, the task score is capped by the
evaluator policy; it is never rescued by a high visual score.

Realism is independent and is not added to task score:

```text
artifact_only_unbounded = 0.60 * geometry_score + 0.40 * frame_evidence_score
artifact_only_proxy = min(80, 0.80 * artifact_only_unbounded)
reviewed_realism = 0.15 * geometry_score
                 + 0.15 * frame_evidence_score
                 + 0.70 * GM(appearance_detail, physical_realism,
                              spatial_consistency, motion_naturalness,
                              visual_presentation)
```

Without an independent review, `realism_claim=not_established` and
`score_kind=artifact_only_proxy` are mandatory.

## Outer-loop Harness evolution

1. Run attempt 1 for 10 unique train and 10 paired dev cases.
2. Update the Markdown memory immediately with every rendered or
   `NOT_RENDERED` case.
3. Aggregate only repeated train failures with the same failure ID/root cause
   across at least two distinct cases.
4. Route the proposal to exactly one Harness owner. Valid owners include the
   Director parser/scheduler, trajectory planner, camera planner, Blender code
   adapter, executor, proxy renderer, and evaluator interface; the patch must
   be under the permitted Harness scope.
5. Re-run the same paired train/dev cases with identical fingerprints.
6. Accept only when strict paired train improves, paired dev does not regress,
   overall dev does not regress, hard failures do not increase, and artifact
   completion does not decrease. Renderer/realism patches additionally need
   realism improvement with no task regression.
7. Continue attempts 2–5 only while there is new actionable evidence. Stop on
   accepted patch, rejected/rolled-back patch without new evidence, no patch,
   or attempt 5.
8. At round end run cumulative train plus all dev. After round five run frozen
   test once in no-proposal mode.

Do not repeat inner-loop repairs to the same scene as training. Blender job
retry is only for render failure; Harness evolution is the outer loop.

## Mandatory Memory table

Update `docs/t2blendercodeharness-multi-training-memory-v1.md` after every
split and every round. Keep exact prompt text, absolute proxy path, immutable
DirectorPlan hash, and natural-language handling:

| Round | Attempt | Split | Case ID | Prompt | Proxy video address | Director plan score | Task score | Realism score | Review source/confidence | Harness problem | Harness fix location/method | Before->after delta | Natural-language handling |
|---:|---:|---|---|---|---|---:|---:|---:|---|---|---|---:|---|

For missing videos use `NOT_RENDERED: <failure reason>`; never invent a path.
Keep JSON evidence alongside the table:

```text
round-XX/attempt-YY/real/{train,dev}/
round-XX/overall/real/{train,dev}/
round-XX/attempt_report.json
round-XX/overall_report.json
round-XX/patch_manifest.json
memory/harness_updates.jsonl
```

Plot Director plan, task, realism, artifact completion, hard-failure rate,
train-dev gap, and findings-by-owner curves separately. Missing visual review
is an unavailable point, never zero.

## Required commands

```powershell
uv run python scripts/validate_multi_entity_dataset.py --dataset-root dataset/trajectory-v4-multi
uv run python scripts/train_real_harness.py --mode protocol --dataset-root dataset/trajectory-v4-multi --round-root out/training/multi-five-rounds-v1
uv run python scripts/train_real_harness.py --mode attempt --round 1 --attempt 1 --dataset-root dataset/trajectory-v4-multi --round-root out/training/multi-five-rounds-v1 --blender-bin D:\blender\blender.exe --workers 12 --vlm-model gpt-5.6-luna --markdown-path docs/t2blendercodeharness-multi-training-memory-v1.md
uv run python scripts/train_real_harness.py --mode overall --round 1 --dataset-root dataset/trajectory-v4-multi --round-root out/training/multi-five-rounds-v1 --blender-bin D:\blender\blender.exe --workers 12 --vlm-model gpt-5.6-luna --markdown-path docs/t2blendercodeharness-multi-training-memory-v1.md
```

Before any completion claim, rerun the full suite, capability check, dataset
validator, final artifact audit, and frozen blind test report. Do not claim
trained, accepted, or video-evaluated without the corresponding real evidence.
