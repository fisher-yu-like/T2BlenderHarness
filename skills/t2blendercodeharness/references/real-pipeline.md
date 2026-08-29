# Active Real Agent Pipeline

This reference describes the production path for
`t2blendercodeharness-v5-executable-director`. Historical deterministic and
template experiments remain available as explicit comparison artifacts, not as
silent production fallbacks.

## Per-case production path

```text
exact prompt + case metadata
 -> external structured DirectorAgent
 -> DirectorPlan evidence/order/trajectory/camera gate
 -> BlenderCodeAgent + local CodexExecProvider (provider_mode=model)
 -> static/runtime/case coverage gate
 -> frozen per-case blender_job.py
 -> D:\blender\blender.exe in an isolated directory
 -> proxy.blend + real proxy.mp4 + frames + telemetry
 -> RealArtifactGate + deterministic + independent oracle
 -> one shared visual review
 -> separate task and realism channels
```

`DirectorAgent.plan` is the only production planning entry point. It combines
prompt interpretation, event scheduling, actor/prop trajectories, interaction
lifecycle, and multi-target camera choreography. `SceneContract` and
`TrajectoryPlan` are compatibility projections for existing adapters.

`DirectorAgent` first uses an external OpenAI-compatible structured provider;
it reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` and calls the documented
`/v1/chat/completions` path. `BlenderCodeAgent` then receives the exact
DirectorPlan and verified `blender/lib/signatures.json`, and generates one
case-specific source through the local `CodexExecProvider`/`codex exec` CLI.
The source must pass schema, AST, runtime, and `case coverage gate` checks
before it is frozen. Provider kinds, model IDs, call IDs, request hashes, and
response hashes are recorded as separate stages. `CodexLocalProvider`/
`codex-local` is retained only as the explicit `rule_template_baseline`
diagnostic arm. There is no template fallback in the formal path. Provider,
schema, static, or coverage errors are `fail-closed`.

The optional local Codex visual-review diagnostic uses no external endpoint;
that statement applies only to visual review, not to the external Director
provider described above.

`template_baseline` is permitted only when explicitly requested for a historical
paired comparison. It is never an agent fallback. The official local inner
loop allows `max_inner_attempts=3`: a plan/non-compliance, Blender render, or
artifact failure regenerates a complete case candidate. Same-source retry is
zero, and after three failures the case is `NOT_RENDERED`; source mutation or
template substitution is a hard failure. Parallel rendering is capped at
**at most 12 workers** and the worker count, Blender path, render settings,
retry count, candidate attempt count, and hashes are recorded in the run
report.

## Evidence gates

| Stage | Required evidence | Failure behavior |
|---|---|---|
| prepared | exact prompt, DirectorPlan, projections, manifest, generated source | stop before Blender |
| code-ready | schema/AST/runtime and case coverage pass | `codegen_failed`, no video |
| rendered | real Blender return success, `.blend`, animation PNGs, telemetry | retry same source, then `render_failed` |
| artifact-valid | manifest, playable MP4, index, at least three readable samples, hashes | hard fail; no visual score |
| deterministic-pass | timebase, identity, events, trajectories, camera, constraints, oracle pass | hard fail; no patch preference |
| visually reviewed | compliant `gpt-5.6-luna`, `gpt-5.6-terra`, or auditable human/local review | unavailable/needs-human-review; no imputation |

The traceability chain is:

```text
prompt_hash -> plan_hash -> code_hash -> artifact_hash
```

`RealRunManifest` retains the prompt/plan/Director/code identity fields.
`RealArtifactReport` retains per-file `artifact_hashes` and their canonical
aggregate `artifact_hash`.

## Outer-loop protocol

The active dataset is the verbatim VBench-2.0 index
`dataset/vbench2-agent-training-index-v1`: exactly 60 train, 60 dev, and 20
frozen test cases. Six rounds are paired by ten-case families. The training
entry point validates that every prompt is identical to raw `prompt_en` and
rejects self-built or augmented prompt datasets; benchmark records have no
pre-authored event/entity/oracle labels, so DirectorAgent derives those at
runtime.
Each outer attempt renders 10 train + 10 dev cases; each round ends with a
separate full 60 train + 60 dev evaluation. There are at most five attempts per
round and a theoretical maximum of
`6 × (5 × 20 + 120) = 1320` real video executions.

The `1320` value counts committed protocol case slots. With at most three fresh
case candidates per slot, the worst-case candidate-generation budget is `3960`;
successful cases normally stop at the first passing candidate. The inner loop
is execution recovery only; it is not a local scene-repair loop and does not
modify the evaluator or the Harness owner.

The outer loop aggregates repeated train findings, requires one Harness owner,
regenerates the plan and source for that candidate version, and accepts only
strict train improvement with paired/full dev non-regression, no new hard
regressions, and no artifact-completion regression. The frozen test set is used
only for final blind verification.

The canonical append-only table is
`docs/t2blendercodeharness-agent-training-memory-v1.md`. Each row keeps the
exact prompt, absolute MP4 path or `NOT_RENDERED: reason`, Director/task/realism
scores, review provenance, Harness problem, fix location/method, delta, and
natural-language handling. JSONL evidence is written beside the round reports.

## Score boundary

Deterministic values are gate/diagnostic outputs. For an eligible shared visual
review:

```text
semantic_core = GM(applicable prompt_compliance, physical_plausibility,
                    object_trajectory, event_timing,
                    character_trajectory when an actor is applicable)
choreography = GM(camera_coverage,
                  camera_innovation when camera motion is required,
                  character_trajectory when an actor is applicable,
                  temporal_smoothness)
observability = GM(camera_coverage, visual_clarity)
task_score = .75 * semantic_core + .25 * observability
task_final_score = task_score
realism_vlm = GM(appearance_detail, physical_realism,
                  spatial_consistency, motion_naturalness,
                  visual_presentation)
reviewed_realism = .15 * geometry_score + .15 * frame_evidence_score + .70 * realism_vlm
```

Task and realism are separate channels. Frame statistics can report artifact
health only; they cannot create semantic or realism scores. The optional
`overall_vlm_score` is a reporting summary (`.70 * task_score + .30 *
realism_vlm`), not a replacement for either channel. A VLM transport or schema
failure stays unavailable and cannot be converted to zero or 100. Every
applicable dimension must have evidence references and completeness 1.0 for
the strict formal visual score; low confidence enters human-review rather than
being imputed.
