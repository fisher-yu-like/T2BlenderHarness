---
name: t2blendercodeharness-training
description: Train and evaluate T2Blendercodeharness with real Blender CLI proxy videos on trajectory-v4-multi, five outer-loop rounds, paired train/dev gates, separate Director/task/realism scores, and append-only Markdown memory.
---

# T2Blendercodeharness Training

This skill is the real-video outer-loop protocol for `T2Blendercodeharness`.
The object being optimized is the Harness, not the dataset labels, evaluator
policy, Blender installation, or a neural model. Use `director-agent` for
prompt interpretation and keep the Director, task, and realism channels
separate.

## Frozen protocol

- Dataset: `dataset/trajectory-v4-multi` — exactly 50 train, 60 dev, 30 frozen test.
- Output root: `out/training/multi-five-rounds-v1`.
- Memory: `docs/t2blendercodeharness-multi-training-memory-v1.md`.
- Report: `docs/t2blendercodeharness-multi-training-report-v1.md`.
- Blender: `D:\blender\blender.exe`.
- Workers: 12 by default; use 12 workers for the standard run, with parallel rendering per immutable case directory.
- External endpoint IDs: lowercase `gpt-5.6-luna` or `gpt-5.6-terra`.
- Canonical report names: `gpt-5.6-Luna` and `gpt-5.6-Terra`.
- Rounds: exactly five. Each round may use at most five attempts.

Do not start batch training until the full test suite, capability check,
dataset validator, and a real two-actor/two-prop Blender smoke pass.

## Per-attempt protocol

Each attempt runs at most 10 train cases (unique) and 10 paired dev cases. For
every case, execute this chain:

```text
prompt
 -> DirectorAgent exact-prompt DirectorPlan
 -> event graph / interaction lifecycle
 -> multi-entity trajectories / multi-target camera
 -> Blender CLI generated job (with retry on failure)
 -> proxy.blend + PNG sequence + host MP4 + telemetry
 -> RealArtifactGate
 -> deterministic + independent Director evaluator
 -> one shared visual review for eligible real videos
 -> separate Director plan, task, and realism scores
```

Use `scripts/prepare_real_jobs.py`,
`scripts/render_proxy_jobs_parallel.py`, `scripts/evaluate_real_runs.py`,
and `scripts/evaluate_real_videos.py`. A failed Blender job is retried using
the configured retry policy; after retries are exhausted, record an explicit
`NOT_RENDERED` row with the error and never invent a video path or score.

The visual review is `if/else`: use external `gpt-5.6-luna` or
`gpt-5.6-terra` when the endpoint is available; otherwise use the Codex local
review request/response path. If neither produces an auditable review, write
`unavailable` and exclude the numeric visual score. Never turn endpoint
failure, missing frames, or low confidence into zero.

## Score channels

`director_plan_score` is an independent deterministic score for entity
identity, evidence/uncertainty, event dependencies, interaction lifecycle,
path collision, and multi-target visibility. It is not added to any other
score.

`task_score` evaluates prompt compliance, event timing, character/object
trajectories, camera coverage/innovation, temporal smoothness, and clarity.

`realism_score` uses the separate realism evaluator. Geometry and rendered
frame evidence are safeguards; independent visual review is the main realism
signal. Without a complete review, retain only capped `artifact_only_proxy`
and mark realism `not_established`.

## Outer-loop evolution gates

1. Aggregate only repeated train failures, grouped by failure ID, owner,
   category, severity, and root cause. A proposal requires two distinct train
   cases and exactly one owner.
2. Patch at most one Harness component. Do not edit the frozen dataset or
   evaluator to manufacture an improvement.
3. Re-run the same paired train/dev cases with identical dataset, evaluator,
   Director provider/policy, Blender, render settings, and fingerprints.
4. Accept only if strict paired train scores improve, paired dev does not
   regress, overall dev does not regress, hard failures do not increase, and
   artifact completion does not decrease.
5. For renderer/realism patches, realism must improve and task score must not
   decrease. Otherwise reject and record rollback.
6. At round end, evaluate all train cases seen so far and all 60 dev cases.
   Reuse artifacts only when fingerprints are identical.
7. After round five, run the frozen 30-case test once in no-proposal mode.

## Memory table is mandatory

Update the Markdown memory immediately after each split and after each round.
Every rendered or `NOT_RENDERED` case gets one row with natural-language
explanation:

| Round | Attempt | Split | Case ID | Prompt | Proxy video address | Director plan score | Task score | Realism score | Review source/confidence | Harness problem | Harness fix location/method | Before→after delta | Natural-language handling |
|---:|---:|---|---|---|---|---:|---:|---:|---|---|---|---:|---|

The table must preserve exact prompt text, absolute `proxy.mp4` path when it
exists, `NOT_RENDERED` plus failure reason otherwise, and the immutable
DirectorPlan hash. A failed proposal records owner, changed component, why it
was accepted/rejected/rolled back, and why dev-only improvement was not used.
Keep JSON evidence alongside the table:

```text
round-XX/attempt-YY/real/{train,dev}/
round-XX/overall/real/{train,dev}/
round-XX/attempt_report.json
round-XX/overall_report.json
round-XX/patch_manifest.json
memory/harness_updates.jsonl
```

Plot separate Director plan, task, realism, artifact completion, hard-failure,
train-dev gap, and owner-finding curves. A missing visual review is shown as
unavailable, never as a zero point.

## Commands

```powershell
uv run python scripts/validate_multi_entity_dataset.py --dataset-root dataset/trajectory-v4-multi
uv run python scripts/train_real_harness.py --mode protocol --dataset-root dataset/trajectory-v4-multi --round-root out/training/multi-five-rounds-v1
uv run python scripts/train_real_harness.py --mode attempt --round 1 --attempt 1 --dataset-root dataset/trajectory-v4-multi --round-root out/training/multi-five-rounds-v1 --blender-bin D:\blender\blender.exe --workers 12 --vlm-model gpt-5.6-luna --markdown-path docs/t2blendercodeharness-multi-training-memory-v1.md
uv run python scripts/train_real_harness.py --mode overall --round 1 --dataset-root dataset/trajectory-v4-multi --round-root out/training/multi-five-rounds-v1 --blender-bin D:\blender\blender.exe --workers 12 --vlm-model gpt-5.6-luna --markdown-path docs/t2blendercodeharness-multi-training-memory-v1.md
```

Before claiming training success, rerun the full test suite, capability check,
dataset validator, artifact gate, and final blind test report. Do not claim
trained, accepted, or video-evaluated without the corresponding real artifact
and append-only Memory evidence.
