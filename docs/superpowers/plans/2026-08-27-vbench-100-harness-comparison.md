# VBench-derived 100-case Harness comparison

## Objective

Measure whether the trained multi-entity Harness improves over the pretraining
Harness on a held-out, VBench-2.0-derived prompt set. This is a paired external
benchmark, not another training round: the 100 prompts are generated once, and
neither baseline nor current results may be used to patch the Harness during
this run.

## Frozen versions

- pretraining Harness: commit `7fe017a`; label `h-t2-hard-v4-pretraining-baseline`;
- current Harness: commit `306e4d2`; label `h-t2-hard-v4-director-prompt-elliptical-return-order-v1`;
- Blender: `D:\blender\blender.exe`, Blender 5.1.2;
- renderer: Blender CLI in background mode, real `.blend` plus PNG frames and MP4;
- render settings: EEVEE Next, 128×128, 1 sample, 12 fps, 6 seconds (72
  frames). This
  benchmark setting keeps full temporal video and real geometry/telemetry
  checks while making a 200-video paired run tractable on the local CPU;
- evaluator: the checked-in `real-v4-shared-evidence-separate-scores` path;
- visual review: local Codex frame review if the external VLM endpoint is unavailable;
  no synthetic or guessed visual scores.

## Dataset construction

Use the official VBench-2.0 prompt file and select five dimensions, 20 source
prompts each: `Camera_Motion`, `Human_Interaction`,
`Motion_Order_Understanding`, `Complex_Plot`, and
`Dynamic_Spatial_Relationship`. Each dimension contributes 14 train-like and
6 dev-like cases, for 70/30 total. The train/dev names describe benchmark
reporting only; no score is used for a Harness patch.

The original source prompt, source index, dimension, auxiliary cue, and source
SHA-256 are retained. The executable text is marked VBench-derived and adds
exactly two named performers and two proxy objects, ordered temporal events,
camera coverage constraints, and negative constraints. This lets the existing
Blender proxy renderer expose the trajectory/camera behavior while keeping the
source provenance auditable.

## Paired execution

1. Build `dataset/vbench-derived-100-v1` without touching
   `dataset/trajectory-v4-multi`.
2. Prepare 100 jobs from the baseline worktree and 100 jobs from the current
   worktree. Store them below separate benchmark roots so plans, code, videos,
   telemetry, and reports cannot overwrite each other.
3. Render all missing jobs with one global bounded executor. The target is 12
   concurrent Blender workers, but the runner must retry an individual failed
   job and lower concurrency if memory pressure or repeated timeouts appear.
4. For every job, require `proxy.blend`, telemetry, animation frames, a
   playable MP4, and the exact frame index before scoring. A failed Blender
   execution is retried; a missing artifact is never assigned a fake score.
5. Run the same deterministic/director/interaction and geometry audits on both
   roots. Keep task and realism scores separate.
6. Review exact sampled frames for every eligible video using the local Codex
   reviewer when the VLM endpoint is not usable. Record review source,
   sampled-frame hash, confidence, and review JSON path.
7. Join by `case_id` and report baseline/current scores, deltas, and
   win/tie/loss counts for all 100 cases and for each category and split.

## Acceptance and anti-overfit checks

- no baseline/current pair is dropped silently; missing pairs are listed as
  unevaluated rather than imputed;
- both Harness versions receive byte-identical prompt text and proxy specs;
- evaluator version, Blender binary hash/version, render settings, source
  dataset fingerprint, and sampled-frame policy match across the pair;
- dev is reported separately and is not used for any modification;
- improvement is claimed only when the current mean task score improves on
  the paired set without a dev task regression and without a realism regression;
- any category with fewer than 10 valid pairs is reported as insufficient,
  not generalized;
- geometry/proxy limitations are reported as a separate realism finding and
  are not hidden inside task score.

## Deliverables

- dataset files and provenance under `dataset/vbench-derived-100-v1`;
- baseline/current job roots under `out/benchmarks/vbench-100-current-vs-pretrain-v1`;
- paired per-case table with prompt, both proxy video paths, task/realism scores,
  deltas, findings, and artifact/review status;
- category/split summary, curve PNG, and an experiment report with commands,
  hashes, retries, missing cases, and conclusion;
- no Harness source change as a consequence of this comparison run.
