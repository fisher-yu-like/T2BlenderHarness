# Real Proxy Evaluation and Harness Evolution Design

## Goal

Turn the deterministic/fake Harness into a real evaluation pipeline that produces white-material Blender proxy videos through the connected Blender MCP, validates the resulting artifacts and trajectories, evaluates eligible videos with a VLM, and records MetaHarness patch proposals and train/dev acceptance decisions.

## Confirmed execution boundary

- Blender MCP is the real executor for this environment; local `blender`, `ffmpeg`, and `ffprobe` binaries are not available.
- Blender renders animation and sampled PNG frames itself. The host validates files with standard Python/Pillow and reads Blender telemetry JSON.
- “Train Harness” means code-level Harness evolution: analyze repeated failures, propose a patch for exactly one owner, rerun train/dev, and accept only strict train improvement with no dev regression. It does not silently train model weights.

## Pipeline

```text
real case manifest
  -> SceneContractBuilder
  -> TrajectoryPlanner + CameraPlanner
  -> generated Blender MCP job
  -> .blend + proxy.mp4 + sampled frames + telemetry
  -> artifact gate + deterministic evaluator
  -> VLM only for artifact-complete deterministic candidates
  -> per-case record
  -> MetaHarnessOptimizer failure aggregation
  -> one-owner patch brief
  -> train/dev rerun and acceptance gate
```

The first real run is a calibration slice containing one case from each of the ten failure categories. If all required artifacts are complete and injected-failure tests meet the evaluator contract, the same runner can execute the full train/dev/test splits. Each case is immutable under `real_runs/<harness>/<split>/<case_id>/`.

## Real artifact contract

Every successful case must contain `run_manifest.json`, `scene_contract.json`, `trajectory.json`, `camera_plan.json`, `blender_job.py`, `proxy.blend`, `proxy.mp4`, `frames/index.json`, at least three readable PNG frames, `telemetry.json`, `deterministic_report.json`, and optionally `vlm_report.json`. The manifest records MCP request status, Blender version, frame range, render settings, hashes, and evaluator/Harness versions. Missing or stale artifacts are hard failures and prevent VLM scoring.

## VLM boundary

The VLM client receives the original prompt, compact scene contract, sampled frame data URLs, and deterministic findings. It receives no Harness version, patch identity, or candidate owner. The response is validated by the existing strict schema and blended with deterministic scores using the existing weights. Network/API errors are recorded as `vlm_unavailable`; they never override deterministic hard gates.

## MetaHarness evolution

The optimizer consumes real train records, groups failures by normalized failure ID and owner, emits a one-owner `PatchBrief`, and stores a patch proposal plus before/after train/dev reports. The acceptance rule remains `train_after > train_before` and `dev_after >= dev_before`, with rejection on any hard dev regression. The frozen test split is evaluation-only and cannot select a patch.

## Safety and reproducibility

- MCP calls are logged before execution and every job is hash-bound to its source plan.
- Blender jobs use a fixed Eevee/Cycles-independent white proxy setup, fixed resolution/FPS/frame range, and no external textures.
- A real run is resumable only when prompt, plan, Harness, evaluator, Blender, and render fingerprints match.
- The pipeline has a `dry-run` mode that emits jobs without touching Blender and a `real-mcp` mode that requires explicit job execution.
