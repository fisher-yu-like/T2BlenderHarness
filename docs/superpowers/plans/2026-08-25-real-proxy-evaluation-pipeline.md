# Real Proxy Evaluation and Harness Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate real white-material proxy videos through Blender MCP, validate them, score eligible videos, and feed real failures into the existing train/dev Harness evolution gate.

**Architecture:** Keep host-side run state and artifact validation in `src/videoact/real_pipeline.py`; keep Blender-only code in generated job scripts under each immutable run directory. Use the existing contracts/evaluator/inner-loop modules, add an API-bound VLM provider, and persist all MetaHarness decisions as JSONL.

**Tech Stack:** Python 3.12 bundled runtime, Pydantic v2, Pillow, Blender Python via MCP, optional OpenAI-compatible Responses API over `urllib`, JSON/JSONL artifacts, pytest.

---

### Task 0: Freeze the real-run artifact contract

**Files:**
- Create: `src/videoact/real_pipeline.py`, `src/videoact/real_artifacts.py`
- Create: `tests/test_real_artifacts.py`

- [ ] Write tests for required artifact names, PNG readability, hash-bound manifest fields, stale/missing artifact hard failures, and resumable fingerprint matching.
- [ ] Run `pytest tests/test_real_artifacts.py -q` and verify the missing-module failure.
- [ ] Implement `RealRunManifest`, `RealArtifactGate`, `sample_frame_paths`, and `fingerprint_real_run` without importing Blender.
- [ ] Run the tests and persist an explicit `artifact_status` plus `hard_failures` list.

### Task 1: Generate Blender MCP jobs

**Files:**
- Create: `blender/real_proxy_job.py`, `scripts/prepare_real_jobs.py`
- Create: `tests/test_real_job_generation.py`

- [ ] Test that a validated contract/trajectory produces a job containing white material setup, camera/render settings, scene objects, keyframes, telemetry export, `.blend` save, MP4 output, and sampled PNG output.
- [ ] Implement `compile_real_proxy_job(plan, manifest, output_dir)` as a Blender Python source generator; it must not require `bpy` at host compile time.
- [ ] Implement `prepare_real_jobs.py --split calibration|train|dev|test --out-dir ... --dry-run` to write one immutable job per case and a job index.
- [ ] Run generation in dry-run mode and verify all job hashes and paths.

### Task 2: Execute a real calibration slice through Blender MCP

**Files:**
- Create: `scripts/run_real_mcp_job.py`
- Modify: `skills/blender-proxy-executor/SKILL.md`
- Create: `tests/test_real_mcp_state.py`

- [ ] Add a host-side state machine (`prepared -> executing -> rendered -> artifact_valid -> evaluated`) that fails closed on missing MCP responses.
- [ ] Execute one category-representative job through the connected Blender MCP and record request/response plus Blender version.
- [ ] Expand to the ten-case calibration slice only after the one-case artifact gate passes.
- [ ] Keep MCP execution opt-in; dry-run and fake backends remain available for unit tests.

### Task 3: Validate real telemetry and proxy videos

**Files:**
- Modify: `evaluator/deterministic.py`, `evaluator/camera_metrics.py`, `evaluator/physics_metrics.py`
- Create: `tests/test_real_evaluator_gate.py`

- [ ] Write tests for telemetry-vs-plan consistency, frame count, event observability, camera coverage, no penetration/support gaps, and hard failure on unreadable MP4/frames.
- [ ] Add real artifact inputs to `DeterministicEvaluator.evaluate_real(...)` while preserving the existing synthetic API.
- [ ] Run calibration records through rules first; record evaluator recall on injected telemetry failures and artifact completeness.

### Task 4: Add VLM video/frame evaluation

**Files:**
- Create: `evaluator/openai_vlm.py`, `scripts/evaluate_real_videos.py`
- Modify: `evaluator/vlm_judge.py`, `evaluator/aggregate.py`
- Create: `tests/test_openai_vlm_payload.py`, `tests/test_real_video_evaluation.py`

- [ ] Test provider payload construction without exposing Harness identity and validate offline response parsing.
- [ ] Implement an OpenAI-compatible Responses API provider using `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and explicit `OPENAI_VLM_MODEL`; never log secrets or raw authorization headers.
- [ ] Send only artifact-complete, deterministic-gate-eligible videos/frames to the VLM; persist response, confidence, score provenance, and unavailable/error status.
- [ ] Run the calibration slice through VLM evaluation when the provider configuration is present.

### Task 5: Connect MetaHarness optimizer to real records

**Files:**
- Create: `src/videoact/meta_harness.py`, `scripts/run_real_pipeline.py`
- Modify: `src/videoact/evolution.py`, `src/videoact/outer_loop.py`
- Create: `tests/test_meta_harness_real_records.py`

- [ ] Test real-record aggregation, one-owner patch brief generation, patch proposal persistence, train/dev rerun fingerprints, and test-split isolation.
- [ ] Implement `MetaHarnessOptimizer` around existing `aggregate_failures`, `build_patch_brief`, and `evaluate_candidate` primitives.
- [ ] Implement `run_real_pipeline.py` with `--mode dry-run|real-mcp`, `--split calibration|train|dev|test`, and explicit `--harness-version`/`--out-dir`.
- [ ] Execute calibration, then train/dev only when its artifact/evaluator gate passes; write `optimization_record.jsonl`.

### Task 6: Real verification and handoff

**Files:**
- Create: `docs/real-run-protocol.md`, `docs/real-run-report.md`

- [ ] Run unit tests, compileall, dataset validation, dry-run job generation, one real MCP case, and calibration artifact/evaluator checks.
- [ ] Record which cases produced real `.blend`, `.mp4`, telemetry, deterministic, and VLM artifacts.
- [ ] Do not claim Harness acceptance unless train improved, dev did not regress, and the patch owner is singular.
