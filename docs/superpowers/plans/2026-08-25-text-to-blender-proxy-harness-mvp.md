# Text-to-Blender Proxy Harness MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first deterministic vertical slice of the approved Text-to-Blender proxy Harness in the empty `T2BlenderCode` repository.

**Architecture:** Keep runtime contracts in `src/videoact/contracts.py`; put prompt normalization, trajectory planning, adapter execution, evaluation, and orchestration in focused modules. Use fake backends and JSON artifacts for local tests, with interfaces ready for Blender CLI/MCP and optional VLM adapters.

**Tech Stack:** Python 3.11+, Pydantic v2, JSON Schema, pytest, standard-library hashing/subprocess/path APIs. No network or Blender dependency is required for the MVP test suite.

---

### Task 0: Freeze runtime contracts

**Files:**
- Create: `pyproject.toml`, `src/videoact/__init__.py`, `src/videoact/contracts.py`
- Create: `dataset/schemas/scene_contract.schema.json`, `dataset/schemas/trajectory_plan.schema.json`, `dataset/schemas/run_manifest.schema.json`
- Test: `tests/test_contracts.py`

- [ ] Write tests for valid contracts, positive duration/fps, unique IDs, event bounds, valid references, and monotonic frame ranges.
- [ ] Run `pytest tests/test_contracts.py -q` and verify the missing-model failure.
- [ ] Implement Pydantic models and validators.
- [ ] Export checked-in JSON Schemas from the same models.
- [ ] Run the contract tests and commit `feat: freeze scene trajectory and run contracts`.

### Task 1: Normalize scene prompts

**Files:**
- Create: `src/videoact/scene_contract.py`, `skills/scene-contract/SKILL.md`
- Test: `tests/test_scene_contract.py`

- [ ] Test deterministic extraction of walk/reach/grasp events, entities, ordered timings, and camera predicates.
- [ ] Implement `SceneContractBuilder.build(prompt)` with validation and explicit predicates.
- [ ] Document the skill boundary and run the tests.
- [ ] Commit `feat: add scene contract skill`.

### Task 2: Plan trajectories and camera shots

**Files:**
- Create: `src/videoact/trajectory.py`, `src/videoact/camera.py`, `skills/trajectory-planner/SKILL.md`
- Test: `tests/test_trajectory_planner.py`

- [ ] Test frame sampling, interpolation continuity, grasp attachment, event ordering, and camera coverage.
- [ ] Implement deterministic motion primitives and shot planning.
- [ ] Reject discontinuities and uncovered required events.
- [ ] Commit `feat: add camera and entity trajectory planning`.

### Task 3: Add fakeable Blender execution boundary

**Files:**
- Create: `src/videoact/blender_adapter.py`, `src/videoact/run_manifest.py`
- Create: `blender/white_materials.py`, `blender/proxy_scene.py`, `blender/probes.py`, `skills/blender-proxy-executor/SKILL.md`
- Test: `tests/test_blender_adapter.py`, `tests/test_proxy_scene.py`

- [ ] Test fake CLI construction, MCP JSONL logging, timeout/error results, and MCP-to-CLI fallback.
- [ ] Implement common `ExecutionResult`, backend interfaces, and manifest-bound script compilation.
- [ ] Implement white-material helper and probe payloads without importing Blender at test collection time.
- [ ] Commit `feat: add Blender CLI MCP proxy executor`.

### Task 4: Evaluate deterministic proxy outputs

**Files:**
- Create: `evaluator/prompt_predicates.py`, `evaluator/physics_metrics.py`, `evaluator/camera_metrics.py`, `evaluator/deterministic.py`
- Test: `tests/test_deterministic_evaluator.py`

- [ ] Test event order, support/grasp relations, attachment, velocity spikes, visibility, camera coverage, and hard gates.
- [ ] Implement findings with owner, severity, evidence, and repair route.
- [ ] Persist a stable JSON report and commit `feat: add deterministic proxy evaluator`.

### Task 5: Add bounded inner loop and orchestration

**Files:**
- Create: `src/videoact/inner_loop.py`, `src/videoact/orchestrator.py`, `skills/text-to-blender-proxy/SKILL.md`
- Test: `tests/test_inner_loop.py`, `tests/test_orchestrator_contract.py`

- [ ] Test ordered stages, resume/fail-closed behavior, immutable attempts, repair routing, selection, and six-attempt fallback.
- [ ] Implement the fake-backend vertical slice and artifact manifests.
- [ ] Run the complete test suite and commit `feat: add bounded proxy orchestration`.

### Task 6: Add outer-loop acceptance primitives

**Files:**
- Create: `src/videoact/evolution.py`, `src/videoact/outer_loop.py`
- Test: `tests/test_outer_loop_acceptance.py`, `tests/test_failure_aggregation.py`

- [ ] Test failure grouping, single-owner patch briefs, train improvement, dev non-regression, hard-regression rejection, and rollback records.
- [ ] Implement acceptance and JSONL optimization records.
- [ ] Run the complete test suite and commit `feat: add train dev harness acceptance loop`.

## Verification

Run from `C:\Users\sy\Desktop\T2BlenderCode`:

```powershell
pytest -q
python -m compileall src evaluator blender
```

The MVP is complete only when a prompt can pass through contract, plan, fake execution, deterministic evaluation, bounded repair, and final selection with immutable hash-bound artifacts.
