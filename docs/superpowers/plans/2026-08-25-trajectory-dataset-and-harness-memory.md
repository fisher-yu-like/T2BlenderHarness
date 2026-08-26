# Complex Trajectory Dataset and Harness Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a complex trajectory-focused dataset, run train/dev/test through the existing Harness pipeline, and preserve every Harness update as event-sourced memory.

**Architecture:** Add a versioned dataset builder instead of modifying the frozen baseline. Extend the parser/planner only through backward-compatible tests. Add a memory store and a training runner that records lifecycle events around the existing benchmark and MetaHarness acceptance primitives.

**Tech Stack:** Python 3.12 bundled runtime, Pydantic v2, JSONL, pytest, existing fake benchmark, optional Blender MCP for calibration.

---

### Task 1: Add failing tests for complex prompts and memory

**Files:**
- Create: `tests/test_trajectory_dataset_v2.py`
- Create: `tests/test_harness_memory.py`

- [ ] Test that a long prompt produces ordered `walk -> reach -> grasp -> lift -> carry -> place -> release` events and camera constraints for follow/orbit/dolly/closeup.
- [ ] Test that the trajectory plan contains extra states, motion primitives, attach/detach events, and observable camera shots.
- [ ] Test that the dataset builder emits 80 unique cases with 50/20/10 train/dev/test cases and expected trajectory metadata.
- [ ] Test that memory records proposal, patch, train, dev, and acceptance events with one owner and rejects test-case leakage.
- [ ] Run the focused tests and confirm they fail before implementation.

### Task 2: Extend contract, trajectory, and camera planning

**Files:**
- Modify: `src/videoact/scene_contract.py`
- Modify: `src/videoact/trajectory.py`
- Modify: `src/videoact/camera.py`
- Modify: `tests/test_scene_contract.py`
- Modify: `tests/test_trajectory_planner.py`

- [ ] Implement ordered action extraction without changing existing simple-prompt event results.
- [ ] Add phase-aware character/target states and explicit attach/detach lifecycle for lift/carry/place/release prompts.
- [ ] Add motion primitives for phase transitions and camera shot types driven by prompt constraints.
- [ ] Preserve event coverage, one-based frame bounds, continuity, and existing evaluator contracts.
- [ ] Run all planner tests and inspect a representative complex plan JSON.

### Task 3: Build the trajectory-v2 dataset

**Files:**
- Create: `scripts/build_trajectory_dataset.py`
- Create: `tests/test_trajectory_dataset_v2.py`
- Create: `dataset/trajectory-v2/` generated manifest/splits/labels files
- Modify: `scripts/validate_dataset.py` only if a reusable root/version option is needed

- [ ] Implement 10 families × 8 variants with complex prompts, explicit action order, camera intents, duration, FPS, and expected plan metadata.
- [ ] Use deterministic prompt hashes and disjoint 50/20/10 splits with category coverage in every split.
- [ ] Validate unique prompts, entity/relation fields, required events, camera requirements, and label references.
- [ ] Build the dataset and run its validator before any training benchmark.

### Task 4: Implement event-sourced Harness memory

**Files:**
- Create: `training/harness_memory.py`
- Create: `tests/test_harness_memory.py`
- Create: `training/memory/README`-free JSONL output contract via code/doc references

- [ ] Implement `HarnessMemoryStore` with append-only events, stable memory IDs, hashable fingerprints, owner validation, and forbidden test IDs.
- [ ] Implement retrieval by owner/failure/category and a compact history export for future Harness decisions.
- [ ] Ensure rejected and rolled-back updates remain in memory rather than being deleted.
- [ ] Run memory unit tests, including a rejected dev regression and a successful acceptance.

### Task 5: Run the full training/evaluation pipeline

**Files:**
- Create: `scripts/train_harness_with_memory.py`
- Create: `tests/test_training_pipeline_memory.py`
- Create: `out/training/trajectory-v2/` reports and memory artifacts

- [ ] Run fake benchmark train/dev/test over the 80-case dataset for full coverage.
- [ ] Record dataset/evaluator/Harness fingerprints and lifecycle events for every candidate update or `no_patch` outcome.
- [ ] Use train failures only for proposals, require dev rerun before acceptance, and keep test blind until the end.
- [ ] If the current connected Blender is available, prepare a small real calibration slice without rerendering all 80 cases.
- [ ] Export a final training report that distinguishes deterministic scores, VLM-unavailable metadata, accepted code updates, and no-op iterations.

### Task 6: Final verification

**Files:**
- Create: `docs/trajectory-dataset-and-memory-report.md`

- [ ] Run dataset validation, focused tests, full pytest, compileall, and memory integrity checks.
- [ ] Confirm no test case ID appears in proposal input or training selection.
- [ ] Confirm every memory event has an owner/version/fingerprint and no accepted update lacks train/dev evidence.
- [ ] Report whether any Harness update was accepted; do not call a no-patch run “trained”.
