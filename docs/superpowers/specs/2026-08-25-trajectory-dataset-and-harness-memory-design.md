# Complex Trajectory Dataset and Harness Memory Design

## Goal

Build a larger train/dev/test dataset whose prompts explicitly describe ordered character/object trajectories and camera choreography, run it through the existing contract-first pipeline, and persist every Harness update as searchable memory.

## Dataset

Create `dataset/trajectory-v2` with 80 unique cases: 10 trajectory families × 8 prompt variants. Each family appears in train, dev, and test; variants are assigned 5/2/1 so the split tests generalization rather than category absence. Every record includes:

- a multi-clause prompt with ordered actions, temporal duration, and camera intent;
- expected event order and required camera motion types;
- entity/relation expectations and trajectory complexity metadata;
- evaluator and schema versions;
- a label placeholder that does not leak into train selection.

The original `dataset/` remains unchanged as the frozen 40-case baseline.

## Planner extensions

Extend prompt normalization to recognize `walk`, `reach`, `grasp`, `lift`, `carry`, `place`, `release`, and `reveal` in ordered clauses. Keep simple prompts backward compatible. Extend trajectory planning with phase states, motion primitives, attach/detach lifecycle, and validation intents. Extend camera planning with overview/follow, close-up, orbit, and dolly shots while requiring every event to be observable.

## Training memory

Store event-sourced JSONL under `training/memory/` or a specified output root. A Harness update receives a stable `memory_id` and records `proposal`, `patch_applied`, `train_evaluated`, `dev_evaluated`, `accepted`, `rejected`, `rollback`, or `no_patch` events. Each event stores parent/current Harness versions, owner, dataset/evaluator fingerprints, scores, affected cases, evidence paths, and notes. Test case IDs are forbidden from proposal selection.

## Pipeline

```text
trajectory-v2 manifest
 -> contract + trajectory + camera plans
 -> fake train/dev/test benchmark for full coverage
 -> real calibration/dry-run jobs when Blender MCP is available
 -> deterministic reports
 -> memory event log
 -> one-owner proposal or no_patch
 -> train/dev acceptance gate
 -> frozen test report
```

VLM-unavailable records remain operational metadata and never become numeric labels. No model-weight training is claimed; “training Harness” means versioned code-level evolution under the existing acceptance gate.
