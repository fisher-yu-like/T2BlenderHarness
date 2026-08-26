# Trajectory Dataset and Harness Memory Report

## Dataset

`dataset/trajectory-v2` contains 80 unique multi-clause cases across 10 trajectory families:

- train: 50 cases;
- dev: 20 cases;
- test: 10 cases, frozen until final evaluation.

Each prompt specifies ordered actions such as walk, reach, grasp, lift, carry, place, and release, plus camera intent such as follow, orbit, dolly, close-up, or reveal. Each manifest record stores planner-derived event order, camera trajectory types, motion primitive types, character state count, attachment actions, and camera constraints.

## Pipeline results

- Complex dataset validator: 80/80 valid, 80 unique prompt hashes.
- Fake Harness benchmark: train 50/50 pass, dev 20/20 pass, test 10/10 pass; mean score 100.0 for all splits.
- Outer loop: `no_patch`; there is no repeated actionable train failure, so no Harness update was invented.
- Real path: 20 dev jobs generated and host-compiled; one complex dev job (`traj-01-06`) executed through Blender MCP and passed artifact gate + deterministic evaluator at 100.0.

## Memory

`out/training/trajectory-v2-run2/memory/harness_updates.jsonl` preserves the lifecycle for the current Harness candidate:

1. `proposal` — baseline candidate and fingerprints;
2. `train_evaluated` — 50-case train score;
3. `dev_evaluated` — 20-case dev score and hard-regression flag;
4. `test_evaluated` — frozen 10-case final score;
5. `no_patch` — no repeated failure, therefore no code update accepted.

Future updates must append a new memory ID and retain proposal, patch, train, dev, acceptance/rejection, and rollback events. Test case IDs are rejected from proposal selection. VLM-unavailable records remain metadata and do not become training labels.
