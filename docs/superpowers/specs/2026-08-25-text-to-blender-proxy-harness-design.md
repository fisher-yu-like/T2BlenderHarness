# Text-to-Blender Proxy Harness Design

## Goal

Build a deterministic, versioned harness that turns a natural-language scene prompt into a validated scene contract, an entity/camera trajectory plan, and a white-material Blender proxy execution record. The first implementation milestone uses fake execution and deterministic evaluation so the orchestration can be tested without requiring Blender or an external VLM.

## Boundaries

- Dynamic: prompt interpretation, scene plans, trajectories, repair suggestions, and candidate selection.
- Fixed: Pydantic contracts, JSON schemas, timebase, adapter result shape, manifest hashing, evaluator hard gates, and train/dev acceptance rules.
- The harness never runs generated Blender code directly. All execution goes through a CLI/MCP adapter that records the command or request.
- Every attempt is immutable and keyed by prompt hash, harness version, plan hash, and evaluator version.

## Runtime flow

```text
prompt -> SceneContractBuilder -> TrajectoryPlanner/CameraPlanner
       -> BlenderAdapter -> proxy manifest -> deterministic evaluator
       -> bounded repair loop -> selected candidate
       -> outer-loop failure aggregation and train/dev acceptance
```

The first vertical slice will support deterministic parsing of common scene phrases, linear/ease interpolation, camera coverage checks, a fake Blender backend, JSONL run records, and evaluator-guided acceptance. Real Blender CLI, MCP transport, ffmpeg, VLM judging, and the 40-case dataset remain adapters around these stable interfaces.

## Contracts

`SceneContract` contains entities, ordered timed events, relations, required visible events, physics constraints, and camera constraints. `TrajectoryPlan` contains a fixed timebase, per-entity key states/primitives, camera shots, observability checks, and validation intents. `RunManifest` binds source prompt, plan, harness, evaluator, backend, and produced artifacts. Pydantic models are the runtime authority; checked-in JSON Schemas are interoperability artifacts.

## Evaluation and recovery

Deterministic rules run before any optional VLM score. Hard failures include incomplete output, missing required events, invalid event order, discontinuous motion, and uncovered required camera events. Repair routes are explicit (`scene_contract_repair`, `trajectory_repair`, `camera_repair`, `runtime_repair`, `candidate_recovery`) and are bounded to six attempts. The outer loop changes one harness owner at a time and accepts only strict train improvement with no dev regression.

## Testing strategy

Tests are written before each implementation unit. Unit tests cover validation, normalization, interpolation, adapter logging/fallback, evaluator predicates, immutable attempt records, and acceptance gates. Integration tests exercise the full fake-backend path and assert stable hashes and stage order. Real Blender is optional and only runs when installed.
