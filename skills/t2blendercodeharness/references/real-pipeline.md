# Real Pipeline Gates

## Inner loop

```text
prompt
 -> SceneContract
 -> TrajectoryPlan + CameraPlan
 -> controlled Blender execution
 -> .blend + PNG animation + sampled frames + telemetry
 -> artifact gate
 -> deterministic evaluator
 -> optional VLM sampled-frame judge
 -> local repair or candidate promotion
```

Use the following promotion order:

| Stage | Minimum evidence | If missing |
|---|---|---|
| prepared | prompt, contract, plan, manifest, generated job | fail before Blender |
| executing | persisted state and MCP request | wait or fail closed |
| rendered | successful MCP response plus `.blend` and animation frames | do not evaluate |
| artifact-valid | required files, valid manifest, index, three readable PNGs, host MP4 | hard fail |
| evaluated | deterministic report with terminal status | do not train |
| VLM-scored | compliant structured response and provenance | keep `unavailable`, do not invent score |

The current project uses `RealRunStateMachine`, `RealArtifactGate`, `evaluate_real_runs.py`, and `evaluate_real_videos.py`. Use their reports as evidence instead of inferring success from directory names or the presence of one file.

## Outer loop

1. Read only train records to identify repeated failures.
2. Group by failure ID, owner, category, and severity.
3. Produce one patch proposal per owner with affected cases and evidence paths.
4. Apply at most one Harness-owner patch per candidate.
5. Rerun train and dev with stable evaluator/backend/data fingerprints.
6. Accept only strict train improvement and non-regressing dev; otherwise roll back.
7. Run frozen test only after acceptance and never use test findings to choose the patch.

## VLM boundary

Send only the prompt, compact scene contract, sampled frame data, and deterministic findings required by the judge schema. Do not send Harness version, candidate owner, patch identity, or unrelated workspace files. Network, policy, and schema errors are evidence that the VLM stage is unavailable, not negative labels.
