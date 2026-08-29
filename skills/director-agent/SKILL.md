---
name: director-agent
description: Interpret a text-to-Blender prompt into an evidence-backed multi-entity DirectorPlan with event scheduling, collision-aware trajectories, and multi-target camera choreography.
---

# DirectorAgent

DirectorAgent is the planning front end of `T2Blendercodeharness`. It is not a
renderer and it must not silently replace the prompt with a fixed scene
template. Its output is a strict, hashable `DirectorPlan` plus compatibility
projections used by the existing Blender and evaluator adapters.

## Contract

Input is a `DirectorRequest` containing the exact prompt, `scene_id`, duration,
FPS, provider ID, and policy ID. The result contains:

- `DirectorEntity` records with stable IDs (`actor_a`, `actor_b`, `red_cube`,
  `blue_cup`), kind, role, label, and attributes;
- an acyclic `DirectorEvent` graph with participants, targets, dependencies,
  concurrency groups, and bounded time windows;
- `InteractionLifecycle` records for attach, transfer, detach, final owner, and
  final support;
- decision evidence, assumptions, uncertainties, provider/policy fingerprints;
- `director_plan_hash`, `DirectorTrajectories`, and `CameraPlan` projections.

The exact prompt is retained in `DirectorPlan.request.prompt`. Coordinates,
motion primitives, and camera shots are not invented by the interpreter.

Build the evidence-backed event order graph before applying generic carry or
handoff matching. Preserve `then`, `after`, `while`, reveal, subjectless,
pause, handoff, and return order. If a prompt contains reveal but the plan has
no explicit reveal event, emit a hard Director finding and stop before Blender
compilation; never let a generic carry match silently erase the reveal.

## Execution order

Use this order for every case:

```text
exact prompt
  -> interpret entities/directives/evidence
  -> schedule bounded event graph and interaction lifecycles
  -> compose per-entity trajectories and owner transitions
  -> compose multi-target camera shots and visibility predicates
  -> critic validates/repairs or rejects hard uncertainty
  -> project SceneContract + TrajectoryPlan
  -> compile Blender job
```

The production runtime entry point is `videoact.director.DirectorAgent.plan`.
Orchestrator and agent real-job preparation must call it rather than calling a
legacy parser and trajectory planner as separate external planning paths.
Historical fake/MCP compatibility and the explicit `template_baseline` arm may
call `DirectorAgent.plan_explicit_baseline`; that method is an explicit baseline
projection, not a production fallback, and must never be reached after a
dynamic Director failure.

## Deterministic and provider-assisted modes

The production interpreter is provider-assisted through the external
OpenAI-compatible structured provider. It reads `OPENAI_API_KEY` and
`OPENAI_BASE_URL`, calls `/v1/chat/completions`, and may fill implicit details,
but it must return the same contracts, evidence fields, and fingerprints. The
provider boundary uses strict structured output and fail-closes on timeout,
schema, JSON, or network errors. The deterministic interpreter remains
network-free and reproducible for explicit historical baseline comparisons
only. Provider output is never allowed to write Blender code directly; the
separate Blender code stage uses the local `CodexExecProvider`.

Provider-assisted decisions must cite prompt spans or provider evidence. An
assumption with no supporting evidence is invalid. A `hard` unresolved
uncertainty stops before Blender compilation; a `soft` uncertainty is retained
in the plan and exposed to the evaluator.

## Trajectory and camera obligations

Every declared actor and prop receives a trajectory entry. Actor lanes are
separated and fail closed on collision. A carried prop follows the declared
owner hand offset. A handoff must record the giver, receiver, transfer window,
and final owner; the Blender compiler validates giver detach and receiver
attach in the same window.

Every event that must be visible is covered by a camera shot. Handoff shots
must include giver, receiver, and prop; concurrent shots must include all
active lanes. Each target has a `visible` predicate, bounded occlusion, and
evidence-backed camera intent. Shot names alone are not quality evidence.

## Repair routing

Route one finding to one owner:

| Failure | Owner | Repair route |
|---|---|---|
| entity/evidence interpretation | `director_prompt_interpreter` | `scene_contract_repair` |
| dependency/timing graph | `director_event_scheduler` | `scene_contract_repair` |
| lane, path, attachment, final owner | `director_trajectory` | `trajectory_repair` |
| target visibility/coverage | `director_camera` | `camera_repair` |
| generated Blender source | `blender_code_agent` | `runtime_repair` |
| Blender execution/telemetry | `blender_executor` | `runtime_repair` |
| proxy artifact/render | `proxy_renderer` | `runtime_repair` |
| evaluator logic | `evaluator` | `candidate_recovery` |

Do not patch the dataset label or evaluator merely to improve a score. The
Director plan score is independent from deterministic, task, and realism
scores.

## Real-video trajectory feedback

When a real-video review repeatedly reports weak visible character/object
phase continuity, treat it as a `director_trajectory` proposal only when the
same normalized finding affects at least two distinct train cases. Inspect the
exact chronological PNGs and the corresponding proxy video before changing
the trajectory component; a valid plan or telemetry record alone is not proof
that grasp, carry, handoff, return, and placement phases are readable on
screen. The next outer-loop attempt must preserve stable entity identity,
include a visible anticipation/contact/settle interval, keep an ownership
change readable for more than one sampled frame, and re-check camera
observability without editing the evaluator formula or dataset labels. Record
the pre/post Harness hash, affected video paths, train/dev deltas, and the
natural-language acceptance or rejection decision in Memory. If frame-only
statistics are used because the external VLM is unavailable, label the output
as `frame_statistics_only-v1` artifact-health evidence. It must not be treated
as a semantic or realism score; semantic claims require a real VLM or an
explicit, auditable human review payload.

## Verification and stop conditions

Before compiling Blender, validate the DirectorPlan hash, references, event
acyclicity, evidence, uncertainty, trajectory coverage, interaction lifecycle,
and camera visibility. Stop on an unresolved hard uncertainty, unknown stable
ID, dependency cycle, path collision, missing handoff, or unresolvable camera
target. Persist `director_plan.json` even when the later artifact gate fails.

## Executable real-video checks

The Director contract is not evidence that the rendered video succeeded. For
each real run, route visible failures to one owner and retain the exact frame
paths. A carried prop must be checked for an attachment constraint, owner
change, and torso/prop penetration; a coordinate offset is not a handoff.
Camera feedback must inspect `max_occlusion`, `continuity_group`, target
coverage, and whether an `orbit` follows a sampled arc rather than a straight
line between endpoints. Missing visible evidence is a finding, not a default
pass.

Frame statistics are `frame_statistics_only-v1`: they provide
`artifact_health` and low-level observations, but their semantic dimensions
are `None` and cannot form a task or realism score. Use a real VLM or
validated human review for event, identity, physics, camera, and trajectory
claims. Patch proposals must expose `predicted_fixes`,
`predicted_regressions`, and `prediction_rationale`; use the shared
`function_library` and append-only `memory_entry` rather than untestable prose.
