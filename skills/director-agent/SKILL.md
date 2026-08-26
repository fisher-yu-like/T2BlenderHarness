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

The runtime entry point is `videoact.director.DirectorAgent.plan`. Orchestrator,
inner-loop, and real-job preparation must call it rather than calling a legacy
parser and trajectory planner as separate external planning paths.

## Deterministic and provider-assisted modes

The current deterministic mode is network-free and reproducible. A future
provider-assisted interpreter may fill implicit details, but it must return the
same contracts, evidence fields, and fingerprints. Provider output is never
allowed to write Blender code directly.

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

## Verification and stop conditions

Before compiling Blender, validate the DirectorPlan hash, references, event
acyclicity, evidence, uncertainty, trajectory coverage, interaction lifecycle,
and camera visibility. Stop on an unresolved hard uncertainty, unknown stable
ID, dependency cycle, path collision, missing handoff, or unresolvable camera
target. Persist `director_plan.json` even when the later artifact gate fails.
