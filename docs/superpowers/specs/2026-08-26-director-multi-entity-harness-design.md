# DirectorAgent and Multi-Entity Harness Design

**Date:** 2026-08-26  
**Status:** Approved design, pending implementation plan  
**Project:** T2BlenderCode / T2Blendercodeharness

## 1. Objective

Evolve the current single-character, single-prop Harness into a staged multi-character, multi-object system without losing causal attribution during Harness self-evolution.

The work proceeds in this order:

1. Freeze the existing single-entity result as a baseline.
2. Repair current Harness, evaluator, dependency, test, and Skill inconsistencies.
3. Introduce a unified `DirectorAgent` interface whose internals remain independently testable.
4. Build and validate a harder multi-entity dataset.
5. Run five real-video train/dev rounds with one-owner patches.
6. Add model-assisted planning for concise and implicit prompts only after the multi-entity deterministic path is stable.

This is Harness/code evolution, not neural-network weight training.

## 2. Current Baseline and Known Gaps

The retained baseline is:

- run root: `out/training/single-five-rounds-v1`;
- completed evidence: round 1, attempt 3, 10 train and 10 dev cases;
- train deterministic mean: `100.0`;
- dev deterministic mean: `100.0`;
- train artifact-only realism mean: `68.6689`;
- dev artifact-only realism mean: `68.3419`;
- visual-review status: incomplete because the external endpoint returned HTTP 403;
- task and independent-review realism scores: unavailable, not zero.

This baseline must be indexed by immutable fingerprints. It must not be presented as a completed five-round result and must not be used as multi-entity training evidence.

The current implementation has the following blockers:

- `SceneContractBuilder` selects one primary target and relies on keyword extraction.
- `TrajectoryPlanner` assumes a literal `character`, one target, and fixed positions.
- `CameraPlanner` primarily follows one target.
- `trajectory-v3-hard` is not a balanced multi-entity training dataset: train contains at most one character and the corpus contains at most two props.
- deterministic scores can saturate while visible video quality remains unreviewed.
- several evaluator owner routes do not match the training Skill's allowed owners.
- finding deduplication is not sufficiently interaction-specific for repeated multi-entity failures.
- `scripts/train_real_harness.py` contains repeated Markdown writer definitions and the current full test run has one failing table-header assertion.
- `pyproject.toml` omits runtime packages used by the evaluator and video path.
- Skill/report model names are not consistently presented as `gpt-5.6-Luna` and `gpt-5.6-Terra`.

## 3. Chosen Architecture

The chosen approach is a staged DirectorAgent facade. The Codex Host invokes one DirectorAgent entry point, but the implementation retains internal component boundaries so failures can be assigned to one owner.

```text
Codex Host
  -> T2Blendercodeharness Skill
     -> Training Skill
     -> DirectorAgent Skill
        -> Prompt Interpreter
        -> Scene and Event Scheduler
        -> Character/Object Trajectory Composer
        -> Camera Choreography Planner
        -> Plan Critic and Repair
     -> BlenderCodeAgent
     -> Blender MCP/CLI Executor
     -> Artifact Gate
     -> Deterministic Evaluator
     -> Shared Visual Review
     -> MetaHarnessOptimizer
     -> One-owner train/dev acceptance gate
```

The old scene-contract and trajectory-planner responsibilities remain as internal contracts and validators. They are not exposed as competing top-level orchestration paths after DirectorAgent is introduced.

## 4. Component Contracts

### 4.1 DirectorAgent

`DirectorAgent` accepts a `DirectorRequest` containing:

- the exact original prompt;
- duration and FPS;
- available proxy assets and entity vocabulary;
- optional scene bounds and rendering limits;
- dataset case ID, split, and frozen fingerprints;
- planning provider configuration.

It emits a validated `DirectorPlan` containing:

- `entities`: stable IDs, kinds, roles, visual identities, and initial relations;
- `event_graph`: events, participants, targets, dependencies, concurrency groups, timing windows, and completion conditions;
- `entity_timelines`: per-entity states and motion primitives;
- `interaction_lifecycles`: contact, attach, transfer, detach, final owner, and final support;
- `camera_plan`: shots, targets, motion types, framing, visibility predicates, and covered events;
- `must_show`: required visible evidence;
- `negative_constraints`: forbidden collisions, penetrations, identity swaps, occlusions, and camera violations;
- `assumptions`: reasonable details added because the prompt omitted them;
- `uncertainties`: unresolved choices that cannot be silently invented;
- `evidence_map`: the prompt span, dataset rule, or default policy supporting each major plan decision;
- `fingerprints`: prompt, dataset, provider, DirectorAgent, planner-policy, and schema fingerprints.

Information is classified as:

1. explicit user requirement: immutable during local repair;
2. reasonable completion: permitted only when recorded in `assumptions` and `evidence_map`;
3. creative choice: optimizable but subordinate to explicit requirements and safety constraints.

The first implementation uses a deterministic planning provider. A Codex/model-assisted provider is added later behind the same interface. Provider choice must not change schemas, evaluator inputs, or artifact requirements.

### 4.2 Internal Director Components

#### Prompt Interpreter

Produces entities, roles, explicit constraints, unresolved references, and semantic evidence. It must support repeated entity kinds without collapsing identities.

#### Scene and Event Scheduler

Builds a dependency graph rather than a flat keyword list. It supports sequential, concurrent, conditional, pause/resume, return, and handoff events. Cycles are rejected unless explicitly represented as bounded repetitions.

#### Trajectory Composer

Generates a timeline for every moving entity. It resolves shared-space conflicts, preserves object identity, and emits explicit contact/attachment/transfer/detachment state changes.

#### Camera Choreography Planner

Optimizes event observability across several active subjects. It defines who must be visible, the permitted occlusion interval, camera motion, framing, and continuity constraints for every shot.

#### Plan Critic and Repair

Runs schema, semantic, temporal, collision, ownership, observability, and uncertainty checks before Blender code generation. Repairs are bounded and routed to one internal owner. Explicit prompt meaning cannot be rewritten during repair.

### 4.3 BlenderCodeAgent

The BlenderCodeAgent compiles a validated DirectorPlan. It may choose Blender implementation details such as constraints, animation curves, rigs, proxy geometry, materials, lighting, and render settings.

It must not reinterpret or change event order, participants, object identity, ownership transitions, must-show events, or negative constraints. Any required semantic change returns to DirectorAgent as a failed contract rather than being hidden in generated Blender code.

### 4.4 Blender Executor and Renderer

All Blender calls continue through the controlled adapter. The executable is `D:\blender\blender.exe`. Each case has an isolated output directory and a maximum of two render retries. Parallel execution targets 12 workers or more when resources permit, but workers may not overwrite a fingerprint-compatible completed run.

Required artifacts remain:

- run manifest;
- original prompt and DirectorPlan;
- scene contract compatibility projection;
- entity and camera trajectories;
- generated Blender job;
- `proxy.blend`;
- playable `proxy.mp4`;
- telemetry;
- frame index;
- readable event-aligned PNG samples;
- deterministic, geometry, frame-evidence, and visual-review reports.

## 5. Evaluator Design

### 5.1 Layer 0: Dataset and Input Integrity

Checks prompt hash, case ID, split, dataset fingerprint, oracle completeness, entity uniqueness, frozen evaluator configuration, and train/dev/test isolation. Any test leakage is a hard failure.

### 5.2 Layer 1: Director Semantics

Checks explicit prompt coverage, role binding, entity identity, event dependencies, concurrency, final ownership, final placement, assumptions, uncertainties, and evidence mapping. Unsupported over-completion is a finding rather than a hidden improvement.

This layer produces a separate `director_plan_score`. It is not added to task or realism scores.

### 5.3 Layer 2: Multi-Entity Trajectory and Interaction

Checks per-entity state continuity, velocity, acceleration/jerk, collision clearance, support/contact, attach/transfer/detach lifecycle, giver/receiver overlap at handoff, concurrent task synchronization, identity preservation, and final state consistency.

Finding root causes include the affected entity, object, and event instance, for example:

```text
attachment_lifecycle:actor_b:blue_cube:handoff_02
```

This prevents independent failures from being collapsed during deduplication.

### 5.4 Layer 3: Camera Choreography

Checks event coverage, multi-target visibility, handoff visibility, permitted and accidental occlusion, screen-space prominence, shot continuity, axis consistency, motion smoothness, framing, and whether follow/orbit/dolly/reveal behavior actually occurs.

Camera innovation is rewarded only when it improves observability or storytelling without violating continuity and task evidence.

### 5.5 Layer 4: Artifact and Runtime Consistency

The existing hard artifact gate remains mandatory. Telemetry must prove that planned entities, types, frame range, FPS, active camera, transforms, constraints, and ownership transitions were executed in Blender.

### 5.6 Layer 5: Shared Real-Video Review

The canonical report and Skill names are:

- `gpt-5.6-Luna`;
- `gpt-5.6-Terra`.

If an endpoint requires a different internal identifier, the mapping is explicit and the canonical name remains in reports. An external model or Codex local review uses the same schema, event-aligned frames, evidence requirements, and confidence threshold.

The two principal video scores remain separate:

```text
task_score = 0.20 * deterministic_score + 0.80 * task_visual_review

realism_score = 0.15 * geometry_score
              + 0.15 * frame_evidence_score
              + 0.70 * realism_visual_review
```

No shared review means `unavailable`, not zero. Artifact-only evidence remains capped and cannot be renamed as a real-video score.

### 5.7 Findings, Owners, and Scoring

Allowed patch owners are:

- `director_prompt_interpreter`;
- `director_event_scheduler`;
- `director_trajectory`;
- `director_camera`;
- `blender_code_agent`;
- `blender_executor`;
- `proxy_renderer`;
- `evaluator`.

Validators that discover a failure assign it to the component that produced the invalid behavior. `physics_validator` is not a patch owner.

Hard failures block visual scoring and patch promotion. Penalties remain deduplicated by interaction-specific root cause. Scores are accompanied by pass rates and failure counts so averages cannot hide hard regressions.

## 6. Dataset Design

### 6.1 Multi-Entity Dataset

Create `dataset/trajectory-v4-multi` with 140 immutable cases:

| Split | Cases | Purpose |
|---|---:|---|
| train | 50 | explicit, basic but comprehensive multi-entity behavior |
| dev | 60 | harder composition, paraphrase, concurrency, occlusion, and role changes |
| test | 30 | frozen unseen combinations and counterfactual constraints |

The five ten-case train families are:

1. two characters and two objects in sequential transfers;
2. single and repeated giver-to-receiver handoffs;
3. two characters acting concurrently on distinct objects;
4. multi-target camera work with occlusion and reveal;
5. role swaps, pause/resume, return, and crossing paths.

Train prompts are explicit enough to test multi-entity planning without confounding the later concise-prompt Director study. Dev uses two to three characters, two to four objects, new synonyms, concurrent events, repeated handoffs, similar-looking props, narrower paths, and unseen trajectory/camera combinations. Test adds role reversal, non-interchangeable events, counterfactual camera constraints, collision-prohibited crossings, and combined final-owner/final-support requirements.

Every record contains:

- unique original prompt and prompt hash;
- entity identities, kinds, roles, initial transforms, and visual discriminators;
- event dependency graph and concurrency groups;
- required interaction lifecycles;
- required camera evidence;
- negative constraints;
- authored oracle expectations independent of the generated plan;
- proxy-scene specification;
- difficulty metadata;
- split and dataset fingerprint.

Family-level and composition-level holdouts prevent near-duplicate variants from crossing splits.

### 6.2 Concise-Prompt Director Dataset

After the multi-entity phase passes, create `dataset/trajectory-v5-director`. It contains concise, implicit, partially specified, and ambiguous prompts. Its purpose is to train and evaluate assumption quality, uncertainty handling, evidence mapping, and provider-assisted planning. It is not mixed into the phase-2 multi-entity acceptance statistics.

## 7. Training Protocol

The phase-2 multi-entity run has five rounds. Each round introduces 10 unique train cases and uses 10 paired dev cases. An attempt runs real Blender videos for 10 train plus 10 paired dev cases, with at most five attempts per round.

| Round | Primary capability | Expected initial owner |
|---:|---|---|
| 1 | entity identity, roles, and ownership | `director_prompt_interpreter` |
| 2 | handoff and attachment lifecycle | `director_event_scheduler` or `director_trajectory`, selected by evidence |
| 3 | concurrency, conflict resolution, synchronization | `director_trajectory` |
| 4 | multi-target camera, occlusion, reveal | `director_camera` |
| 5 | composition, role swap, return and recovery | selected from repeated train evidence |

The table's expected owner is not permission to patch preemptively. Actual owner selection uses repeated train findings only.

Each attempt performs:

1. immutable job preparation;
2. 12-or-more-worker Blender rendering;
3. artifact and deterministic gates;
4. one eligible shared visual review per case;
5. immediate JSON and Markdown persistence;
6. repeated train-failure aggregation;
7. at most one owner patch proposal;
8. rerun of the same train and paired dev inputs;
9. acceptance, rejection, rollback, or `no_patch`.

Acceptance requires:

```text
paired_train_after > paired_train_before
paired_dev_after >= paired_dev_before
overall_dev_after >= overall_dev_before
hard_regression_count == 0
artifact_completion_rate_after >= artifact_completion_rate_before
```

A renderer or realism-focused patch additionally requires realism improvement without task-score regression.

At each round end, evaluate all train cases seen so far and all 60 dev cases using the accepted Harness. Fingerprint-compatible artifacts may be reused. Cases affected by a changed Harness fingerprint must be regenerated. The frozen 30-case test split runs once after all five rounds.

## 8. Memory and Reporting

The training Markdown table is updated immediately after each split. Each real case has one row with:

- round, attempt, split, and case ID;
- exact original prompt;
- absolute path to a real `proxy.mp4`, or `NOT_RENDERED` plus reason;
- Director plan score;
- task score;
- realism score;
- review source, model, confidence, and frame evidence;
- detected Harness problem in natural language;
- one owner;
- exact repair location and method;
- paired train, paired dev, and overall dev increase or decrease;
- natural-language handling decision and rationale.

The final report contains separate round curves for:

- Director plan score;
- task score;
- realism score;
- artifact completion rate;
- hard-failure rate;
- train-dev gap;
- findings grouped by owner.

No synthetic combined total is reported.

## 9. Skill Layout

Create `skills/director-agent/SKILL.md` as the top-level planning component Skill. Update:

- `skills/t2blendercodeharness/SKILL.md` to invoke DirectorAgent and preserve the evidence gates;
- `skills/t2blendercodeharness-training/SKILL.md` with the phase-2 dataset, five-round protocol, canonical model names, exact Memory columns, and one-owner rules;
- `skills/scene-contract/SKILL.md` to describe the compatibility projection and validation role;
- `skills/trajectory-planner/SKILL.md` to describe the Director internal trajectory contract;
- `skills/blender-proxy-executor/SKILL.md` with DirectorPlan fingerprints;
- `skills/harness-evolution/SKILL.md` with the new owner taxonomy and acceptance gates.

Skills specify how to run and validate the system. They do not silently edit themselves. Any future Skill evolution produces a reviewed proposal and passes tests, capability checks, and a forward test.

## 10. Implementation Phases and Gates

### Phase A: Baseline and Project Health

- index the retained baseline without copying or relabeling scores;
- consolidate the Markdown writer and make all tests pass;
- declare complete runtime/test dependencies;
- canonicalize model display names and internal mappings;
- align evaluator owner routing;
- pass the complete test suite and capability check.

### Phase B: Director Contracts and Multi-Entity Logic

- add DirectorRequest/DirectorPlan schemas;
- add deterministic Director internal components;
- generalize trajectories, interactions, camera targeting, telemetry, and Blender compilation;
- add component and integration tests;
- pass a real two-character/two-object Blender smoke run.

### Phase C: Dataset and Protocol

- generate and independently validate `trajectory-v4-multi`;
- freeze split, oracle, evaluator, and protocol fingerprints;
- update all relevant Skills and training commands;
- verify no family/composition leakage.

### Phase D: Five-Round Real Training

- execute five rounds using real Blender artifacts;
- update Memory after every split;
- accept only evidence-backed one-owner patches;
- run round-end overall evaluation;
- run the frozen test once and generate curves/tables.

### Phase E: Model-Assisted DirectorAgent

- add provider-assisted planning behind the existing DirectorAgent contract;
- build `trajectory-v5-director`;
- evaluate assumptions, uncertainties, evidence maps, plans, videos, and generalization separately from phase 2.

No phase advances when its hard gate is incomplete. In particular, VLM unavailability is reported and blocks any claim that depends on real-video semantic scoring.

