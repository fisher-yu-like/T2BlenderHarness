# DirectorAgent Multi-Entity Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the current Harness, introduce a testable DirectorAgent with multi-character/multi-object planning, build a frozen 50/60/30 dataset, update all Skills, and gate five rounds of real Blender training on complete verification.

**Architecture:** `DirectorAgent` becomes the only planning entry point used by orchestration. Its interpreter, scheduler, trajectory composer, camera choreographer, and critic remain separate modules and emit a strict `DirectorPlan`; SceneContract and TrajectoryPlan become compatibility projections for Blender and existing evaluators. Director plan, task, and realism scores remain separate.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest 8, Pillow, imageio/imageio-ffmpeg, Blender 5.1.2 CLI at `D:\blender\blender.exe`, Markdown/JSONL memory, OpenAI-compatible shared visual review.

---

Do not start batch training until the full test suite, capability check, dataset validation, and a real two-character/two-object Blender smoke all pass.

### Task 1: Make the runtime reproducible

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_project_runtime.py`

- [x] Write a failing test that parses `pyproject.toml` and requires `pydantic`, `typing-extensions`, `Pillow`, `imageio`, and `imageio-ffmpeg` in project dependencies.
- [x] Run `uv run --extra test python -m pytest tests/test_project_runtime.py -q -p no:cacheprovider --basetemp .pytest-tmp-runtime`; expect failure for missing packages.
- [x] Set runtime dependencies to:

```toml
dependencies = [
  "pydantic>=2.0,<3.0",
  "typing-extensions>=4.12",
  "Pillow>=10.0",
  "imageio>=2.34",
  "imageio-ffmpeg>=0.5",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]
```

- [x] Re-run the test and `uv run --extra test python -c "from PIL import Image; import imageio; import pydantic; print('runtime-ok')"`; expect pass and `runtime-ok`.
- [x] Commit with `git commit -m "build: declare real video runtime dependencies"`.

### Task 2: Consolidate the Memory writer and restore a green baseline

**Files:**
- Modify: `scripts/train_real_harness.py`
- Modify: `tests/test_real_batch_discovery.py`

- [x] Update the existing Memory test to require these exact headers:

```python
required = (
    "轮数", "Attempt", "Split", "Case ID", "Prompt", "Proxy 视频地址",
    "Director plan 分", "Task score", "Realism score", "Review",
    "检测出的 Harness 问题", "Owner", "修复位置/方法", "提升或下降", "自然语言处理",
)
```

- [x] Use a row containing round, attempt, split, case ID, exact prompt, real video path, three separate scores, review source/confidence, one owner, fix, delta, and natural-language decision.
- [x] Run the focused test; expect failure because five duplicate writer definitions exist and the active schema is incomplete.
- [x] Delete four duplicate `write_training_memory_markdown` definitions and retain one implementation. Escape pipes/newlines and render missing scores as `unavailable`, never `0`.
- [x] Run `uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-health`; expect at least 140 passed and zero failures.
- [x] Commit with `git commit -m "fix: consolidate Harness training memory writer"`.

### Task 3: Index the retained baseline truthfully

**Files:**
- Create: `training/baselines/single-v1-round01-attempt03.json`
- Create: `scripts/index_training_baseline.py`
- Create: `tests/test_training_baseline.py`

- [x] Write a failing test requiring run root `out/training/single-five-rounds-v1`, round 1, attempt 3, 10 train, 10 dev, deterministic means 100, unavailable visual review, null task score, and a 64-character source-report SHA-256.
- [x] Implement an indexer that reads `round-01/attempt_report.json`, validates the expected identity/counts, computes its hash, and writes only the summary JSON.
- [x] Run:

```powershell
uv run python scripts/index_training_baseline.py --run-root out/training/single-five-rounds-v1 --out training/baselines/single-v1-round01-attempt03.json
uv run --extra test python -m pytest tests/test_training_baseline.py -q -p no:cacheprovider --basetemp .pytest-tmp-baseline
```

- [x] Stop instead of inventing values if source evidence is missing.
- [x] Commit with `git commit -m "docs: index retained single-entity baseline"`.

### Task 4: Add strict DirectorAgent contracts

**Files:**
- Create: `src/videoact/director_contracts.py`
- Create: `tests/test_director_contracts.py`
- Modify: `src/videoact/__init__.py`

- [x] Write tests for two actors, two props, concurrency, handoff lifecycle, assumptions, uncertainties, evidence, and fingerprints. Add rejection tests for unknown references, duplicate IDs, unsupported assumptions, dependency cycles, and invalid final owners.
- [x] Run the test and verify `ModuleNotFoundError`.
- [x] Implement `DirectorRequest`, `DirectorEntity`, `DirectorEvent`, `InteractionLifecycle`, `DirectorDecisionEvidence`, `DirectorPlan`, and `DirectorResult` as Pydantic models with `extra="forbid"`.
- [x] Use these key fields:

```python
class DirectorEvent(ContractModel):
    id: str
    action: str
    participant_ids: list[str]
    target_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    concurrency_group: str | None = None
    start: float
    end: float

class InteractionLifecycle(ContractModel):
    id: str
    prop_id: str
    giver_id: str | None = None
    receiver_id: str | None = None
    attach_event_id: str
    transfer_event_id: str | None = None
    detach_event_id: str
    final_owner_id: str | None = None
    final_support_id: str | None = None
```

- [x] Run `tests/test_director_contracts.py`; expect all pass.
- [x] Commit with `git commit -m "feat: add strict DirectorAgent contracts"`.

### Task 5: Interpret prompts and schedule event graphs

**Files:**
- Create: `src/videoact/director_prompt.py`
- Create: `src/videoact/director_schedule.py`
- Create: `tests/test_director_prompt.py`
- Create: `tests/test_director_schedule.py`

- [x] Write failing interpretation tests using named actors and repeated prop kinds. Require stable IDs such as `actor_a`, `actor_b`, `red_cube`, and `blue_cube`, explicit giver/receiver roles, and evidence linked to prompt spans.
- [x] Write failing scheduler tests for sequential transfer, concurrent independent carrying, handoff, pause/resume, and return; require acyclic dependencies and bounded timing.
- [x] Implement `DeterministicPromptInterpreter.interpret(request)` without coordinates or camera shots.
- [x] Implement `EventScheduler.schedule(request, interpretation)` returning events and interaction lifecycles without trajectories.
- [x] Run both test files; expect pass.
- [x] Commit with `git commit -m "feat: interpret and schedule multi-entity prompts"`.

### Task 6: Compose collision-aware multi-entity trajectories

**Files:**
- Create: `src/videoact/director_trajectory.py`
- Create: `tests/test_director_trajectory.py`
- Modify: `src/videoact/contracts.py`

- [x] Add failing tests for per-entity paths, concurrent lane separation, attach/transfer/detach, current owner, final support, and collision failure.
- [x] Extend `AttachmentEvent.action` to `attach | transfer | detach` and motion primitives only with tested values needed by the dataset (`arc`, `s_curve`, `zigzag`, `bezier`).
- [x] Implement `MultiEntityTrajectoryComposer` using proxy-scene transforms, independent actor lanes, prop-to-owner coupling, explicit transfers, and fail-closed conflict resolution.
- [x] Remove all new-code assumptions that the only actor ID is `character`.
- [x] Run `tests/test_director_trajectory.py`, `tests/test_trajectory_planner.py`, and `tests/test_contracts.py`; expect pass.
- [x] Commit with `git commit -m "feat: compose multi-entity trajectories"`.

### Task 7: Choreograph multi-target cameras

**Files:**
- Create: `src/videoact/director_camera.py`
- Create: `tests/test_director_camera.py`
- Modify: `src/videoact/contracts.py`

- [x] Write failing tests for a giver/receiver/prop handoff two-shot, concurrent action coverage, bounded occlusion/reveal, axis continuity, and all must-show events.
- [x] Extend camera shot fields with visibility predicates, maximum occlusion, continuity group, and evidence-backed innovation intent.
- [x] Implement `MultiTargetCameraChoreographer.compose(request, interpretation, schedule, trajectories)` without inferring quality from shot names.
- [x] Run camera, trajectory-planner, and evaluator-completeness tests; expect pass.
- [x] Commit with `git commit -m "feat: choreograph multi-target cameras"`.

### Task 8: Build the DirectorAgent facade and compatibility projection

**Files:**
- Create: `src/videoact/director.py`
- Create: `src/videoact/director_projection.py`
- Create: `tests/test_director_agent.py`
- Modify: `src/videoact/orchestrator.py`
- Modify: `src/videoact/inner_loop.py`
- Modify: `scripts/prepare_real_jobs.py`

- [ ] Write failing tests that require exact-prompt preservation, provider/policy fingerprints, a DirectorPlan, and compatible SceneContract/TrajectoryPlan outputs. Reject unresolved hard uncertainty before Blender compilation.
- [ ] Implement the facade in this order:

```python
interpretation = interpreter.interpret(request)
schedule = scheduler.schedule(request, interpretation)
trajectories = trajectory.compose(request, interpretation, schedule)
camera = camera_choreographer.compose(request, interpretation, schedule, trajectories)
director_plan = critic.validate_and_repair(
    request=request,
    interpretation=interpretation,
    schedule=schedule,
    trajectories=trajectories,
    camera=camera,
)
return projector.project(director_plan)
```

- [ ] Persist `director_plan.json`, `scene_contract.json`, `trajectory.json`, and `camera_plan.json`.
- [ ] Route Orchestrator, inner loop, and real-job preparation exclusively through DirectorAgent while retaining legacy compatibility projections.
- [ ] Run Director, orchestrator, real-job, and proxy-scene tests; expect pass.
- [ ] Commit with `git commit -m "feat: route planning through DirectorAgent"`.

### Task 9: Generalize Blender compilation and telemetry

**Files:**
- Modify: `blender/real_proxy_job.py`
- Modify: `src/videoact/real_artifacts.py`
- Create: `tests/test_multi_entity_blender_job.py`

- [ ] Write a failing generated-job test for two actors and two props. Require every stable ID, per-entity animation, transfer constraints, multi-target camera data, current-owner telemetry, visibility, and interaction state.
- [ ] Change entity kind/role lookup to DirectorPlan or proxy specification; never infer character kind from `entity_id == "character"`.
- [ ] Animate every trajectory. Compile transfer as a validated giver detach plus receiver attach in the same handoff window.
- [ ] Extend telemetry and artifact fingerprinting with the DirectorPlan hash.
- [ ] Run multi-job, real-job, and artifact tests; expect pass.
- [ ] Commit with `git commit -m "feat: compile and audit multi-entity Blender jobs"`.

### Task 10: Add Director and interaction evaluator layers

**Files:**
- Create: `evaluator/director_metrics.py`
- Create: `evaluator/interaction_metrics.py`
- Modify: `evaluator/deterministic.py`
- Modify: `evaluator/physics_metrics.py`
- Modify: `evaluator/findings.py`
- Create: `tests/test_director_evaluator.py`
- Create: `tests/test_interaction_evaluator.py`

- [ ] Write failing tests for unsupported assumptions, missing evidence, dependency mismatch, identity swap, incomplete handoff contact, wrong final owner, path collision, and multi-target invisibility.
- [ ] Require interaction-specific roots such as `attachment_lifecycle:actor_b:blue_cube:handoff_02`.
- [ ] Add an independent `director_plan_score`; do not add it to deterministic, task, or realism scores.
- [ ] Route failures only to:

```python
ALLOWED_OWNERS = {
    "director_prompt_interpreter", "director_event_scheduler", "director_trajectory",
    "director_camera", "blender_code_agent", "blender_executor", "proxy_renderer", "evaluator",
}
```

- [ ] Remove `physics_validator` as a patch owner; route a physics finding to the component that produced the invalid behavior.
- [ ] Run all evaluator, failure-aggregation, and policy tests; expect pass.
- [ ] Commit with `git commit -m "feat: evaluate Director plans and interactions"`.

### Task 11: Strengthen one-owner evolution and anti-overfit gates

**Files:**
- Modify: `src/videoact/evolution.py`
- Modify: `src/videoact/meta_harness.py`
- Modify: `src/videoact/outer_loop.py`
- Modify: `scripts/train_real_harness.py`
- Modify: `tests/test_failure_aggregation.py`
- Modify: `tests/test_meta_harness_real_records.py`
- Modify: `tests/test_outer_loop_acceptance.py`

- [ ] Add failing tests that group by failure ID, owner, category, severity, and root cause; require two distinct train cases; reject mixed-owner proposals; and refuse artifact-rate regression.
- [ ] Update `OWNER_FILES` to the approved Director/Blender/evaluator taxonomy.
- [ ] Require strict paired-train improvement, non-regressing paired dev and overall dev, zero hard regressions, and non-regressing artifact completion.
- [ ] For renderer/realism patches, require realism improvement with no task-score regression.
- [ ] Persist every failed check in the acceptance record.
- [ ] Run focused evolution and Memory tests; expect pass.
- [ ] Commit with `git commit -m "feat: enforce multi-entity one-owner acceptance"`.

### Task 12: Canonicalize Luna/Terra reporting

**Files:**
- Modify: `evaluator/openai_vlm.py`
- Modify: `evaluator/shared_review.py`
- Modify: `scripts/train_real_harness.py`
- Modify: `tests/test_vlm_judge_contract.py`
- Modify: `tests/test_real_video_evaluation.py`

- [ ] Add failing tests for `canonical_vlm_name("gpt-5.6-luna") == "gpt-5.6-Luna"` and the equivalent Terra mapping.
- [ ] Keep explicit endpoint IDs separate from canonical report names:

```python
VLM_MODELS = {
    "gpt-5.6-Luna": "gpt-5.6-luna",
    "gpt-5.6-Terra": "gpt-5.6-terra",
}
```

- [ ] Ensure request payloads use endpoint IDs while reports, Skills, CLI help, and Markdown use canonical names.
- [ ] Run VLM payload, schema, and real-video evaluation tests; expect pass.
- [ ] Commit with `git commit -m "fix: canonicalize Luna and Terra reporting"`.

### Task 13: Build and freeze `trajectory-v4-multi`

**Files:**
- Create: `scripts/build_multi_entity_dataset.py`
- Create: `scripts/validate_multi_entity_dataset.py`
- Create: `tests/test_multi_entity_dataset.py`
- Generate: `dataset/trajectory-v4-multi/manifest.jsonl`
- Generate: `dataset/trajectory-v4-multi/proxy_specs.jsonl`
- Generate: `dataset/trajectory-v4-multi/labels.jsonl`
- Generate: `dataset/trajectory-v4-multi/splits.json`
- Generate: `dataset/trajectory-v4-multi/metadata.json`

- [ ] Write failing tests requiring exactly 50 train, 60 dev, and 30 test cases; unique hashes; no family/composition leakage; authored event graphs/interactions/camera evidence/negative constraints; and a harder dev/test distribution.
- [ ] Implement five ten-case train families: sequential transfers, repeated handoffs, concurrent independent work, occlusion/reveal, and role-swap/pause/return/crossing paths.
- [ ] Give dev two to three actors and two to four props with unseen compositions. Freeze test with role reversals, counterfactual camera constraints, prohibited crossings, and combined final-owner/final-support requirements.
- [ ] Sort canonical JSON by case ID and compute a reproducible dataset fingerprint.
- [ ] Run:

```powershell
uv run python scripts/build_multi_entity_dataset.py --out dataset/trajectory-v4-multi
uv run python scripts/validate_multi_entity_dataset.py --dataset-root dataset/trajectory-v4-multi
uv run --extra test python -m pytest tests/test_multi_entity_dataset.py -q -p no:cacheprovider --basetemp .pytest-tmp-dataset
```

- [ ] Rebuild in a temporary directory and assert the fingerprint is identical.
- [ ] Commit with `git commit -m "feat: add frozen multi-entity trajectory dataset"`.

### Task 14: Package the protocol and Skills

**Files:**
- Modify: `scripts/train_real_harness.py`
- Modify: `scripts/plot_training_curves.py`
- Create: `skills/director-agent/SKILL.md`
- Modify: `skills/t2blendercodeharness/SKILL.md`
- Modify: `skills/t2blendercodeharness-training/SKILL.md`
- Modify: `skills/scene-contract/SKILL.md`
- Modify: `skills/trajectory-planner/SKILL.md`
- Modify: `skills/blender-proxy-executor/SKILL.md`
- Modify: `skills/harness-evolution/SKILL.md`
- Create: `tests/test_director_skill.py`
- Modify: `tests/test_autodesign_skill_tools.py`

- [ ] Write failing tests requiring five rounds, 10 train plus 10 paired dev per attempt, five attempts maximum, all 60 dev at round end, 30 frozen test cases, 12 workers, canonical model names, exact Memory columns, one-owner rules, and separate curves.
- [ ] Add a `multi-five-rounds` protocol; do not reuse a misleading six-round name.
- [ ] Set defaults to `dataset/trajectory-v4-multi`, `out/training/multi-five-rounds-v1`, `docs/t2blendercodeharness-multi-training-memory-v1.md`, `D:\blender\blender.exe`, 12 workers, and `gpt-5.6-Luna`.
- [ ] Create the Director Skill with schemas, deterministic/provider-assisted modes, evidence/uncertainty policy, repair routing, and stop conditions.
- [ ] Update all component Skills so commands, fingerprints, owners, Memory schema, and scores match implementation.
- [ ] Run focused Skill tests and the full suite; expect zero failures.
- [ ] Commit with `git commit -m "feat: package DirectorAgent multi-entity training workflow"`.

### Task 15: Pass capability checks and a real multi-entity smoke

**Files:**
- Generate: `out/skill_capability_report_director_v1.json`
- Generate: `out/smoke/director-multi-v1/`
- Create: `docs/director-multi-smoke-report.md`

- [ ] Run the full test suite, capability check, and dataset validator:

```powershell
uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-final
uv run python skills/t2blendercodeharness/scripts/capability_check.py --project-root .
uv run python scripts/validate_multi_entity_dataset.py --dataset-root dataset/trajectory-v4-multi
```

- [ ] Prepare `multi-01-01` as an immutable real job under `out/smoke/director-multi-v1`.
- [ ] Render it using `D:\blender\blender.exe`, one worker, 1800-second timeout, and at most two retries.
- [ ] Evaluate real artifacts and inspect at least one handoff frame.
- [ ] Require `proxy.blend`, playable MP4, all entity telemetry, current-owner evidence, active camera, deterministic pass, geometry/frame reports, and three readable event-aligned frames.
- [ ] Record external VLM failure as `unavailable`; never synthesize a score.
- [ ] Write the smoke report and commit with `git commit -m "test: verify real multi-entity Blender smoke"`.

### Task 16: Execute five real training rounds

**Files:**
- Generate: `out/training/multi-five-rounds-v1/`
- Generate/update: `docs/t2blendercodeharness-multi-training-memory-v1.md`
- Generate: `docs/t2blendercodeharness-multi-training-report-v1.md`
- Generate: `docs/figures/multi-training-curves-v1.png`

- [ ] Freeze dataset, evaluator, Director provider, Blender, render-setting, and Harness fingerprints with protocol mode.
- [ ] For each round 1–5, run attempt 1 with 10 unique train, 10 paired dev, real Blender, 12 workers, and `gpt-5.6-Luna`.
- [ ] Immediately verify the Markdown received 20 rows or explicit `NOT_RENDERED` rows.
- [ ] Aggregate only repeated train failures affecting at least two distinct cases.
- [ ] Apply at most one owner patch using `superpowers:test-driven-development`; do not edit dataset labels or evaluator policy to gain score.
- [ ] Run attempts 2–5 only while new actionable evidence exists. Stop on accepted, rejected/rollback without new evidence, `no_patch`, or attempt five.
- [ ] At every round end, evaluate all train cases seen so far and all 60 dev cases. Reuse only fingerprint-identical artifacts.
- [ ] After round five, run the frozen 30-case test exactly once in a mode that cannot create proposals.
- [ ] Plot separate Director plan, task, and realism curves plus artifact completion, hard-failure rate, train-dev gap, and findings by owner.
- [ ] Re-run the full suite and capability check before any completion claim.
- [ ] Commit source-controlled reports with `git commit -m "docs: report five-round multi-entity Harness training"`.

## Completion criteria

- All tests pass from declared dependencies and capability check passes.
- The baseline is fingerprint-indexed without relabeling unavailable scores.
- DirectorAgent is the only orchestration planning entry point.
- No multi-entity path relies on literal single-entity IDs.
- The 50/60/30 dataset is reproducible and leakage-free.
- A real two-character/two-object smoke passes artifact and deterministic gates.
- Skills match actual commands, contracts, model names, Memory fields, and owners.
- Five rounds use real videos and update Memory immediately.
- VLM-unavailable is never converted to zero or a synthetic score.
- The frozen test runs once and never influences patches.
- Final curves keep Director plan, task, and realism channels separate.
