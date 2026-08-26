---
name: t2blendercodeharness
description: Use when a user asks to turn text into Blender proxy scenes or videos, run scene/trajectory/camera planning, execute Blender MCP or CLI, evaluate artifacts or VLM frames, repair failures, evolve a Harness from train/dev evidence, or package and validate the T2Blendercodeharness workflow as a reusable skill.
---

# T2Blendercodeharness

Apply this skill to operate a contract-first Text-to-Blender proxy Harness. Preserve the separation between Codex Host, MetaHarnessOptimizer, DesignHarness components, Dataset, and Evaluator.

**REQUIRED TRAINING SUB-SKILL:** When the request includes Harness training, fixed train/dev rounds, full-train scoring, real Blender video evaluation, VLM scoring, or Memory, use `t2blendercodeharness-training` and its `scripts/train_real_harness.py` entry point.
**REQUIRED DIRECTOR SUB-SKILL:** For prompts with implicit intent, multiple actors/props, handoffs, concurrency, or camera choreography, use `director-agent` first. DirectorAgent is the only external planning entry point; legacy SceneContract/TrajectoryPlan objects are compatibility projections.

**Core principle:** evidence gates every promotion. A plan is not a render, telemetry is not a complete artifact, a deterministic pass is not a VLM score, and a proposal is not an accepted Harness patch.

## Component handoff

Use the existing component skills when installed:

- **REQUIRED SUB-SKILL:** Use `director-agent` for evidence-backed prompt interpretation, event scheduling, multi-entity trajectories, and multi-target camera choreography.
- **REQUIRED SUB-SKILL:** Use `scene-contract` for validating the Director projection and legacy single-entity compatibility.
- **REQUIRED SUB-SKILL:** Use `trajectory-planner` for validating projected entity states, camera shots, and event observability.
- **REQUIRED SUB-SKILL:** Use `blender-proxy-executor` for controlled MCP/CLI execution and run manifests.
- **REQUIRED SUB-SKILL:** Use `harness-evolution` for failure aggregation and train/dev acceptance.

If a component skill is unavailable, use the project modules with the same responsibility. Never bypass a missing contract or evaluator by writing ad-hoc Blender code.

## Run the pipeline

1. Discover the project root, Python runtime, dataset split, available Blender backend, and existing run state. Prefer the project's Python 3.11+ runtime; do not assume the shell's `python` is compatible.
2. Run `DirectorAgent.plan` on the exact prompt. Reject empty prompts, unknown entity references, unsupported assumptions, unresolved hard uncertainty, invalid timing, broken relations, and contradictory event order.
3. Validate the projected `SceneContract`, multi-entity `TrajectoryPlan`, interaction lifecycle, one-based frame bounds, state continuity, target visibility, shot coverage, and event observability before execution.
4. Execute through the controlled adapter or Blender MCP. Persist prompt/plan/Harness/evaluator fingerprints, MCP response, state transitions, and immutable artifacts.
5. Apply the real artifact gate. Require manifest, contract, trajectory, camera plan, job source, `.blend`, host-assembled `.mp4`, telemetry, index, and at least three readable sampled PNGs.
6. Run deterministic evaluation first. Hard failures block VLM, training records, and patch selection. Inspect a sampled frame when a semantic or visibility failure is plausible.
7. Run VLM evaluation only for artifact-complete deterministic-pass runs and only through a compliant endpoint. Record `unavailable` for network, policy, or schema failures; never convert it to zero or a synthetic preference.
8. Repair locally with bounded attempts. Route each finding to one component, keep the original contract immutable, and stop after the project's configured attempt limit.
9. For the outer loop, aggregate repeated train failures by failure ID, owner, category, severity, and root cause. A proposal requires the same failure to affect two distinct train cases and exactly one owner.
10. Re-run paired train/dev with the same dataset, evaluator, backend, and fingerprints. Accept only strict paired train improvement, paired and overall dev non-regression, zero hard regressions, and non-regressing artifact completion. For realism/renderer patches, realism must improve and task score must not fall. Keep test split frozen and use it only for final blind verification.

## Project commands

When the repository contains the current implementation, prefer these entry points:

```text
scripts/prepare_real_jobs.py       # immutable jobs
scripts/run_real_pipeline.py       # dry-run/evaluate host stages
scripts/evaluate_real_runs.py      # artifact gate + deterministic reports
scripts/evaluate_real_videos.py    # eligible sampled-frame VLM stage
scripts/run_real_outer_loop.py     # train/dev failure aggregation
scripts/build_multi_entity_dataset.py      # reproducible trajectory-v4-multi builder
scripts/validate_multi_entity_dataset.py   # 50/60/30 leakage and contract validator
```

Run `python skills/t2blendercodeharness/scripts/capability_check.py --project-root .` before claiming the skill works in a new project. Read `references/real-pipeline.md` for state and artifact details.

## Skill self-evolution

## Realism evaluator boundary

For realism probes, use the project training sub-skill and `evaluator/realism.py` v4. The Blender geometry audit is an eligibility gate, not a realism oracle: it may reach 100 only for structural compliance. One shared visual-review call (external `gpt-5.6-luna`/`gpt-5.6-terra` or local Codex `assistant_local_review`) returns separate task and realism dimensions; the scores are never added. Realism uses `.15` geometry, `.15` rendered-frame evidence, and `.70` independent visual review. If no review is available, retain only the capped `artifact_only_proxy` evidence score and mark it `not_established`; never copy it into a VLM score.

Use `python skills/t2blendercodeharness/scripts/propose_skill_update.py --records <records.jsonl> --out <proposal.json>` after a real evaluation batch. The script may group repeated failures and propose a single-owner section update, but it must not edit `SKILL.md`, source code, evaluator code, or dataset labels. Require human review, capability checks, project tests, and a forward-test before applying a proposal. Read `references/self-evolution.md`.

## Stop conditions

Stop and report a blocked stage when any of these occurs:

- contract or trajectory validation fails;
- MCP/CLI execution is failed or still running;
- required artifacts are missing, stale, or unreadable;
- deterministic hard gate fails;
- VLM endpoint is unavailable or non-compliant;
- train evidence has no repeated actionable failure;
- a proposal has more than one owner;
- no repeated failure affects two distinct train cases;
- dev regresses or test data would influence patch selection.

Do not claim "trained", "accepted", or "video-evaluated" unless the corresponding report and evidence artifact exists.

## Common mistakes

| Temptation | Required correction |
|---|---|
| Telemetry exists, so evaluate | Wait for the full artifact gate and readable samples. |
| VLM returned 403, so use score 0 | Record `unavailable`; exclude it from numeric training data. |
| Several owners failed, so patch all files | Split proposals by owner and rerun the acceptance gate. |
| Train passed, so accept the patch | Require strict train improvement plus non-regressing dev. |
| Test reveals the fix | Keep test frozen until final blind verification. |
| Skill can edit itself immediately | Emit a reviewed proposal; never self-apply. |
