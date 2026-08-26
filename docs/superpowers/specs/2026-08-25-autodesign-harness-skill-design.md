# AutoDesign Harness Skill Design

## Goal

Package the contract-first Text-to-Blender proxy Harness as a reusable umbrella skill with explicit component boundaries, executable capability checks, and a safe proposal-only self-evolution loop.

## Scope

The package covers:

- natural-language scene contract construction;
- frame-indexed trajectory and camera planning;
- controlled Blender MCP/CLI proxy execution;
- artifact, telemetry, deterministic, and optional VLM evaluation;
- bounded inner-loop repair;
- train/dev outer-loop failure aggregation and one-owner patch acceptance;
- skill capability checks and evidence-backed skill update proposals.

It does not copy the project implementation into the skill, train model weights, silently upload frames to an unapproved endpoint, or auto-edit its own `SKILL.md`.

## Packaging Architecture

```text
skills/autodesign-harness/
├── SKILL.md                         # concise trigger + orchestration rules
├── agents/openai.yaml               # UI metadata
├── references/
│   ├── component-map.md              # component responsibilities and contracts
│   ├── real-pipeline.md              # artifact/state/evaluator gates
│   └── self-evolution.md             # proposal-only update policy
└── scripts/
    ├── capability_check.py           # deterministic smoke test against a project
    └── propose_skill_update.py       # failure aggregation -> reviewed proposal
```

The existing component skills remain the project-local sub-skills: `scene-contract`, `trajectory-planner`, `blender-proxy-executor`, and `harness-evolution`. The umbrella skill references them by name and falls back to the project modules when the sub-skills are not installed.

## Runtime Workflow

1. Discover the project root, Python runtime, available Blender backend, dataset split, and existing run artifacts.
2. Build and validate `SceneContract`; stop on invalid entities, event ordering, timing, or relations.
3. Build `TrajectoryPlan`; reject discontinuities and uncovered required events before Blender execution.
4. Execute only through the controlled adapter/MCP boundary and persist a run manifest/state record.
5. Require the complete artifact contract before evaluator or VLM promotion.
6. Run deterministic checks first; send only eligible sampled frames to a compliant VLM endpoint.
7. Run bounded inner repair without changing the contract silently.
8. Aggregate train failures by normalized owner and require exactly one Harness owner per proposal.
9. Re-run train and dev with stable fingerprints; accept only strict train improvement and no dev regression. Keep test frozen until final blind verification.

## Safety and Self-Evolution

The skill treats `unavailable` VLM results as missing evidence, never as zero scores. Self-evolution consumes JSONL evaluation records and emits a proposal with repeated failure evidence, affected section, and required regression checks. Applying a proposal requires human approval plus capability-check and project-test results; the proposal script never edits `SKILL.md` automatically.

## Validation

- Run `skill-creator/scripts/quick_validate.py` on the skill directory.
- Run `scripts/capability_check.py --project-root .` and inspect its JSON report.
- Run `scripts/propose_skill_update.py` on synthetic repeated failures and verify it emits a proposal without modifying the skill.
- Run the project test suite and compile the bundled scripts.
- Forward-test the installed skill on artifact-gate, one-owner, and VLM-unavailable scenarios.
