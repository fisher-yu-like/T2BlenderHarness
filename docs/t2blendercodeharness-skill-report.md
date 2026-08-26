# T2Blendercodeharness Skill Report

## Package

The reusable package is `skills/t2blendercodeharness/`. It contains the umbrella `SKILL.md`, UI metadata, component/real-pipeline/self-evolution references, and two network-free scripts:

- `scripts/capability_check.py` validates the project boundary, contract/planner imports, evaluator interfaces, clean no-action behavior, and one-owner proposal behavior.
- `scripts/propose_skill_update.py` groups repeated deterministic failures into reviewed one-owner proposals and ignores VLM-unavailable records.

The existing `scene-contract`, `trajectory-planner`, `blender-proxy-executor`, and `harness-evolution` skills remain the component-level pieces in `skills/`.

## RED/GREEN forward-test evidence

Three no-skill pressure scenarios were run first. They established the failure modes the skill must prevent: telemetry-only promotion, multi-owner patching, and turning VLM 403 into a numeric training signal.

The same three scenarios were run with `t2blendercodeharness`:

- incomplete sampled artifacts: blocked at artifact gate;
- scene parser + camera planner + executor failures: split into owner-scoped proposals and reject one multi-owner patch;
- VLM 403: record `unavailable`, continue deterministic-only evidence, and do not train on a fabricated score.

## Local verification

- `quick_validate.py skills/t2blendercodeharness`: passed.
- `capability_check.py --project-root .`: passed all 6 checks.
- `propose_skill_update.py` synthetic repeated-failure run: emitted one camera-owner proposal, ignored one VLM-unavailable record, and reported `requires_human_review=true`.
- Skill tool tests: 4 passed.

Self-evolution is intentionally proposal-only. Applying a proposal requires human review, the capability check, the full project test suite, and forward-testing of the original scenario plus a variation.
