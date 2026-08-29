---
name: harness-evolution
description: Aggregate cross-task failures and accept one-owner Harness patches through train/dev gates.
---

# Harness Evolution Skill

Aggregate normalized findings by failure ID, owner, category, severity, root cause, affected cases, and evidence. A repeated failure must affect at least two distinct train cases. Generate a patch brief for one owner only; mixed-owner proposals are rejected. The acceptance gate records every check and requires strict paired-train improvement, paired and overall dev non-regression, zero hard regressions, and non-regressing artifact completion. Renderer/realism patches additionally require realism improvement with task non-regression. The independent `director_plan_score` is never folded into deterministic, task, or realism scores. Evaluator and generator changes are recorded separately and the candidate is rolled back when the gate rejects it.

## Falsifiable patch contract

Every proposal must include `predicted_fixes`, `predicted_regressions`, and a
`prediction_rationale`. `predicted_fixes` must reference repeated train case
IDs; `predicted_regressions` may be empty only with an explicit rationale.
Attribution runs before root-cause distillation and writes an append-only
verdict. `confirmed` requires every predicted fix to improve with no break;
an unpredicted negative delta is `refuted`, sets `rollback_required`, and rolls
back only the affected files; incomplete evidence is `partial`.

`frame_statistics` and `artifact_health` are artifact diagnostics, never
semantic review and never patch-gate evidence. Lowercase VLM IDs are
`gpt-5.6-luna` and `gpt-5.6-terra`; unavailable is not zero. Prefer changes to
the executable `function_library`, `owner_mapping`, or `memory_entry`. A
`prose_guidance`-only edit is flagged and is not counted as Harness training
without an independently demonstrated runtime effect.
