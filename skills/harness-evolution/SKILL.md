---
name: harness-evolution
description: Aggregate cross-task failures and accept one-owner Harness patches through train/dev gates.
---

# Harness Evolution Skill

Aggregate normalized findings by failure ID, owner, category, severity, root cause, affected cases, and evidence. A repeated failure must affect at least two distinct train cases. Generate a patch brief for one owner only; mixed-owner proposals are rejected. The acceptance gate records every check and requires strict paired-train improvement, paired and overall dev non-regression, zero hard regressions, and non-regressing artifact completion. Renderer/realism patches additionally require realism improvement with task non-regression. The independent `director_plan_score` is never folded into deterministic, task, or realism scores. Evaluator and generator changes are recorded separately and the candidate is rolled back when the gate rejects it.
