---
name: harness-evolution
description: Aggregate cross-task failures and accept one-owner Harness patches through train/dev gates.
---

# Harness Evolution Skill

Aggregate normalized findings by failure ID, owner, category, severity, affected cases, and evidence. Generate a patch brief for one owner only. The acceptance gate requires strict train improvement, no dev score regression, and no hard dev regression. Evaluator and generator changes are recorded separately and the candidate is rolled back when the gate rejects it.
