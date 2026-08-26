# Skill Self-Evolution Policy

Self-evolution means improving the skill’s instructions from repeated, reviewable evidence. It does not mean allowing the skill to rewrite itself during a run.

## Input record

Use JSONL records with at least:

```json
{"case_id":"case-07","status":"fail","findings":[{"failure_id":"camera_event_uncovered","owner":"camera_planner","category":"camera_coverage","severity":"hard","message":"...","evidence":["deterministic_report.json"]}]}
```

Mark VLM failures as `status: unavailable` or `source: vlm`; the proposal tool ignores those records for numeric evolution. Keep deterministic failures and VLM availability failures separate.

## Proposal rules

- Require the same normalized failure in at least two distinct cases by default.
- Emit separate proposals for separate owners; never combine owners to reduce work.
- Include target skill section, failure IDs, affected cases, evidence paths, and required regression commands.
- Set `requires_human_review: true` and include the current `SKILL.md` hash.
- Write only a proposal JSON. Never modify `SKILL.md`, project source, evaluator code, or labels.

## Apply a proposal

After human approval, edit the smallest relevant section, rerun the capability check, run the project tests, and forward-test the original pressure scenario plus a variation. Reject the update if it weakens artifact, VLM, one-owner, dev, or test-split safeguards.
