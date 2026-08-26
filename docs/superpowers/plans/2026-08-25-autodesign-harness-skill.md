# AutoDesign Harness Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a reusable `autodesign-harness` skill that orchestrates the existing Text-to-Blender Harness, checks its capabilities, and proposes safe skill self-evolution updates.

**Architecture:** Add one umbrella skill under `skills/autodesign-harness/` with concise instructions, references for component and real-run contracts, and two deterministic scripts. Keep the existing Harness implementation and component skills as the source of truth; use the skill only as the reusable operating procedure and evaluation guardrail.

**Tech Stack:** Markdown/YAML skill metadata, Python 3.11+, JSON/JSONL, existing Pydantic contracts, pytest, `quick_validate.py`.

---

### Task 1: Create the initialized skill skeleton

**Files:**
- Create: `skills/autodesign-harness/` via `skill-creator/scripts/init_skill.py`
- Create: `skills/autodesign-harness/agents/openai.yaml`

- [ ] Run `init_skill.py autodesign-harness --path skills --resources scripts,references --interface display_name="AutoDesign Harness" --interface short_description="Run and evolve a contract-first Blender proxy Harness" --interface default_prompt="Use $autodesign-harness to evaluate this text-to-Blender scene through the safe proxy pipeline."`.
- [ ] Confirm the generated folder contains only required skill files and resource directories.
- [ ] Run `quick_validate.py skills/autodesign-harness` and keep the expected template failure as the RED checkpoint before replacing placeholders.

### Task 2: Write the umbrella workflow and component references

**Files:**
- Modify: `skills/autodesign-harness/SKILL.md`
- Create: `skills/autodesign-harness/references/component-map.md`
- Create: `skills/autodesign-harness/references/real-pipeline.md`
- Create: `skills/autodesign-harness/references/self-evolution.md`

- [ ] Replace template text with frontmatter whose description starts with `Use when...` and contains text-to-Blender, Blender MCP, proxy video, evaluator, train/dev, Harness patch, and VLM-unavailable triggers.
- [ ] Describe the staged workflow, stop conditions, component handoffs, and current-project command discovery without copying source code.
- [ ] Document the complete artifact gate, deterministic-before-VLM rule, one-owner patch rule, strict acceptance gate, frozen test rule, and proposal-only self-evolution rule.
- [ ] Run the skill validator and a placeholder/line-count scan; keep the body below 500 lines.

### Task 3: Add deterministic capability checking

**Files:**
- Create: `skills/autodesign-harness/scripts/capability_check.py`
- Create: `tests/test_autodesign_skill_tools.py`

- [ ] Write tests first for project-root discovery, contract/trajectory smoke checks, missing-component failure, and JSON report shape.
- [ ] Implement `capability_check.py --project-root <path> --out <path>` to import the project contracts, build a probe plan, verify required component files, exercise one-owner proposal and no-action behavior, and emit machine-readable evidence without Blender/network calls.
- [ ] Run the focused tests and execute the script against the current project.

### Task 4: Add safe skill self-evolution proposals

**Files:**
- Create: `skills/autodesign-harness/scripts/propose_skill_update.py`
- Modify: `tests/test_autodesign_skill_tools.py`

- [ ] Write tests for repeated failures producing one proposal, mixed owners being rejected or split, VLM unavailable records being ignored, and no source/skill mutation.
- [ ] Implement JSONL ingestion, failure grouping, minimum repeat threshold, target-section selection, proposal output, and `requires_human_review=true`.
- [ ] Ensure the script never edits `SKILL.md`, source code, evaluator code, or dataset labels.
- [ ] Run the focused tests on synthetic records and inspect the proposal JSON.

### Task 5: Forward-test and deploy the skill

**Files:**
- Modify: `docs/real-run-protocol.md` only if command names or safety wording need alignment.
- Create: `docs/autodesign-harness-skill-report.md`

- [ ] Run three generic forward scenarios with the skill: incomplete artifacts, multi-owner train failures, and VLM 403.
- [ ] Compare behavior against the no-skill baseline and record pass/fail evidence without leaking expected answers into the prompts.
- [ ] Run `quick_validate.py`, capability check, self-evolution proposal check, `pytest -q`, and `compileall`.
- [ ] Record that project-local packaging is complete; install to a global Codex skill directory only if the user grants filesystem approval for that external write.
