# Harness-RSI VLM 驱动训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Enable an AI-only VLM-driven six-round Harness-RSI run in which real video scores drive train-only Harness patch proposals, dev non-regression, and start/end-only frozen testing.

**Architecture:** Keep the existing real Blender pipeline and `CodexVisualReviewProvider`, but separate `ai_only_vlm_rsi` from `diagnostic_only` and formal human-calibrated release training. The new mode passes a real visual provider into train/dev/test evaluation, converts only evidence-backed VLM failures into train evidence, and connects the existing outer controller/patch executor boundary without permitting frozen component edits.

**Tech Stack:** Python 3.12, pytest/uv, Blender 5.1.2, Pydantic contracts, local `codex exec` visual provider, existing `OuterTransitionController` and `PatchExecutor`.

**Spec:** `docs/superpowers/specs/2026-09-05-harness-rsi-vlm-rsi-design.md`

## Global Constraints

- VLM provider is required for `ai_only_vlm_rsi`; no assistant-local proxy fallback is accepted as a visual score.
- `dataset/vbench2-agent-training-index-v1` is train/dev input; `dataset/vbench2-agent-test-100-v1` is frozen test input.
- Test schedule is `baseline_final_only`: round 0 and round 6 only; test never enters patch selection.
- Patch target paths are limited to Harness code under `src/videoact`; dataset, evaluator, observer, tests, and test policy are frozen.
- `formal_training_allowed=false` remains explicit because human golden review and paired gates are not complete.
- No template-backed or fallback-generated output may enter scoring or patch selection.
- Every production behavior change must have a test written and observed failing before implementation.

### Task 1: Add failing contracts for VLM-RSI mode and score-to-patch boundary

**Files:**
- Modify: `tests/test_skill_contracts.py`
- Modify: `tests/test_six_round_resume.py`
- Create: `tests/test_vlm_rsi_protocol.py`

**Interfaces:**
- Tests will require `build_vlm_rsi_protocol(...)` or the equivalent public training entry to emit `execution_mode="ai_only_vlm_rsi"`, `visual_scores_permitted=true`, `vlm_required=true`, and only round 0/6 scheduled tests.
- Tests will require the new mode to pass a non-`None` visual provider to evaluation and to reject a missing provider before creating a training report.
- Tests will require train records containing VLM evidence to be accepted by the patch boundary while test records remain excluded.

- [ ] Write the failing tests for the mode manifest, provider requirement, schedule, and train-only patch input.
- [ ] Run `uv run python scripts/run_unit_tests.py -q tests/test_vlm_rsi_protocol.py tests/test_skill_contracts.py tests/test_six_round_resume.py --basetemp .pytest-vlm-rsi-red`.
- [ ] Confirm failure is caused by the missing VLM-RSI mode/contract, not by test collection errors.

### Task 2: Implement the explicit AI-only VLM-RSI protocol mode

**Files:**
- Modify: `scripts/train_real_harness.py`
- Modify: `evaluator/codex_visual.py` only if provider metadata or fail-closed behavior needs a contract correction
- Test: `tests/test_vlm_rsi_protocol.py`

**Interfaces:**
- Add an explicit `ai_only_vlm_rsi`/`vlm-rsi-six-rounds` entry point separate from `diagnostic_only`.
- The entry point accepts `visual_provider`, `vlm_model`, `provider_mode`, `test_schedule`, and `outer_transition` and writes a protocol manifest with VLM enabled.
- A missing provider raises a deterministic configuration error before case generation.

- [ ] Add the smallest mode flag/API that distinguishes `ai_only_vlm_rsi` from `diagnostic_only`.
- [ ] Run the red tests and confirm the manifest/provider assertions turn green.
- [ ] Preserve `formal_training_allowed=false` while setting `visual_scores_permitted=true` and `vlm_required=true`.

### Task 3: Connect VLM train evidence to the outer controller and patch executor

**Files:**
- Modify: `scripts/train_real_harness.py`
- Modify: `src/videoact/outer_controller.py` only where VLM evidence normalization is missing
- Modify: `src/videoact/patch_executor.py` only where the existing execution boundary cannot be called from the training entry point
- Create: `tests/test_vlm_rsi_patch_boundary.py`

**Interfaces:**
- Add a train-evidence adapter that accepts only train rows with `review_source=codex_local_visual_review`, finite VLM scores, confidence/evidence references, and an owner mapping.
- The adapter returns an `OuterTransitionController` input with no dev/test IDs or metrics.
- Patch application returns a manifest containing `status`, `owner`, `changed_files`, `diff_sha256`, parent hashes, and rollback result.

- [ ] Write a failing test proving a VLM-backed train failure reaches the controller and test evidence is rejected.
- [ ] Write a failing test proving a patch outside `src/videoact` is rejected.
- [ ] Run those tests and verify the expected failures.
- [ ] Implement the evidence adapter and wire the controller/executor into the VLM-RSI round transition.
- [ ] Run the tests again and verify they pass without weakening frozen-path checks.

### Task 4: Make round acceptance depend on VLM and dev non-regression

**Files:**
- Modify: `scripts/train_real_harness.py`
- Modify: `evaluator/realism.py` only if score-kind aggregation incorrectly treats proxy-only evidence as VLM evidence
- Create: `tests/test_vlm_rsi_acceptance.py`

**Interfaces:**
- A round transition must return `no_patch` when VLM is unavailable, `rejected`/`rollback` when dev regresses, and `patch`/`accepted` only after train evidence, patch scope, and dev gate pass.
- The decision must record `vlm_scored_count`, `task_final_score`, `visual_score`, `realism_score_kind`, and the dev comparison.

- [ ] Write failing tests for VLM-unavailable fail-closed, dev regression rejection, and accepted evidence-backed patch.
- [ ] Run the red tests.
- [ ] Implement the minimal acceptance/rollback logic.
- [ ] Run the green tests and inspect the generated transition manifest.

### Task 5: Add the new training runner and detailed user-facing documentation

**Files:**
- Create: `scripts/run_vlm_rsi_six_rounds.py`
- Create: `docs/harness-rsi-vlm-rsi-training-2026-09-05.md`
- Modify: `README.md` only to document the stable command and safety boundary
- Test: `tests/test_vlm_rsi_protocol.py`

**Interfaces:**
- Runner defaults to `D:\harness-rsi-training\vlm-rsi-six-rounds-20260905`, `gpt-5.6-luna`, `provider_mode=model`, two workers, and `baseline_final_only`.
- Runner writes a start manifest before launching Blender and exits nonzero when the provider cannot be constructed.

- [ ] Write the user-facing training record with exact command, data, VLM contract, round schedule, patch scope, output files, and known formal-gate limitation.
- [ ] Add the runner with no template/fallback mode.
- [ ] Run `uv run python scripts/run_vlm_rsi_six_rounds.py --help` and its dry protocol test.

### Task 6: Regression verification and release preparation

**Files:**
- No production files unless a test exposes a defect.
- Test artifacts: `out/preflight/` only if ignored/generated.

- [ ] Run targeted VLM-RSI tests with a clean basetemp.
- [ ] Run the full unit suite with a fresh basetemp and record the exact counts.
- [ ] Run `git diff --check` and inspect staged paths, excluding generated outputs and pytest temp directories.
- [ ] Review the implementation against the spec's nine acceptance criteria.

### Task 7: Publish to GitHub main and launch the new run

**Files:**
- Commit all intended source, tests, and documentation changes in the `harness-rsi` worktree.

- [ ] Fetch `origin/main` and record its SHA before overwrite.
- [ ] Create a commit whose message identifies VLM-RSI enablement and the training protocol document.
- [ ] Verify commit contents and test evidence before push.
- [ ] Push with `git push --force-with-lease origin HEAD:main`.
- [ ] Verify `git ls-remote origin refs/heads/main` equals the pushed commit.
- [ ] Launch `scripts/run_vlm_rsi_six_rounds.py` in the new output root and record PID/log paths.

