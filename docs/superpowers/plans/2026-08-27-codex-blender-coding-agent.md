# Codex BlenderCodingAgent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace local fixed Blender scene templates with a real per-case `codex exec` BlenderCodingAgent and rebuild the reusable Harness path.

**Architecture:** `prepare_real_jobs.py` continues to obtain the exact DirectorPlan, but delegates Blender source creation to `BlenderCodingAgent`. The agent writes only case-local context and generated code, validates the code, retries Codex failures, and never falls back to a local Blender template. Blender CLI remains the real execution backend.

**Tech Stack:** Python 3.11+, `codex exec`, existing Pydantic Director contracts, Blender 5.1.2, existing CLI renderer, pytest, JSON/Markdown artifacts.

---

### Task 1: Freeze deletion scope and add design evidence

**Files:**
- Create: `docs/superpowers/specs/2026-08-27-codex-blender-coding-agent-design.md`
- Create: `docs/superpowers/plans/2026-08-27-codex-blender-coding-agent.md`

- [x] **Step 1: Write the design and plan**

The design fixes the deletion boundary, Director mode/raw mode input contract, Codex write boundary, retry behavior, no-fallback policy, compile gate, and smoke acceptance.

- [x] **Step 2: Verify the plan has no placeholders**

Run:

```powershell
rg -n "TBD|TODO|implement later|fill in" docs/superpowers/specs/2026-08-27-codex-blender-coding-agent-design.md docs/superpowers/plans/2026-08-27-codex-blender-coding-agent.md
```

Expected: no matches.

### Task 2: Define the Codex agent interface with failing tests

**Files:**
- Create: `tests/test_blender_coding_agent.py`
- Create: `src/videoact/blender_coding_agent.py`

- [ ] **Step 1: Write failing tests**

Add tests for these exact behaviors:

```python
def test_request_context_preserves_prompt_and_hashes(tmp_path):
    request = BlenderCodingAgentRequest(
        case_id="case-01",
        output_dir=tmp_path,
        prompt="A red kite circles a lighthouse while a child watches.",
        planning_mode="director",
        director_plan={"director_plan_hash": "dp"},
        trajectory_plan={"plan_hash": "tp"},
        camera_plan={"camera_hash": "cp"},
        proxy_spec={"geometry": {"detail_required": True}},
        manifest={"prompt_hash": "pp"},
    )
    context = request.write_context()
    payload = json.loads(context.read_text(encoding="utf-8"))
    assert payload["prompt"] == request.prompt
    assert payload["planning_mode"] == "director"
    assert payload["hashes"] == {"prompt": "pp", "director_plan": "dp", "trajectory_plan": "tp", "camera_plan": "cp"}


def test_agent_retries_compile_failure_without_local_scene_fallback(tmp_path):
    runner = FakeCodexRunner(["not python", "from pathlib import Path\nPath('blender_job.py').write_text('import bpy\\n')"])
    result = BlenderCodingAgent(runner=runner, max_attempts=2).generate(request)
    assert result.status == "generated"
    assert result.attempt_count == 2
    assert result.used_local_template is False
    assert "bpy" in (tmp_path / "blender_job.py").read_text(encoding="utf-8")


def test_agent_fails_closed_when_codex_never_writes_code(tmp_path):
    runner = FakeCodexRunner(["no code", "still no code"])
    result = BlenderCodingAgent(runner=runner, max_attempts=2).generate(request)
    assert result.status == "failed"
    assert result.reason == "missing_generated_code"
    assert result.used_local_template is False
```

The test module must define a runner fake that only records calls and writes no production scene template. Import the request/result classes from the new module so the tests fail first with an import error.

- [ ] **Step 2: Run the focused tests and observe RED**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run pytest -q tests/test_blender_coding_agent.py
```

Expected: collection fails because `src/videoact/blender_coding_agent.py` does not exist.

### Task 3: Implement the transport-only BlenderCodingAgent

**Files:**
- Modify: `src/videoact/blender_coding_agent.py`
- Test: `tests/test_blender_coding_agent.py`

- [ ] **Step 1: Implement request/result models and context writing**

Implement `BlenderCodingAgentRequest` with fields `case_id`, `output_dir`, `prompt`, `planning_mode`, `director_plan`, `trajectory_plan`, `camera_plan`, `proxy_spec`, and `manifest`. `write_context()` must write only `coding_agent_context.json` under `output_dir`, preserving exact prompt and hash values.

- [ ] **Step 2: Implement Codex command transport**

Implement a default runner that invokes:

```text
codex exec --ephemeral --json --sandbox workspace-write -C <case_dir> -o <case_dir>/codex_agent_last_message.md <strict prompt>
```

The command must not include `--dangerously-bypass-approvals-and-sandbox`. The strict prompt tells Codex to write only `blender_job.py` and `generated_code_manifest.json` in the case directory, to read `coding_agent_context.json`, and to return a summary. Do not include Blender scene code in this module.

- [ ] **Step 3: Implement output validation and bounded retry**

After each call, require `blender_job.py` to exist and compile with `compile(source, ..., "exec")`. On missing code, nonzero Codex return, timeout, or compile failure, write `coding_agent_attempts.json` and retry up to `max_attempts`. Return `status="failed"`, `reason="missing_generated_code"`, `"codex_failed"`, or `"generated_code_compile_error"` as appropriate. Set `used_local_template=False` in every result.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run pytest -q tests/test_blender_coding_agent.py
```

Expected: all focused tests pass.

### Task 4: Replace local compiler in real job preparation

**Files:**
- Modify: `scripts/prepare_real_jobs.py:17-124`
- Modify: `skills/t2blendercodeharness/scripts/capability_check.py`
- Create: `tests/test_prepare_real_jobs_uses_codex_agent.py`

- [ ] **Step 1: Write failing integration-boundary tests**

Test that `prepare_real_jobs.py` imports `BlenderCodingAgent`, does not import `compile_real_proxy_job`, and calls the agent with `planning_mode="director"`, exact prompt, and nonempty Director/Trajectory/Camera payloads. Test that a failed agent result raises a clear `RuntimeError` and never creates a fallback `blender_job.py`.

- [ ] **Step 2: Run the boundary tests and observe RED**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run pytest -q tests/test_prepare_real_jobs_uses_codex_agent.py
```

Expected: FAIL because the existing script still imports and calls `compile_real_proxy_job`.

- [ ] **Step 3: Wire `prepare_real_jobs.py` to Codex**

After DirectorAgent creates the contracts and metadata, instantiate `BlenderCodingAgent` and submit the request. Do not compile a local job. On a generated result, record `coding_agent_context.json`, `coding_agent_attempts.json`, `generated_code_manifest.json`, and the code hash in job index. On failure, persist metadata and stop the case with no fallback code.

- [ ] **Step 4: Add capability check**

Add `blender_coding_agent` to `REQUIRED_COMPONENTS` and construct a request in the capability check using a fake runner. The check must prove the interface is present without invoking a network model or generating a Blender scene.

- [ ] **Step 5: Run integration tests and the full suite**

Run:

```powershell
$env:PYTHONPATH=(Get-Location).Path
uv run pytest -q tests/test_blender_coding_agent.py tests/test_prepare_real_jobs_uses_codex_agent.py
uv run pytest -q
```

Expected: focused and full tests pass.

### Task 5: Update skill/training contracts

**Files:**
- Modify: `skills/t2blendercodeharness/SKILL.md`
- Modify: `skills/t2blendercodeharness-zh/SKILL.md`
- Modify: `skills/t2blendercodeharness-training/SKILL.md`
- Modify: `skills/director-agent/SKILL.md`
- Modify: `docs/t2blendercodeharness-multi-training-memory-v1.md`

- [ ] **Step 1: Add the Codex coding-agent rule**

Document that Blender source must be generated by `codex exec` per case, that local Python is transport/validation only, that no template fallback is allowed, and that prompt/plan/code hashes must be kept in Memory.

- [ ] **Step 2: Add the dynamic-entity rule**

Document that no actor/prop count, name, shape, material, trajectory, or camera motion may be hardcoded as a universal default. The exact prompt and DirectorPlan determine the scene.

- [ ] **Step 3: Add failure and retry rules**

Separate Codex generation retries from Blender render retries. A generation failure cannot enter evaluator scoring or Harness training.

- [ ] **Step 4: Run skill validators and tests**

Run:

```powershell
uv run python skills/t2blendercodeharness/scripts/capability_check.py --project-root . --out out/codex-coding-agent-capability.json
$env:PYTHONPATH=(Get-Location).Path
uv run pytest -q
```

Expected: capability pass and full test suite pass.

### Task 6: Delete invalid fixed-template ablation assets

**Files/directories to delete after tests are green:**
- `blender/direct_prompt_code.py`
- `scripts/prepare_direct_prompt_jobs.py`
- `scripts/evaluate_three_arm_ablation.py`
- `scripts/prepare_three_arm_reviews.py`
- `scripts/aggregate_three_arm_ablation.py`
- `scripts/author_three_arm_local_reviews.py`
- `scripts/build_three_arm_skill_records.py`
- `tests/test_direct_prompt_ablation.py`
- `tests/test_three_arm_blind_review.py`
- `tests/test_three_arm_evaluator.py`
- `tests/test_three_arm_local_review.py`
- `docs/t2blendercodeharness-three-arm-ablation-v1.md`
- `docs/harness-release-bundle-v1.md`
- `docs/superpowers/plans/2026-08-27-three-arm-ablation-and-skill-training.md`
- `out/benchmarks/vbench-100-three-arm-ablation-v1/`
- `out/releases/t2blendercodeharness-release-v1.zip`
- `out/releases/t2blendercodeharness-release-v1/`

- [ ] **Step 1: Verify paths are inside the current worktree**

Run a PowerShell path check for every target and refuse deletion if any resolved path is outside the current worktree.

- [ ] **Step 2: Delete only the invalid fixed-template assets**

Use native PowerShell `Remove-Item -LiteralPath` after the path check. Do not delete the valid current/pretrain benchmark, datasets, evaluator, or training memory.

- [ ] **Step 3: Verify deletion and historical labels**

Run:

```powershell
Test-Path blender/direct_prompt_code.py
Test-Path out/benchmarks/vbench-100-three-arm-ablation-v1
rg -n "local fixed|not a Codex|non-Codex|template ablation" docs skills
```

Expected: deleted paths are false and historical documents do not present the deleted result as a Codex coding-agent benchmark.

### Task 7: Real Codex + Blender smoke

**Files:**
- Create: `out/smoke/codex-blender-agent-v1/` generated artifacts only
- Modify: `docs/superpowers/plans/2026-08-27-codex-blender-coding-agent.md`

- [ ] **Step 1: Check Codex authentication without exposing secrets**

Run `codex doctor` and record only exit status and non-secret summary. If unavailable, stop with a recorded blocker; do not use a local template.

- [ ] **Step 2: Generate one complex Director case with real Codex**

Use a prompt containing at least two actors, two different props, a handoff, a return, parallel motion, and a motivated camera move. Run the new preparation entry point. Verify `coding_agent_context.json`, `coding_agent_attempts.json`, `generated_code_manifest.json`, and `blender_job.py` exist.

- [ ] **Step 3: Run Python compile and Blender CLI**

Run `D:\blender\blender.exe` through the existing real renderer with one isolated worker and at most 2 render retries. Verify `.blend`, MP4, telemetry, sample frames, and artifact gate.

- [ ] **Step 4: Inspect generated scene and audit dynamic entities**

Confirm the case uses the prompt's actual entities and does not silently replace them with Alice/Bob or red cube/blue cup. Run deterministic and geometry audits; preserve any failure as evidence.

- [ ] **Step 5: Update plan checkpoint**

Mark the smoke task complete only when real Codex generated code and Blender produced a playable MP4. If Codex is unavailable, mark the plan blocked at this task and do not claim the Harness rebuild is complete.

### Task 8: Final verification and handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-08-27-codex-blender-coding-agent.md`
- Create: `docs/codex-blender-coding-agent-rebuild-report.md`

- [ ] **Step 1: Run fresh verification**

Run full tests, capability check, compileall, git diff check, and inspect the smoke artifact manifest.

- [ ] **Step 2: Write rebuild report**

Include exact Codex command mode, generated code path, prompt/plan/code hashes, real Blender artifact paths, retry counts, evaluator findings, limitations, and the list of deleted invalid assets.

- [ ] **Step 3: Do not run full training until smoke passes**

The six-round training plan remains pending until this smoke proves that Codex is actually producing per-case Blender code. No prior fixed-template scores may be used as the new baseline.
