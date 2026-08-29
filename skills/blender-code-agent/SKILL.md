---
name: blender-code-agent
description: Use when generating case-specific Blender job source from a DirectorPlan, composing verified blender.lib primitives, validating runtime artifacts, or diagnosing fail-closed code-generation failures.
---

# BlenderCodeAgent

`BlenderCodeAgent` is the code-generation boundary of
`T2Blendercodeharness`. It receives one frozen `DirectorPlan` per case and
returns a typed `CodegenResponse`; it does not reinterpret the prompt or
silently choose a scene template.

## Layers

- **L2 verified library:** `blender/lib/geometry.py`, `rigging.py`,
  `constraints.py`, `camera.py`, `layout.py`, and `scaffolding.py`. Public
  functions are registered and exported through `signatures.json`.
- **L3 composition:** the provider reads the exact DirectorPlan and library
  signatures, then writes one case-specific `blender_job.py` using the listed
  primitives.
- **L4 explicit escape hatch:** only a response with
  `status=library_insufficient` may request new geometry. It remains an
  agent-generated source path and must pass the stricter artifact, geometry,
  penetration, and visual-review gates before it can enter evidence.

## Required handoff

The generated source must preserve the DirectorPlan's entity IDs, ordered event
IDs, trajectory and attachment semantics, camera targets/cues, and runtime
traceability. It must import `bpy` and verified `blender.lib` code, bind
`DIRECTOR_PLAN`, write `telemetry.json` and `frames/index.json`, save
`proxy.blend`, and render animation frames. Static validation rejects syntax
errors, forbidden imports/calls, unknown library calls, missing runtime
markers, and references to `compile_real_proxy_job`, `direct_prompt_code`, or
other legacy template compilers.

## generate-once-freeze, fail-closed, and retry rules

- Missing provider, provider/schema error, invalid source, or coverage failure
  produces `hard_uncertainty`/`codegen_failed`; no `blender_job.py` is created
  for the agent case and no score is entered.
- `template_baseline` is an explicit historical comparison arm only. It is
  never a fallback from L3/L4 and never runs during production agent training.
- Code generation is cached by `plan_hash + harness_version`; a cache hit is
  reused without another provider call. A new library or prompt policy
  requires an explicit Harness version bump or `--force-regen`.
- Blender process failure may retry the same frozen source at most two times.
  Rendering never regenerates code, changes the source, or switches to a
  template. A source mutation during retry is a hard failure.

## Evidence and ownership

Record `plan_hash`, `director_plan_hash`, `code_hash`, provider/model,
`llm_call_id`, cache state, source path, and failure reason in the run
manifest and code cache. Route repeated failures to one owner:

| Evidence | Owner |
|---|---|
| L2 primitive contract or implementation | `blender_code_agent` / library |
| L3 prompt, schema, or composition | `blender_code_agent` |
| L4 gate or new-primitive promotion | `blender_code_agent` |
| Blender process, telemetry, or stale artifact | `blender_executor` |

Only a repeated train finding across at least two distinct cases may propose a
Harness patch. Dev is a paired holdout; frozen test is milestone-only.

## Reviewed context boundary

Few-shot context is optional, but it is never implicit. Before an example can
enter `CodegenRequest.context_examples`, run
`scripts/validate_codegen_examples.py`. The validator requires a real playable
MP4, readable sampled frames, a complete artifact gate, `director_plan.json`,
the exact plan hash, the exact source hash, deterministic evidence, and an
eligible visual review provenance (`human_review`, `codex_local_visual_review`,
`gpt-5.6-luna`, or `gpt-5.6-terra`). An artifact-only or template example is
rejected. A present-but-invalid context manifest makes the case
`codegen_failed` before the Blender code provider is called; a missing
manifest is recorded as `context_status=none`. The cache manifest records the
context status and example IDs, while the frozen source remains keyed only by
`plan_hash + harness_version` and is never silently replaced.

L4 primitive promotion is report-only through
`scripts/promote_fallback_primitives.py`. Three distinct real reviewed cases
may produce a candidate, but the library is not edited until human approval,
tests, and a new Harness version exist.

## Verification

Use the project capability check and focused tests before any real batch:

```powershell
uv run python scripts/export_library_signatures.py --output blender/lib/signatures.json
uv run --extra test python -m pytest -q -p no:cacheprovider --basetemp .pytest-tmp-codegen tests/test_blender_code_agent.py tests/test_case_coverage_gate.py tests/test_agent_real_job_generation.py
uv run python skills/t2blendercodeharness/scripts/capability_check.py --project-root .
```

Do not call a VLM or report a video score for a failed code-generation or
artifact gate. An unavailable visual provider is recorded as `unavailable`,
not as zero or a plan-derived score.
