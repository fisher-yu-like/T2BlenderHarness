# Active Component Map

| Component | Input | Output | Hard boundary |
|---|---|---|---|
| `DirectorAgent` | exact prompt, case obligations, duration, FPS | evidence-backed `DirectorPlan`, trajectories, camera cues, compatibility projections | unresolved hard uncertainty, unknown IDs, broken order, or missing coverage stops the case |
| `blender/lib` verified library | typed primitive parameters | geometry, rigging, constraints, camera, layout, and runtime scaffolding calls | signatures and unit contracts are the only L2 capabilities exposed to codegen |
| `BlenderCodeAgent` | DirectorPlan, library signatures, constraints, Harness version | one case-specific `blender_job.py` | schema/AST/runtime/case-coverage failure is fail-closed; no template fallback |
| external structured provider | Director interpretation request via `OPENAI_BASE_URL` | prompt interpretation with schema and call evidence | endpoint/network/schema errors stay hard uncertainty |
| `CodexExecProvider` | local Blender-code request | per-case source response with schema and call evidence | CLI/JSON/policy errors stay hard uncertainty |
| `CodeCache` | plan hash + Harness version | immutable source and code hash | cache hit reuses exact source; regeneration is explicit |
| Blender CLI renderer | frozen source, isolated run directory | `.blend`, animation frames, telemetry, render response | real `D:\blender\blender.exe`; max 12 workers; retries reuse source at most twice |
| `RealArtifactGate` | run directory and manifest | artifact status, per-file hashes, aggregate artifact hash | missing/unreadable/stale artifact blocks visual scoring |
| deterministic evaluator | contract, plan, telemetry, artifact report | hard gate and diagnostic findings | not the main visual quality score |
| independent oracle | authored case expectations, generated contract/plan/telemetry | independent mismatch and negative-constraint findings | missing evidence is a finding; no silent pass |
| shared visual review | exact prompt, chronological sampled frames, compact plan summary | separate task and realism dimensions | valid only for real VLM or auditable human/local review |
| `MetaHarnessOptimizer` | repeated train findings | one-owner proposal, prediction, acceptance/rollback record | frozen evaluator/dataset/test; one Harness owner per candidate |
| training Memory | every split/round report | Markdown table + JSONL + curves | includes exact prompt, absolute video or `NOT_RENDERED`, scores, fix, delta, and handling |

Legacy `SceneContract`/`TrajectoryPlan` builders and
`template_baseline` remain explicit compatibility/ablation paths only. They do
not sit behind a failed DirectorAgent or BlenderCodeAgent call.
