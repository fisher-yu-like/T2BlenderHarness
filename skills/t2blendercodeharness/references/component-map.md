# Component Map

Use this map to preserve interfaces while changing implementation details.

| Component | Input | Output | Hard boundary |
|---|---|---|---|
| Scene/Prompt Parser | natural-language prompt, duration, FPS | validated `SceneContract` | no Blender execution; no unknown references |
| Camera Choreography Planner | contract, entity trajectories | camera shots and required-event coverage | every required event needs a shot and observability record |
| Character/Object Trajectory Planner | contract events and timebase | frame-indexed states, motion primitives, attachments | one-based frames, continuity, explicit attachment transitions |
| Blender MCP/CLI Executor | validated plan and manifest | `.blend`, frame sequence, telemetry, execution response | controlled adapter only; persist response and fingerprints |
| Blender CLI Renderer / host video assembler | rendered PNG animation | sampled PNGs and `proxy.mp4` | host assembly is valid only after all required frames exist |
| Proxy Validator | artifacts, telemetry, plan | artifact report and deterministic findings | hard gate before VLM or training |
| Candidate Selection/Fallback | attempt reports | selected passing candidate or bounded failure | no silent contract rewrite; bounded attempts |
| Dataset + Evaluator | split records and artifacts | per-case reports, aggregate score, failure evidence | test split is evaluation-only |
| MetaHarnessOptimizer | train records | one-owner proposal and acceptance record | train strict improvement; dev no regression |

When the project uses different names, map by responsibility rather than forcing these filenames. Keep contracts serializable and hashable so runs remain comparable.
