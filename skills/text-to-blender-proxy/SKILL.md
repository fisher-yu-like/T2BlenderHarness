---
name: text-to-blender-proxy
description: Run the contract-first Text-to-Blender proxy Harness with bounded repair and resume checks.
---

# Text-to-Blender Proxy Skill

Run stages in this order: `contract`, `plan`, `execute`, `render`, `evaluate`, `repair`, `finalize`. The contract and plan must validate before the adapter is called. The inner loop keeps immutable numbered attempts, routes findings explicitly, promotes only a passing candidate, and stops after six attempts. Resume is allowed only when prompt and Harness fingerprints match the existing final selection.
