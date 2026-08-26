---
name: blender-proxy-executor
description: Execute validated proxy scripts through the controlled Blender CLI/MCP adapter.
---

# Blender Proxy Executor Skill

Use `videoact.BlenderAdapter` for every Blender operation. CLI and MCP calls return the same `ExecutionResult` shape; MCP requests are appended to `mcp_calls.jsonl`, and a failed MCP call may fall back to CLI with `fallback_used=true`.

The proxy script must be compiled from a validated `TrajectoryPlan`, carry the run ID, Harness version, evaluator version, frame range, plan hash, and manifest hash, and must not be published without a fresh manifest. White-material setup and probes remain Blender-side helpers and are not imported during ordinary test collection.
