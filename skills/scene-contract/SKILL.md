---
name: scene-contract
description: Normalize a natural-language scene prompt into a validated SceneContract.
---

# Scene Contract Skill

Input is a user scene prompt plus optional duration and frame rate. Output is a validated `SceneContract` JSON object containing entities, ordered events, relations, physics constraints, and camera constraints.

The builder may interpret wording, but the result must pass `SceneContract` validation before any trajectory or Blender code is generated. Empty prompts, missing timing, duplicate IDs, unknown entity references, and out-of-bounds events are failures. Runtime failures use `scene_contract_repair` as their repair route.
