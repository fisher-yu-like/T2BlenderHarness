"""Compile a TrajectoryPlan into a Blender Python script without importing bpy."""

from __future__ import annotations

import json
from pathlib import Path

from videoact.contracts import RunManifest, TrajectoryPlan


def compile_proxy_script(
    plan: TrajectoryPlan,
    manifest: RunManifest,
    output_path: str | Path | None = None,
) -> str:
    plan_payload = json.dumps(plan.model_dump(mode="json"), sort_keys=True)
    metadata = {
        "run_id": manifest.run_id,
        "harness_version": manifest.harness_version,
        "evaluator_version": manifest.evaluator_version,
        "plan_hash": manifest.plan_hash,
        "manifest_hash": manifest.content_hash(),
        "frame_start": manifest.frame_start,
        "frame_end": manifest.frame_end,
    }
    script = f'''"""Generated white-material proxy scene. Do not edit by hand."""
import json
import bpy

PLAN = {plan_payload!r}
METADATA = {json.dumps(metadata, sort_keys=True)!r}

scene = bpy.context.scene
metadata = json.loads(METADATA)
for key, value in metadata.items():
    scene["proxy_" + key] = value
scene.frame_start = metadata["frame_start"]
scene.frame_end = metadata["frame_end"]

# The runtime adapter owns execution; this script only materializes the validated plan.
for entity_id, trajectory in PLAN["entities"].items():
    if entity_id not in bpy.data.objects:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.name = entity_id
    else:
        obj = bpy.data.objects[entity_id]
    for state in trajectory["states"]:
        obj.location = state["position"]
        obj.keyframe_insert(data_path="location", frame=state["frame"])
'''
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(script, encoding="utf-8")
    return script
