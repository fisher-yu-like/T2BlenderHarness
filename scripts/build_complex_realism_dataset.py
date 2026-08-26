"""Build a small, deliberately difficult, prompt-diverse realism probe set.

The set is used to measure the current Harness without changing it.  The
``geometry.detail_required`` contract is intentionally stricter than the
ordinary proxy benchmark: a coarse primitive is evidence of a failed visual
representation, even when the plan and video container are valid.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset" / "complex-realism-v1"


def _scene(case_id: str, entities: list[tuple[str, str]], *, path_shape: str, seed: int, support: tuple[float, float], start: tuple[float, float], drop: tuple[float, float], blocker: bool = False) -> dict:
    required_ids = [entity_id for entity_id, _ in entities]
    required_kinds = {entity_id: kind for entity_id, kind in entities}
    return {
        "scene_id": f"complex-realism-{case_id}",
        "scene_seed": seed,
        "material": "ProxyWhiteMaterial",
        "entities": [
            {"id": entity_id, "kind": kind, "role": "actor" if entity_id == "character" else "receiver" if entity_id == "assistant" else "environment" if kind in {"support", "occluder"} else "target_object"}
            for entity_id, kind in entities
        ],
        "geometry": {
            "detail_required": True,
            "min_vertices": 256,
            "min_faces": 128,
            "required_entity_ids": required_ids,
            "required_entity_kinds": required_kinds,
            "forbid_primitive_hints": ["uv_sphere", "sphere", "cylinder", "cube"],
            "rationale": "Complex realism probe: no sphere/cylinder/cube stand-ins for people, props, or scene elements.",
        },
        "layout": {
            "character_start_position": [start[0], start[1], 0.0],
            "support_position": [support[0], support[1], 0.0],
            "drop_zone_position": [drop[0], drop[1], 0.0],
            "path_shape": path_shape,
            "object_scale": [0.72, 0.72, 0.72],
            "support_scale": [2.2, 1.5, 0.5],
            "lighting_rig": "soft_three_point",
            "foreground_blocker": blocker,
        },
        "camera": {
            "must_show_events": ["walk", "reach", "grasp", "lift", "carry", "place", "release"],
            "trajectory_types": ["follow", "orbit", "dolly"],
        },
        "required_artifacts": ["proxy.blend", "proxy.mp4", "telemetry.json", "frames/index.json", "geometry_raw.json"],
    }


PROMPTS = [
    (
        "complex-01",
        "Inside a sunlit glasshouse, establish the worktable, the foreground blocker, the assistant, and a red cup. The primary character walks around the blocker, reveals the red cup through the opening, reaches, grasps it, lifts it clear, carries it along a curved path, hands it to the assistant, and the assistant carries it to the marked destination, places it, and releases it. Use a continuous follow for the approach, an orbit around the handoff, and a slow dolly close-up for the final release; keep both actors, the cup, the blocker, and the destination legible without a cut.",
        _scene("complex-01", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("foreground_blocker", "occluder"), ("opening", "support"), ("assistant", "character"), ("red_cup", "prop")], path_shape="arc", seed=8101, support=(-1.2, -0.5), start=(-3.8, -2.2), drop=(2.2, 1.4), blocker=True),
    ),
    (
        "complex-02",
        "At a crowded workshop bench, the character walks in from the left, reaches across the support, grasps a blue cube, lifts it, carries it through a two-turn zigzag around a green ball, transfers it to the second actor, and the assistant rotates toward the drop zone, places the cube, and releases it. The camera must follow the entrance, orbit the handoff so both faces remain readable, then dolly into the placement while the green ball stays as a visible spatial landmark; preserve the full event order in one continuous take.",
        _scene("complex-02", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("assistant", "character"), ("blue_cube", "prop"), ("green_ball", "prop")], path_shape="zigzag", seed=8102, support=(-0.8, 0.0), start=(-3.5, -1.8), drop=(2.3, 0.4)),
    ),
    (
        "complex-03",
        "In a quiet library restoration room, reveal a yellow book behind the partition as the character advances from the far aisle. The character reaches over the support, grasps the book, lifts it slowly, carries it through a reverse zigzag around the assistant, places it on the marked destination, and releases it. Start with a wide reveal, switch to a restrained follow during the reach, orbit the reverse path to show the book staying in the hands, and finish with a gentle dolly close-up that still includes the destination and both actors.",
        _scene("complex-03", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("partition", "occluder"), ("assistant", "character"), ("yellow_book", "prop")], path_shape="reverse_zigzag", seed=8103, support=(0.0, 0.3), start=(-4.0, 1.8), drop=(2.0, -1.2), blocker=True),
    ),
    (
        "complex-04",
        "On an outdoor film set, the primary character walks from the loading mark to the table, reaches for a red cup, grasps it, lifts it, carries it across the frame, places it at the drop platform, and releases it while the assistant simultaneously walks in carrying a blue cube, reaches, grasps, lifts, carries, places, and releases the cube at a second visible mark. Use a lateral follow for the crossing, an orbit at the synchronized midpoint, and a final dolly that frames both completed placements; the camera must preserve readable relative timing rather than hiding either action.",
        _scene("complex-04", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("assistant", "character"), ("red_cup", "prop"), ("blue_cube", "prop")], path_shape="crossing", seed=8104, support=(-1.0, 0.2), start=(-4.0, -2.0), drop=(2.4, 1.1)),
    ),
    (
        "complex-05",
        "In a narrow museum corridor, begin behind the foreground blocker and reveal the green ball on the surface as the character walks forward. The character reaches, grasps, lifts, carries the ball through an S-shaped path past the assistant, places it at the drop zone, and releases it; then hold the final composition long enough to show the assistant and the empty support. Use a reveal followed by a close follow, orbit around the S-shaped carry, and a measured dolly out after release, with no abrupt camera jumps and no cropped hands or object.",
        _scene("complex-05", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("foreground_blocker", "occluder"), ("assistant", "character"), ("green_ball", "prop")], path_shape="s_curve", seed=8105, support=(-0.6, -0.2), start=(-3.2, -1.4), drop=(2.2, 1.5), blocker=True),
    ),
    (
        "complex-06",
        "For a small stage play, establish the full set with the character, assistant, table, red cup, yellow book, and marked destination. The character walks to the table, reaches, grasps the red cup, lifts it, carries it behind the assistant, places it, and releases it; immediately afterward the assistant walks to the table, reaches for the yellow book, grasps it, lifts it, carries it back along the reverse path, places it, and releases it. Choreograph a follow for the first pass, a high orbit showing the crossing, and a dolly close-up on the second release while keeping the two objects visually distinct.",
        _scene("complex-06", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("assistant", "character"), ("red_cup", "prop"), ("yellow_book", "prop")], path_shape="reverse_crossing", seed=8106, support=(0.0, 0.0), start=(-3.0, -2.0), drop=(2.0, 1.0)),
    ),
    (
        "complex-07",
        "In a kitchen rehearsal, the character walks around the island, reveals a blue cube beside a red cup, reaches for the cube, grasps it, lifts it, carries it in a slow arc, hands it to the assistant, and the assistant places it at the marked destination and releases it. The red cup remains an untouched reference object on the support. Make the camera follow the walk, orbit the handoff with both hands visible, and dolly into the final placement; preserve spatial continuity and a clear final hold.",
        _scene("complex-07", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("foreground_blocker", "occluder"), ("assistant", "character"), ("blue_cube", "prop"), ("red_cup", "prop")], path_shape="arc", seed=8107, support=(-0.7, 0.1), start=(-3.6, -1.8), drop=(2.1, 0.8), blocker=True),
    ),
    (
        "complex-08",
        "At a robotics demonstration, start with a wide view of the platform and opening. The character advances, reaches through the opening, grasps a yellow book, lifts it, carries it along a reverse zigzag while the assistant tracks beside it, places the book on the drop platform, and releases it. Use a wide reveal, a smooth follow that respects the moving pair, an orbit at the midpoint, and a slow dolly toward the book at the end; every phase must remain observable and the final object must not disappear behind the support.",
        _scene("complex-08", [("character", "character"), ("table", "support"), ("drop_zone", "support"), ("opening", "support"), ("assistant", "character"), ("yellow_book", "prop")], path_shape="reverse_zigzag", seed=8108, support=(-0.9, 0.2), start=(-3.9, -1.5), drop=(2.3, 1.2)),
    ),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (case_id, prompt, proxy_scene) in enumerate(PROMPTS, start=1):
        event_order = ["walk", "reach", "grasp", "lift", "carry", "place", "release"]
        required_constraints = ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle"]
        if "reveal" in prompt.lower():
            required_constraints.append("occlusion_reveal")
        if "reverse" in prompt.lower():
            required_constraints.append("reverse_path")
        records.append({
            "case_id": case_id,
            "difficulty": 5,
            "duration_s": 18.0,
            "fps": 24,
            "split": "test",
            "template_family": "complex_story_realism_probe",
            "evaluator_version": "real-v4-geometry-realism-v1",
            "prompt": prompt,
            "proxy_scene": proxy_scene,
            "oracle_expectations": {
                "event_order": event_order,
                "required_attachment_actions": ["attach", "detach"],
                "required_camera_constraints": required_constraints,
                "required_camera_types": ["follow", "orbit", "dolly"],
                "required_entity_ids": proxy_scene["geometry"]["required_entity_ids"],
                "required_entity_kinds": proxy_scene["geometry"]["required_entity_kinds"],
                "required_motion_primitives": ["ease_in_out", "linear"],
            },
        })
    (OUT / "manifest.jsonl").write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    (OUT / "splits.json").write_text(json.dumps({"train": [], "dev": [], "test": [record["case_id"] for record in records], "calibration": []}, indent=2), encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps({
        "dataset_id": "complex-realism-v1",
        "purpose": "Frozen no-Harness-change probe for story complexity, multi-actor choreography, and geometric realism.",
        "case_count": len(records),
        "render_policy": {"resolution": [512, 512], "samples": 64, "blender_binary": "D:/blender/blender.exe"},
        "geometry_policy": "detail_required=true; primitive stand-ins are hard failures; minimum 256 vertices and 128 faces per required mesh.",
    }, indent=2), encoding="utf-8")
    print(json.dumps({"dataset_root": str(OUT), "case_count": len(records), "case_ids": [record["case_id"] for record in records]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
