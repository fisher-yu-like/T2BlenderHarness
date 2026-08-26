"""Build the versioned complex trajectory-focused train/dev/test dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.scene_contract import SceneContractBuilder
from videoact.trajectory import TrajectoryPlanner


OBJECTS = ["red cup", "blue cube", "green ball", "yellow book"]
SUPPORTS = ["table", "worktable", "platform", "support surface"]
VARIANT_CUES = [
    "Begin from the left side of frame.",
    "Begin from the right side of frame.",
    "Use a brief anticipation hold before the approach.",
    "Use a measured ease-in and ease-out through each phase.",
    "Keep the target centered during every handoff.",
    "Allow a short hold after the final release.",
    "Reveal the support edge before the character enters.",
    "End with a two-second observable hold.",
]
FAMILIES = [
    (
        "approach_lift_carry_release",
        "Begin with a wide establishing shot. The character walks to the {support}, reaches for the {object}, grasps it, lifts it, carries it to the drop zone, places it down, and releases it. The camera follows the approach and dollies into a close-up of the release while keeping the object visible before grasp.",
        16.0,
    ),
    (
        "orbit_transfer",
        "Use a stable overview before the character walks to the {support}. The character reaches for and grasps the {object}, lifts it clear of the surface, carries it across the frame, places it at the drop zone, and releases it. The camera follows the handoff and orbits during the carry before holding the final placement.",
        16.0,
    ),
    (
        "occlusion_reveal",
        "A foreground blocker creates a brief occlusion. The camera reveals the {object}, then the character walks to the {support}, reaches, grasps, lifts, carries, places, and releases the object. Track the reveal, orbit around the carry, and preserve a close-up of the release.",
        18.0,
    ),
    (
        "closeup_grasp_release",
        "Start wide on the {support}. The character walks in, reaches toward the {object}, grasps it, lifts it slowly, carries it to a marked drop zone, places it down, and releases it. Follow the approach, then dolly from the hand close-up to the final release without losing target visibility.",
        14.0,
    ),
    (
        "dolly_drop_zone",
        "Establish the {support} and the {object} in a wide shot. The character walks and reaches in one continuous approach, grasps the object, lifts it, carries it to the drop zone, places it, and releases it. Dolly toward the grasp, orbit briefly during transport, and hold after release.",
        16.0,
    ),
    (
        "orbit_before_place",
        "The character walks from the left to the {support}, reaches for the {object}, and grasps it. Lift the object, carry it along a smooth arc, place it at the drop zone, and release it. The camera follows the character, orbits the carried object, then uses a close-up for the placement and release.",
        18.0,
    ),
    (
        "long_duration_retake",
        "In a long continuous shot, establish the {support}, let the character walk toward it, reach, grasp the {object}, lift it above the surface, carry it through the scene, place it at the drop zone, and release it. Track the full trajectory, orbit during the carry, and dolly into the final close-up.",
        24.0,
    ),
    (
        "support_contact_lifecycle",
        "Show the {object} resting on the {support} before any contact. The character walks in, reaches, grasps, lifts without penetration, carries the object, places it back onto the drop zone, and releases it. Follow the approach and use a close-up to verify the attachment lifecycle.",
        16.0,
    ),
    (
        "multi_shot_handoff",
        "Use an establishing hold, a follow shot for the walk, an orbit shot for the lift-and-carry phase, and a dolly close-up for release. The character walks to the {support}, reaches for and grasps the {object}, lifts it, carries it to the drop zone, places it, and releases it while the target remains observable before grasp.",
        20.0,
    ),
    (
        "smooth_phase_transfer",
        "Plan a smooth ease-in approach: the character walks to the {support}, reaches toward the {object}, grasps it, lifts it, carries it at a steady height, places it at the drop zone, and releases it. The camera follows the approach, orbits the steady carry, and dollies into the release close-up.",
        16.0,
    ),
]


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_dataset(root: str | Path = "dataset/trajectory-v2") -> dict[str, Any]:
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    builder = SceneContractBuilder()
    planner = TrajectoryPlanner()
    records: list[dict[str, Any]] = []
    splits = {"train": [], "dev": [], "test": []}
    labels: list[dict[str, Any]] = []

    for family_index, (family, template, duration_s) in enumerate(FAMILIES):
        for variant in range(8):
            object_name = OBJECTS[(family_index + variant) % len(OBJECTS)]
            support_name = SUPPORTS[(family_index + variant) % len(SUPPORTS)]
            prompt = template.format(object=object_name, support=support_name) + " " + VARIANT_CUES[variant]
            case_id = f"traj-{family_index + 1:02d}-{variant + 1:02d}"
            contract = builder.build(prompt, duration_s=duration_s, fps=24)
            plan = planner.plan(contract)
            record = {
                "case_id": case_id,
                "prompt": prompt,
                "category": family,
                "entities": [entity.model_dump(mode="json") for entity in contract.entities],
                "required_events": [event.id for event in contract.events],
                "expected_relations": [relation.model_dump(mode="json") for relation in contract.relations],
                "duration_s": duration_s,
                "fps": 24,
                "evaluator_version": "deterministic-v1",
                "trajectory_expectations": {
                    "event_order": [event.id for event in contract.events],
                    "camera_types": sorted({shot.trajectory_type for shot in plan.camera.shots}),
                    "camera_constraints": list(contract.camera_constraints),
                    "min_character_states": len(plan.entities["character"].states),
                    "motion_primitive_types": sorted({primitive.type for primitive in plan.entities["character"].motion_primitives}),
                    "attachment_actions": [event.action for event in plan.entities["character"].attachment_events],
                },
            }
            records.append(record)
            split = "train" if variant < 5 else "dev" if variant < 7 else "test"
            splits[split].append(case_id)
            labels.append(
                {
                    "case_id": case_id,
                    "pass_fail": "unreviewed",
                    "event_coverage": None,
                    "physics_plausibility": None,
                    "camera_quality": None,
                    "primary_failure_owner": "unreviewed",
                }
            )

    (destination / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    (destination / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "labels.jsonl").write_text(
        "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels), encoding="utf-8"
    )
    metadata = {
        "dataset_id": "trajectory-v2",
        "schema_version": "trajectory-dataset-v2",
        "cases": len(records),
        "splits": {name: len(values) for name, values in splits.items()},
        "families": len(FAMILIES),
        "fingerprint": _fingerprint(records),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    build_dataset()
