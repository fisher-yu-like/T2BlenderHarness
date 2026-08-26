"""Build a 140-case compositional-holdout prompt and proxy dataset.

The oracle expectations in this manifest are authored independently from the
current parser/planner.  The benchmark compares the generated plan against
these expectations so a self-consistent planner cannot earn a perfect score
merely by validating its own output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


OBJECTS = ["red cup", "blue cube", "green ball", "yellow book"]
SUPPORTS = ["table", "worktable", "platform", "support surface"]
DROP_ZONES = ["drop zone", "drop platform", "marked destination"]

BASE_EVENTS = ["walk", "reach", "grasp", "lift", "carry", "place", "release"]
BASE_CAMERAS = ["follow", "orbit", "dolly"]


FAMILIES: list[dict[str, Any]] = [
    {
        "id": "supported_baseline",
        "split": "train",
        "difficulty": 3,
        "template": (
            "Establish the {support} and {object}. The character walks to the support, reaches for the object, "
            "grasps it, lifts it clear, carries it to the {drop_zone}, places it, and releases it. "
            "Follow the approach, orbit during carry, and dolly into the release close-up."
        ),
        "events": BASE_EVENTS,
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle"],
        "primitives": ["ease_in_out", "linear"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "lexical_action_paraphrase",
        "split": "train",
        "difficulty": 4,
        "template": (
            "Open on the {support}. The actor strolls toward it, extends an arm to the {object}, seizes the object, "
            "hoists it above the surface, transports it to the {drop_zone}, sets it down, and lets go. "
            "Track the approach, circle the handoff, and push in for the final release."
        ),
        "events": BASE_EVENTS,
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle"],
        "primitives": ["ease_in_out", "linear"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "dual_object_handoff",
        "split": "train",
        "difficulty": 4,
        "template": (
            "Show a red cup and a blue cube on the {support}. The character walks in, reaches for the cup, grasps "
            "and lifts it, carries it to the {drop_zone}, places and releases it, then repeats the same handoff for "
            "the cube. Keep both objects visible during the two attachment lifecycles; use follow, orbit, and dolly shots."
        ),
        "events": BASE_EVENTS + ["walk", "reach", "grasp", "lift", "carry", "place", "release"],
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "dual_handoff"],
        "primitives": ["ease_in_out", "linear"],
        "attachments": ["attach", "detach", "attach", "detach"],
        # Both named objects are added explicitly in _proxy_scene.  Keeping
        # this list empty prevents a duplicate blue_cube entity.
        "extra_entities": [],
    },
    {
        "id": "foreground_occluder",
        "split": "train",
        "difficulty": 4,
        "template": (
            "A foreground blocker hides the {object} at first. Reveal the object before the character walks to the "
            "{support}; the character reaches, grasps, lifts, carries, places, and releases it at the {drop_zone}. "
            "Orbit around the blocker, follow the carry, and dolly to the release while the object remains observable."
        ),
        "events": ["reveal"] + BASE_EVENTS,
        "cameras": BASE_CAMERAS,
        "constraints": ["occlusion_reveal", "target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle"],
        "primitives": ["ease_in_out", "linear", "hold"],
        "attachments": ["attach", "detach"],
        "extra_entities": [{"id": "foreground_blocker", "kind": "occluder", "role": "environment"}],
    },
    {
        "id": "pause_return_cycle",
        "split": "train",
        "difficulty": 5,
        "template": (
            "The character approaches the {support}, reaches for and grasps the {object}, lifts and carries it to the "
            "{drop_zone}, places it, releases it, pauses for a beat, and returns to the starting mark. "
            "Use a follow shot for approach, an orbit for carry, and a final dolly that holds after the return."
        ),
        "events": BASE_EVENTS + ["pause", "return"],
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "post_release_hold"],
        "primitives": ["ease_in_out", "linear", "hold"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "camera_rhythm_and_hold",
        "split": "train",
        "difficulty": 5,
        "template": (
            "Use a locked establishing hold, then a follow shot as the character walks to the {support}. "
            "Reach for, grasp, lift, and carry the {object}; orbit the transport, place and release at the {drop_zone}, "
            "then dolly closer and hold the last frame for two seconds."
        ),
        "events": BASE_EVENTS,
        "cameras": ["hold", "follow", "orbit", "dolly"],
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "post_release_hold"],
        "primitives": ["ease_in_out", "linear", "hold"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "novel_spatial_transfer",
        "split": "dev",
        "difficulty": 5,
        "template": (
            "Begin behind the {support}. The actor advances in a shallow arc, extends toward the {object}, snatches it, "
            "elevates it without penetration, conveys it along an S-curve to the {drop_zone}, lowers it onto the mark, "
            "and uncouples the grip. Track the arc, rotate around the handoff, and move in for a close release."
        ),
        "events": BASE_EVENTS,
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "no_penetration"],
        "primitives": ["ease_in_out", "linear", "orbit"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "reverse_zigzag_transfer",
        "split": "dev",
        "difficulty": 5,
        "template": (
            "Open on the {object} resting on the {support}. The actor approaches from the far side, reaches across, "
            "grasps and lifts the object, carries it through a reverse zigzag to the {drop_zone}, settles it, and "
            "releases only after the object is stable. Follow the approach, arc against the travel direction, and "
            "push in for the final placement."
        ),
        "events": BASE_EVENTS,
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "reverse_path"],
        "primitives": ["ease_in_out", "linear", "orbit"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "timed_reveal_transfer",
        "split": "dev",
        "difficulty": 5,
        "template": (
            "Hold on an empty {support}, then reveal the {object} from the side before the actor advances. The actor "
            "reaches, grips, raises, transports the object to the {drop_zone}, lowers it onto the mark, and lets go. "
            "Keep the reveal observable, track the handoff, orbit during transport, and move in without cutting away."
        ),
        "events": ["reveal"] + BASE_EVENTS,
        "cameras": ["hold", "follow", "orbit", "dolly"],
        "constraints": ["occlusion_reveal", "target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle"],
        "primitives": ["hold", "ease_in_out", "linear", "orbit", "dolly"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "pause_resume_transfer",
        "split": "dev",
        "difficulty": 5,
        "template": (
            "The actor walks to the {support}, reaches for the {object}, grasps and lifts it, carries it to the "
            "{drop_zone}, pauses while holding it above the mark, places it, releases, and returns to the start. "
            "Use a following approach, a circular carry shot, a release dolly, and a final held composition."
        ),
        "events": BASE_EVENTS + ["pause", "return"],
        "cameras": ["follow", "orbit", "dolly", "hold"],
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "post_release_hold"],
        "primitives": ["ease_in_out", "linear", "hold", "orbit", "dolly"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "diagonal_clearance_transfer",
        "split": "dev",
        "difficulty": 5,
        "template": (
            "Keep the {object} and the {drop_zone} visible as the actor approaches the {support} on a diagonal. "
            "Reach without collision, grasp, lift clear of the surface, carry across the frame, place on the "
            "destination, and detach. Preserve continuous coverage with follow, orbit, and dolly shots around the "
            "diagonal transfer."
        ),
        "events": BASE_EVENTS,
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "no_penetration", "diagonal_transfer"],
        "primitives": ["ease_in_out", "linear", "orbit", "dolly"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
    {
        "id": "two_actor_exchange",
        "split": "dev",
        "difficulty": 5,
        "template": (
            "The primary actor walks to the {support}, reaches for and grasps the {object}, lifts it, and carries it to "
            "the assistant at the {drop_zone}. The assistant receives it, holds it briefly, and places it down before "
            "release. Keep both actors and the object visible, with follow, orbit, and dolly coverage."
        ),
        "events": BASE_EVENTS + ["receive", "hold"],
        "cameras": BASE_CAMERAS,
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "two_actor_visibility"],
        "primitives": ["ease_in_out", "linear", "hold"],
        "attachments": ["attach", "detach", "attach", "detach"],
        "extra_entities": [{"id": "assistant", "kind": "character", "role": "receiver"}],
    },
    {
        "id": "hidden_geometry_reveal",
        "split": "test",
        "difficulty": 5,
        "template": (
            "Start with the {object} hidden behind a partition and the destination off axis. The actor crouches, "
            "reaches through the opening, seizes the object, raises it, carries it backward around the partition, "
            "settles it on the {drop_zone}, and unhands it. Reveal the opening, arc counterclockwise, dolly through the "
            "reveal, and preserve a final hold."
        ),
        "events": ["reveal", "crouch"] + BASE_EVENTS,
        "cameras": ["hold", "orbit", "dolly"],
        "constraints": ["occlusion_reveal", "target_visible_before_grasp", "attachment_lifecycle", "post_release_hold"],
        "primitives": ["hold", "linear", "orbit", "dolly"],
        "attachments": ["attach", "detach"],
        "extra_entities": [
            {"id": "partition", "kind": "occluder", "role": "environment"},
            {"id": "opening", "kind": "support", "role": "environment"},
        ],
    },
    {
        "id": "counterfactual_camera_constraint",
        "split": "test",
        "difficulty": 5,
        "template": (
            "Do not cut away during contact. The actor approaches the {support}, extends an arm, picks up the {object}, "
            "raises it, transports it diagonally to the {drop_zone}, sets it atop the mark, and detaches. "
            "Back away for the approach, swing around the diagonal transfer, push in at placement, and keep the target "
            "visible before and after grasp."
        ),
        "events": BASE_EVENTS,
        "cameras": ["follow", "orbit", "dolly"],
        "constraints": ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "no_cut_during_contact"],
        "primitives": ["ease_in_out", "linear", "orbit", "dolly"],
        "attachments": ["attach", "detach"],
        "extra_entities": [],
    },
]


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _proxy_scene(
    family: dict[str, Any],
    object_name: str,
    support_name: str,
    drop_zone: str,
    *,
    scene_number: int,
) -> dict[str, Any]:
    entities = [
        {"id": "character", "kind": "character", "role": "actor"},
        {"id": "table", "kind": "support", "role": "environment"},
        {"id": "drop_zone", "kind": "support", "role": "environment"},
    ]
    if family["id"] == "dual_object_handoff":
        # The prompt explicitly names both objects.  Do not derive the proxy
        # target from the rotating variant object, or the oracle can disagree
        # with the prompt (and can even create duplicate blue_cube IDs).
        entities.extend([
            {"id": "red_cup", "kind": "prop", "role": "target_object"},
            {"id": "blue_cube", "kind": "prop", "role": "secondary_target"},
        ])
    else:
        entities.append({"id": object_name.replace(" ", "_"), "kind": "prop", "role": "target_object"})
    entities.extend(family["extra_entities"])
    return {
        "scene_id": f"proxy-scene-{scene_number:03d}",
        "scene_seed": 7000 + scene_number,
        "layout": {
            "support_position": [round(-2.4 + (scene_number % 7) * 0.8, 2), round(-1.5 + (scene_number % 5) * 0.7, 2), 0.0],
            "drop_zone_position": [round(1.0 + (scene_number % 6) * 0.65, 2), round(1.6 - (scene_number % 4) * 0.8, 2), 0.0],
            "character_start_position": [round(-4.0 + (scene_number % 4) * 0.5, 2), round(-2.8 + (scene_number % 6) * 0.45, 2), 0.0],
            "support_scale": [round(1.8 + (scene_number % 5) * 0.35, 2), round(1.2 + (scene_number % 3) * 0.3, 2), 0.5],
            "object_scale": [round(0.65 + (scene_number % 4) * 0.12, 2)] * 3,
            "path_shape": ["straight", "arc", "zigzag", "s_curve", "reverse_arc"][scene_number % 5],
            "lighting_rig": ["key_left", "key_right", "top_soft", "side_rim"][scene_number % 4],
        },
        "geometry": {
            "support_label": support_name,
            "drop_zone_label": drop_zone,
            "occluder_count": sum(item["kind"] == "occluder" for item in family["extra_entities"]),
            "requires_opening": family["id"] == "hidden_geometry_reveal",
        },
        "entities": entities,
        "material": "ProxyWhiteMaterial",
        "required_artifacts": ["proxy.blend", "proxy.mp4", "telemetry.json", "frames/index.json"],
        "camera": {"trajectory_types": family["cameras"], "must_show_events": family["events"]},
    }


def build_dataset(root: str | Path = "dataset/trajectory-v3-hard") -> dict[str, Any]:
    destination = Path(root)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    splits = {"train": [], "dev": [], "test": []}
    labels: list[dict[str, Any]] = []
    proxy_specs: list[dict[str, Any]] = []

    for family_index, family in enumerate(FAMILIES):
        for variant in range(10):
            object_name = OBJECTS[(family_index + variant) % len(OBJECTS)]
            support_name = SUPPORTS[(family_index * 2 + variant) % len(SUPPORTS)]
            drop_zone = DROP_ZONES[(family_index + variant) % len(DROP_ZONES)]
            prompt = family["template"].format(object=object_name, support=support_name, drop_zone=drop_zone)
            prompt += (
                f" Variant {variant + 1}: begin from the {'left' if variant % 2 == 0 else 'right'} side, "
                f"use a {['brief', 'measured', 'delayed'][variant % 3]} anticipation, and preserve a "
                f"{['wide', 'medium', 'tight'][variant % 3]} final composition."
            )
            case_id = f"hard-{family_index + 1:02d}-{variant + 1:02d}"
            proxy = _proxy_scene(
                family,
                object_name,
                support_name,
                drop_zone,
                scene_number=family_index * 10 + variant + 1,
            )
            oracle = {
                "event_order": family["events"],
                "required_camera_types": family["cameras"],
                "required_camera_constraints": family["constraints"],
                "required_motion_primitives": family["primitives"],
                "required_attachment_actions": family["attachments"],
                "required_entity_ids": [item["id"] for item in proxy["entities"]],
                "required_entity_kinds": {item["id"]: item["kind"] for item in proxy["entities"]},
            }
            record = {
                "case_id": case_id,
                "prompt": prompt,
                "template_family": family["id"],
                "split": family["split"],
                "difficulty": family["difficulty"],
                "duration_s": 20.0 if family["difficulty"] >= 5 else 16.0,
                "fps": 24,
                "evaluator_version": "deterministic-v2-independent-oracle",
                "proxy_scene": proxy,
                "oracle_expectations": oracle,
            }
            records.append(record)
            splits[family["split"]].append(case_id)
            proxy_specs.append({"case_id": case_id, "proxy_scene": proxy})
            labels.append(
                {
                    "case_id": case_id,
                    "pass_fail": "unreviewed",
                    "label_source": "independent_oracle_pending_visual_calibration",
                    "primary_failure_owner": "unreviewed",
                    "vlm_status": "unavailable",
                }
            )

    (destination / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    (destination / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (destination / "labels.jsonl").write_text(
        "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels), encoding="utf-8"
    )
    (destination / "proxy_specs.jsonl").write_text(
        "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in proxy_specs), encoding="utf-8"
    )
    metadata = {
        "dataset_id": "trajectory-v3-hard",
        "schema_version": "trajectory-dataset-v3-hard",
        "cases": len(records),
        "splits": {name: len(values) for name, values in splits.items()},
        "families": len(FAMILIES),
        "split_policy": "template_family_holdout_compositional",
        "oracle_policy": "independent_expectations_not_derived_from_runtime_plan",
        "proxy_policy": "white_material_geometry_and_artifact_contract_per_case",
        "fingerprint": _fingerprint(records),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(build_dataset(), indent=2, sort_keys=True))
