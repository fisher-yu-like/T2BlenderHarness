"""Build the reproducible 50/60/30 multi-entity Director dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ACTOR_SETS = [
    ("Alice", "Bob"),
    ("Alice", "Carla"),
    ("Bob", "Dana"),
    ("Alice", "Bob", "Carla"),
    ("Bob", "Carla", "Dana"),
]
PROP_SETS = [
    ("red cube", "blue cup"),
    ("green ball", "yellow book"),
    ("red cup", "blue cube"),
    ("green book", "yellow ball"),
    ("red cube", "green cup", "blue book"),
    ("yellow cube", "blue ball", "red book", "green cup"),
]
CAMERA_RHYTHMS = (
    "a lateral follow, a restrained orbit at contact, and a final dolly reveal",
    "a wide establishing hold, a curved follow, and a short rack-like reveal",
    "a low follow through the handoff, an orbit that preserves the axis, and a held finish",
    "a measured push-in before contact, a reverse arc during transfer, and a stable final frame",
)
VARIANT_DETAILS = (
    "begin from the left lane with a half-beat anticipation",
    "begin from the right lane and preserve a readable pause before contact",
    "cross the foreground only after the first object is secured",
    "delay the reveal until the receiver has entered the shared frame",
    "keep the second object visible while the first object changes owner",
    "use a shallow S-curve and keep both hands separated in depth",
    "hold the post-placement composition long enough to verify support",
    "reverse the role order without changing the object identities",
    "make the final return visibly distinct from the transport path",
    "preserve a clean axis through the last exchange",
)

TRAIN_FAMILIES = (
    "sequential_transfers",
    "repeated_handoffs",
    "concurrent_independent_work",
    "occlusion_reveal",
    "role_swap_pause_return_crossing",
)
DEV_FAMILIES = (
    "dev_three_actor_two_prop",
    "dev_two_actor_three_prop",
    "dev_three_actor_three_prop",
    "dev_role_reversal",
    "dev_occlusion_countermotion",
    "dev_four_prop_camera_constraint",
)
TEST_FAMILIES = (
    "test_role_reversal_final_owner",
    "test_counterfactual_camera_visibility",
    "test_prohibited_crossing_support",
)


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(_json_line(record) for record in sorted(records, key=lambda item: item["case_id"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ids(labels: list[str], *, prefix: str) -> list[str]:
    if prefix == "actor":
        return [f"{prefix}_{chr(ord('a') + index)}" for index, _ in enumerate(labels)]
    return [f"{prefix}_{index}" for index, _ in enumerate(labels)]


def _make_event_graph(actor_ids: list[str], prop_ids: list[str], family: str) -> list[dict[str, Any]]:
    first_actor = actor_ids[0]
    second_actor = actor_ids[1]
    events = [
        {"id": "approach_01", "action": "approach", "participant_ids": [first_actor], "target_ids": [prop_ids[0]], "depends_on": [], "start": 0.0, "end": 2.0},
        {"id": "attach_01", "action": "attach", "participant_ids": [first_actor], "target_ids": [prop_ids[0]], "depends_on": ["approach_01"], "start": 2.0, "end": 3.0},
        {"id": "carry_01", "action": "carry", "participant_ids": [first_actor], "target_ids": [prop_ids[0]], "depends_on": ["attach_01"], "start": 3.0, "end": 6.0},
        {"id": "handoff_01", "action": "handoff", "participant_ids": [first_actor, second_actor], "target_ids": [prop_ids[0]], "depends_on": ["carry_01"], "start": 6.0, "end": 7.5},
        {"id": "place_01", "action": "place", "participant_ids": [second_actor], "target_ids": [prop_ids[0]], "depends_on": ["handoff_01"], "start": 7.5, "end": 9.0},
        {"id": "detach_01", "action": "detach", "participant_ids": [second_actor], "target_ids": [prop_ids[0]], "depends_on": ["place_01"], "start": 9.0, "end": 10.0},
    ]
    if len(prop_ids) > 1:
        events.extend([
            {"id": "attach_02", "action": "attach", "participant_ids": [second_actor], "target_ids": [prop_ids[1]], "depends_on": ["detach_01"], "start": 10.0, "end": 11.0},
            {"id": "carry_02", "action": "carry", "participant_ids": [second_actor], "target_ids": [prop_ids[1]], "depends_on": ["attach_02"], "start": 11.0, "end": 14.0},
            {"id": "handoff_02", "action": "handoff", "participant_ids": [second_actor, first_actor], "target_ids": [prop_ids[1]], "depends_on": ["carry_02"], "start": 14.0, "end": 15.0},
            {"id": "place_02", "action": "place", "participant_ids": [first_actor], "target_ids": [prop_ids[1]], "depends_on": ["handoff_02"], "start": 15.0, "end": 16.0},
            {"id": "detach_02", "action": "detach", "participant_ids": [second_actor], "target_ids": [prop_ids[1]], "depends_on": ["place_02"], "start": 15.0, "end": 16.0},
        ])
    if family in {"concurrent_independent_work", "dev_three_actor_two_prop", "dev_three_actor_three_prop"}:
        events[0]["concurrency_group"] = "parallel_approach"
        events[0]["participant_ids"] = [first_actor]
        events[1]["concurrency_group"] = "parallel_approach"
        events[1]["participant_ids"] = [second_actor]
        events[1]["target_ids"] = [prop_ids[1]]
        events[1]["depends_on"] = []
    if family in {"occlusion_reveal", "dev_occlusion_countermotion", "test_counterfactual_camera_visibility"}:
        events.insert(0, {"id": "reveal_01", "action": "reveal", "participant_ids": [first_actor], "target_ids": [prop_ids[0]], "depends_on": [], "start": 0.0, "end": 1.5})
        for event in events[1:]:
            if "reveal_01" not in event["depends_on"] and event["id"] == "approach_01":
                event["depends_on"] = ["reveal_01"]
    if family in {"role_swap_pause_return_crossing", "dev_role_reversal", "test_role_reversal_final_owner"}:
        events.extend([
            {"id": "pause_01", "action": "pause", "participant_ids": [second_actor], "target_ids": [prop_ids[0]], "depends_on": ["detach_01"], "start": 16.0, "end": 17.0},
            {"id": "return_01", "action": "return", "participant_ids": [second_actor], "target_ids": [prop_ids[0]], "depends_on": ["pause_01"], "start": 17.0, "end": 19.0},
        ])
    return events


def _make_interactions(actor_ids: list[str], prop_ids: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"attachment_lifecycle:{receiver}:{prop_id}:{transfer_event_id}",
            "prop_id": prop_id,
            "giver_id": giver,
            "receiver_id": receiver,
            "attach_event_id": "attach_01" if index == 0 else "attach_02",
            "transfer_event_id": transfer_event_id,
            "detach_event_id": "detach_01" if index == 0 else "detach_02",
            "final_owner_id": receiver,
            "final_support_id": "support_surface",
        }
        for index, prop_id in enumerate(prop_ids)
        for giver, receiver, transfer_event_id in [
            (actor_ids[0], actor_ids[1], "handoff_01") if index == 0 else (actor_ids[1], actor_ids[0], "handoff_02")
        ]
    ]


def _make_camera_evidence(actor_ids: list[str], prop_ids: list[str], events: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    target_ids = [*actor_ids, *prop_ids]
    return [
        {
            "shot_id": "shot_establish",
            "required_event_ids": [event["id"] for event in events[:2]],
            "target_ids": target_ids,
            "visibility_predicates": {target_id: "visible" for target_id in target_ids},
            "max_occlusion": 0.2 if "occlusion" in family or "counterfactual" in family else 0.35,
            "innovation_intent_evidence": f"camera preserves all targets for {family}",
        },
        {
            "shot_id": "shot_handoff_orbit",
            "required_event_ids": ["handoff_01"],
            "target_ids": [actor_ids[0], actor_ids[1], prop_ids[0]],
            "visibility_predicates": {target_id: "visible" for target_id in [actor_ids[0], actor_ids[1], prop_ids[0]]},
            "max_occlusion": 0.25,
            "innovation_intent_evidence": "handoff orbit preserves giver, receiver, and prop contact",
        },
        {
            "shot_id": "shot_final_dolly",
            "required_event_ids": [event["id"] for event in events[-2:]],
            "target_ids": [actor_ids[1], *prop_ids],
            "visibility_predicates": {target_id: "visible" for target_id in [actor_ids[1], *prop_ids]},
            "max_occlusion": 0.3,
            "innovation_intent_evidence": "final dolly reveals support and ownership state",
        },
    ]


def _make_prompt(actors: list[str], props: list[str], family: str, variant: int) -> str:
    a0, a1 = actors[0], actors[1]
    prop_text = ", ".join(props[:-1]) + (f", and {props[-1]}" if len(props) > 2 else f" and {props[-1]}")
    rhythm = CAMERA_RHYTHMS[variant % len(CAMERA_RHYTHMS)]
    detail = VARIANT_DETAILS[(variant - 1) % len(VARIANT_DETAILS)]
    additional_actor = (
        f" {actors[2]} observes the exchange from a protected third lane."
        if len(actors) > 2
        else ""
    )
    if family in {"concurrent_independent_work", "dev_three_actor_two_prop", "dev_three_actor_three_prop"}:
        action = f"{a0} carries the {props[0]} while {a1} carries the {props[1]}, then {a0} hands the {props[0]} to {a1} and {a1} places the {props[0]} without crossing lanes"
    elif family in {"occlusion_reveal", "dev_occlusion_countermotion", "test_counterfactual_camera_visibility"}:
        action = f"{a0} reveals the {props[0]}, then carries the {props[0]} and hands the {props[0]} to {a1}; {a1} places the {props[0]} while the {props[1]} remains visible"
    elif family in {"role_swap_pause_return_crossing", "dev_role_reversal", "test_role_reversal_final_owner"}:
        action = f"{a0} carries the {props[0]} and hands the {props[0]} to {a1}; {a1} pauses, returns the {props[0]} to {a0}, and {a0} places the {props[0]}"
    else:
        action = f"{a0} carries the {props[0]}, then {a0} hands the {props[0]} to {a1} and {a1} places the {props[0]}; then {a1} carries the {props[1]} and places the {props[1]} through a separate lane"
    return (
        f"{action}. The scene contains {prop_text}; use {rhythm}. {detail}. "
        f"{additional_actor} "
        f"Variant {variant:02d} of the {family} composition must preserve identity, event order, handoff timing, "
        "support contact, and readable camera coverage."
    )


def _make_case(split: str, family: str, variant: int) -> dict[str, Any]:
    actor_labels = list(ACTOR_SETS[(variant + len(family)) % len(ACTOR_SETS)])
    prop_labels = list(PROP_SETS[(variant * 2 + len(family)) % len(PROP_SETS)])
    if split == "train":
        actor_labels = list(ACTOR_SETS[variant % 3])
        prop_labels = list(PROP_SETS[variant % 4])
    actor_ids = _ids(actor_labels, prefix="actor")
    prop_ids = [label.replace(" ", "_") for label in prop_labels]
    events = _make_event_graph(actor_ids, prop_ids, family)
    case_id = f"multi-{split}-{variant:03d}"
    prompt = _make_prompt(actor_labels, prop_labels, family, variant)
    proxy_entities = [
        {"id": entity_id, "kind": "character", "role": "participant", "label": label}
        for entity_id, label in zip(actor_ids, actor_labels)
    ] + [
        {"id": entity_id, "kind": "prop", "role": "target_object", "label": label}
        for entity_id, label in zip(prop_ids, prop_labels)
    ] + [
        {"id": "support_surface", "kind": "support", "role": "environment", "label": "support surface"},
        {"id": "drop_zone", "kind": "support", "role": "environment", "label": "drop zone"},
    ]
    camera_evidence = _make_camera_evidence(actor_ids, prop_ids, events, family)
    negative_constraints = [
        "no_identity_swap",
        "no_prop_penetration",
        "no_unplanned_actor_crossing",
        "handoff_requires_same_window_detach_attach",
        "all_required_targets_visible_in_event_shot",
    ]
    if "occlusion" in family or "counterfactual" in family:
        negative_constraints.append("occlusion_must_end_before_grasp")
    difficulty = {"train": 4, "dev": 7, "test": 9}[split] + (variant % 2)
    record = {
        "case_id": case_id,
        "split": split,
        "template_family": family,
        "difficulty": difficulty,
        "duration_s": 20.0,
        "fps": 24,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "category": family,
        "composition_signature": f"{split}:{family}:{len(actor_ids)}a:{len(prop_ids)}p:v{variant:02d}",
        "entities": [
            {"id": entity_id, "kind": "actor", "role": "participant", "label": label}
            for entity_id, label in zip(actor_ids, actor_labels)
        ] + [
            {"id": entity_id, "kind": "prop", "role": "target_object", "label": label}
            for entity_id, label in zip(prop_ids, prop_labels)
        ],
        "required_events": [event["id"] for event in events],
        "event_graph": events,
        "interactions": _make_interactions(actor_ids, prop_ids),
        "camera_evidence": camera_evidence,
        "negative_constraints": negative_constraints,
        "oracle_expectations": {
            "required_entity_ids": [*actor_ids, *prop_ids],
            "event_order": [event["id"] for event in events],
            "required_camera_events": [event_id for shot in camera_evidence for event_id in shot["required_event_ids"]],
            "required_negative_constraints": negative_constraints,
        },
        "proxy_scene": {
            "scene_id": f"multi-proxy-scene-{split}-{variant:03d}",
            "scene_seed": 40000 + variant + (0 if split == "train" else 1000 if split == "dev" else 2000),
            "entities": proxy_entities,
            "layout": {
                "support_position": [0.0, 0.0, 0.0],
                "drop_zone_position": [3.5, 1.5, 0.0],
                "actor_start_positions": {
                    entity_id: [-3.0, (index - (len(actor_ids) - 1) / 2) * 2.0, 0.0]
                    for index, entity_id in enumerate(actor_ids)
                },
                "path_shape": "s_curve" if variant % 2 else "reverse_arc",
                "object_scale": [0.9, 0.9, 0.9],
                "support_scale": [3.2, 1.8, 0.5],
            },
            "camera": {"must_show_events": [event["id"] for event in events], "trajectory_types": ["follow", "orbit", "dolly"]},
            "geometry": {"detail_required": True, "occluder_count": 1 if "occlusion" in family or "counterfactual" in family else 0, "requires_opening": False},
            "material": "ProxyWhiteMaterial",
            "required_artifacts": ["proxy.blend", "proxy.mp4", "telemetry.json", "frames/index.json"],
        },
    }
    return record


def _families_for(split: str) -> tuple[str, ...]:
    return {"train": TRAIN_FAMILIES, "dev": DEV_FAMILIES, "test": TEST_FAMILIES}[split]


def build_dataset(output_root: str | Path = "dataset/trajectory-v4-multi") -> dict[str, Any]:
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    variant = 1
    for split in ("train", "dev", "test"):
        for family in _families_for(split):
            for _ in range(10):
                record = _make_case(split, family, variant)
                records.append(record)
                splits[split].append(record["case_id"])
                variant += 1
    records.sort(key=lambda item: item["case_id"])
    labels = [
        {
            "case_id": record["case_id"],
            "split": record["split"],
            "template_family": record["template_family"],
            "event_graph": record["event_graph"],
            "interactions": record["interactions"],
            "camera_evidence": record["camera_evidence"],
            "negative_constraints": record["negative_constraints"],
            "oracle_expectations": record["oracle_expectations"],
        }
        for record in records
    ]
    specs = [
        {"case_id": record["case_id"], "split": record["split"], "proxy_scene": record["proxy_scene"]}
        for record in records
    ]
    for name, payload in (("manifest.jsonl", records), ("labels.jsonl", labels), ("proxy_specs.jsonl", specs)):
        (destination / name).write_text("".join(_json_line(item) for item in payload), encoding="utf-8")
    split_payload = {split: sorted(values) for split, values in splits.items()}
    (destination / "splits.json").write_text(json.dumps(split_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "dataset_id": "trajectory-v4-multi",
        "schema_version": "trajectory-dataset-v4-multi",
        "generator_version": "director-multi-v1",
        "cases": len(records),
        "splits": {split: len(values) for split, values in split_payload.items()},
        "families": {split: list(_families_for(split)) for split in split_payload},
        "fingerprint": _fingerprint(records),
        "train_policy": "five authored ten-case families; repeated failures must affect two distinct cases before a patch",
        "dev_policy": "harder unseen actor/prop compositions; paired dev is never allowed to regress",
        "test_policy": "frozen role reversals, counterfactual camera constraints, prohibited crossings, and final-owner/support checks; never used for patch selection",
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dataset/trajectory-v4-multi")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
