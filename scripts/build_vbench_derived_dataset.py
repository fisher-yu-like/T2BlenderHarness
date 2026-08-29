"""Build a reproducible VBench-2.0-derived paired Harness benchmark.

The source prompts are preserved verbatim as provenance.  The executable
prompt is deliberately a derived prompt: it keeps the VBench seed as scene
context, then adds two named actors, two proxy objects, ordered events, and
camera/trajectory constraints that the current DirectorAgent can compile.
This benchmark is for comparison only; it is not used for Harness patch
selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "VBench-2.0/vbench2/VBench2_full_info.json"
)
SOURCE_FILE_NAME = "VBench2_full_info.json"
SOURCE_DIMENSIONS = (
    ("Camera_Motion", "camera_motion"),
    ("Human_Interaction", "human_interaction"),
    ("Motion_Order_Understanding", "motion_order"),
    ("Complex_Plot", "complex_plot"),
    ("Dynamic_Spatial_Relationship", "dynamic_spatial_relationship"),
)
ACTOR_PAIRS = (("Alice", "Bob"), ("Alice", "Carla"), ("Bob", "Dana"), ("Alice", "Dana"))
PROP_PAIRS = (("red cube", "blue cup"), ("green ball", "yellow book"), ("red cup", "blue cube"), ("green book", "yellow ball"))
CAMERA_RHYTHMS = (
    "a restrained push-in during the approach, a readable two-shot orbit at contact, and a slow final dolly",
    "a wide establishing hold, a lateral follow through the handoff, and a centered support reveal",
    "a low tracking move for the first transfer, a short reverse arc for ownership change, and a stable finish",
    "a gentle zoom toward the active object, a side-on follow that keeps both actors visible, and a held final frame",
    "a static opening composition, a motivated pan between lanes, and a quiet pull-back after placement",
)
ACTION_VARIANTS = ("direct_transfer", "reveal_elliptical_return", "subjectless_handoff_return", "parallel_transfer")


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(_json_line(record) for record in sorted(records, key=lambda item: item["case_id"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_source(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("VBench2 full info must be a JSON list")
    return payload


def _pick_spread(items: list[tuple[int, dict[str, Any]]], count: int) -> list[tuple[int, dict[str, Any]]]:
    if len(items) < count:
        raise ValueError(f"not enough VBench source prompts: need {count}, got {len(items)}")
    if len(items) == count:
        return items
    positions = [round(index * (len(items) - 1) / (count - 1)) for index in range(count)]
    return [items[position] for position in positions]


def _context_safe(text: str) -> str:
    # Source text remains untouched in source_prompt.  Only the executable
    # context is sanitized so VBench nouns cannot accidentally become proxy
    # entities under the current deterministic interpreter.
    text = re.sub(r"\b(?:Alice|Bob|Carla|Dana)\b", "a named performer", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:red|blue|green|yellow)\s+(?:cube|cup|book|ball)\b", "a proxy object", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _action_text(variant: str, actors: tuple[str, str], props: tuple[str, str]) -> str:
    a0, a1 = actors
    p0, p1 = props
    if variant == "reveal_elliptical_return":
        return (
            f"{a0} reveals the {p0}, then carries the {p0} to {a1}; hands the {p0} to {a1}. "
            f"{a1} pauses; returns the {p0} to {a0}, and {a0} places the {p0}. "
            f"After that, {a1} places the already-visible {p1} in the drop zone."
        )
    if variant == "subjectless_handoff_return":
        return (
            f"{a0} carries the {p0} to {a1}; then hands the {p0} to {a1}. "
            f"{a1} pauses; returns the {p0} to {a0}; {a0} places the {p0}. "
            f"Meanwhile, {a1} carries the {p1} to the drop zone and places the {p1}."
        )
    if variant == "parallel_transfer":
        return (
            f"{a0} carries the {p0} while {a1} carries the {p1}; then {a0} hands the {p0} to {a1}. "
            f"{a1} places the {p0} and later places the {p1} in the drop zone."
        )
    return (
        f"{a0} carries the {p0} to {a1}, then {a0} hands the {p0} to {a1}; "
        f"{a1} places the {p0}. Separately, {a1} carries the {p1} to the drop zone and places the {p1}."
    )


def _event_graph(variant: str, actors: tuple[str, str], props: tuple[str, str]) -> list[dict[str, Any]]:
    a0, a1 = actors
    p0, p1 = props
    events: list[dict[str, Any]] = []

    def add(event_id: str, action: str, participant_ids: list[str], target_ids: list[str], start: float, end: float, depends_on: list[str] | None = None, concurrency_group: str | None = None) -> None:
        event: dict[str, Any] = {
            "id": event_id,
            "action": action,
            "participant_ids": participant_ids,
            "target_ids": target_ids,
            "depends_on": depends_on or [],
            "start": start,
            "end": end,
        }
        if concurrency_group:
            event["concurrency_group"] = concurrency_group
        events.append(event)

    previous = ""
    if variant == "reveal_elliptical_return":
        add("reveal_01", "reveal", [a0], [p0], 0.0, 1.0)
        previous = "reveal_01"
    add("carry_01", "carry", [a0], [p0], 1.0, 3.0, [previous] if previous else [])
    add("handoff_01", "handoff", [a0, a1], [p0], 3.0, 4.2, ["carry_01"])
    if variant in {"reveal_elliptical_return", "subjectless_handoff_return"}:
        add("pause_01", "pause", [a1], [p0], 4.2, 5.0, ["handoff_01"])
        add("return_01", "return", [a1], [p0], 5.0, 6.5, ["pause_01"])
        add("place_01", "place", [a0], [p0], 6.5, 7.8, ["return_01"])
    else:
        add("place_01", "place", [a1], [p0], 4.2, 5.5, ["handoff_01"])
    if variant == "parallel_transfer":
        add("carry_02", "carry", [a1], [p1], 1.0, 3.0, [], "parallel_01")
        add("place_02", "place", [a1], [p1], 7.8, 9.2, ["carry_02"])
    elif variant == "reveal_elliptical_return":
        add("place_02", "place", [a1], [p1], 8.2, 9.6, ["place_01"])
    else:
        add("carry_02", "carry", [a1], [p1], 7.8, 9.4, ["place_01"])
        add("place_02", "place", [a1], [p1], 9.4, 10.8, ["carry_02"])
    if variant == "parallel_transfer":
        events[0]["concurrency_group"] = "parallel_01"
        events[0]["depends_on"] = []
    for event in events:
        event["start"] = round(float(event["start"]) * 0.5, 4)
        event["end"] = round(float(event["end"]) * 0.5, 4)
    return events


def _record(
    *,
    split: str,
    category: str,
    source_dimension: str,
    source_index: int,
    source_item: dict[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    actors = ACTOR_PAIRS[ordinal % len(ACTOR_PAIRS)]
    props = PROP_PAIRS[ordinal % len(PROP_PAIRS)]
    variant = ACTION_VARIANTS[ordinal % len(ACTION_VARIANTS)]
    case_id = f"vbench2-{category}-{ordinal + 1:03d}"
    source_prompt = str(source_item.get("prompt_en", "")).strip()
    source_context = _context_safe(source_prompt)
    auxiliary = source_item.get("auxiliary_info")
    auxiliary_text = ", ".join(map(str, auxiliary)) if isinstance(auxiliary, list) else str(auxiliary or "unspecified")
    camera_rhythm = CAMERA_RHYTHMS[ordinal % len(CAMERA_RHYTHMS)]
    action = _action_text(variant, actors, props)
    prompt = (
        f"VBench-2.0 seed context (do not add unnamed entities): {source_context}. "
        f"The source dimension is {source_dimension} and its camera/action cue is {auxiliary_text}. "
        f"Use exactly two named performers, {actors[0]} and {actors[1]}, and exactly two proxy objects, "
        f"{props[0]} and {props[1]}. {action} "
        f"Schedule the events in that order over a 6-second continuous shot: anticipate before contact, "
        f"hold the ownership change long enough to read it, and keep the final support state visible. "
        f"Use {camera_rhythm}; every active actor and object must remain identifiable, with no identity swap, "
        "unplanned lane crossing, interpenetration, or camera cut that hides the transfer."
    )
    event_graph = _event_graph(variant, actors, props)
    actor_ids = ["actor_a", "actor_b"]
    prop_ids = [props[0].replace(" ", "_"), props[1].replace(" ", "_")]
    proxy_entities = [
        {"id": "actor_a", "kind": "character", "role": "participant", "label": actors[0]},
        {"id": "actor_b", "kind": "character", "role": "participant", "label": actors[1]},
        {"id": prop_ids[0], "kind": "prop", "role": "target_object", "label": props[0]},
        {"id": prop_ids[1], "kind": "prop", "role": "target_object", "label": props[1]},
        {"id": "support_surface", "kind": "support", "role": "environment", "label": "support surface"},
        {"id": "drop_zone", "kind": "support", "role": "environment", "label": "drop zone"},
    ]
    camera_events = [event["id"] for event in event_graph]
    seed = 60000 + source_index + ordinal * 17
    proxy_scene = {
        "scene_id": f"vbench2-proxy-scene-{ordinal + 1:03d}",
        "scene_seed": seed,
        "entities": proxy_entities,
        "layout": {
            "support_position": [0.0, 0.0, 0.0],
            "drop_zone_position": [3.5, 1.5, 0.0],
            "actor_start_positions": {"actor_a": [-3.0, -1.0, 0.0], "actor_b": [-3.0, 1.0, 0.0]},
            "path_shape": ("reverse_arc", "s_curve", "bezier")[ordinal % 3],
            "object_scale": [0.9, 0.9, 0.9],
            "support_scale": [3.2, 1.8, 0.5],
        },
        "camera": {"must_show_events": camera_events, "trajectory_types": ["follow", "orbit", "dolly"]},
        "geometry": {"detail_required": True, "occluder_count": 0, "requires_opening": False},
        "material": "ProxyWhiteMaterial",
        "required_artifacts": ["proxy.blend", "proxy.mp4", "telemetry.json", "frames/index.json"],
    }
    negative_constraints = [
        "no_identity_swap",
        "no_prop_penetration",
        "no_unplanned_actor_crossing",
        "handoff_requires_same_window_detach_attach",
        "all_required_targets_visible_in_event_shot",
    ]
    return {
        "case_id": case_id,
        "split": split,
        "category": category,
        "template_family": f"vbench2_{category}_{variant}",
        "difficulty": 7 + ordinal % 4,
        "duration_s": 6.0,
        "fps": 12,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "source_dataset": "VBench-2.0",
        "source_prompt": source_prompt,
        "source_dimension": source_dimension,
        "source_index": source_index,
        "source_auxiliary_info": auxiliary,
        "adaptation": "VBench seed retained as context; deterministic two-actor/two-proxy trajectory contract appended for Blender compilation.",
        "entities": [
            {"id": "actor_a", "kind": "actor", "role": "participant", "label": actors[0]},
            {"id": "actor_b", "kind": "actor", "role": "participant", "label": actors[1]},
            {"id": prop_ids[0], "kind": "prop", "role": "target_object", "label": props[0]},
            {"id": prop_ids[1], "kind": "prop", "role": "target_object", "label": props[1]},
        ],
        "required_events": camera_events,
        "event_graph": event_graph,
        "interactions": [
            {
                "id": f"attachment_lifecycle:actor_b:{prop_ids[0]}:handoff_01",
                "prop_id": prop_ids[0],
                "giver_id": "actor_a",
                "receiver_id": "actor_b",
                "attach_event_id": "carry_01",
                "transfer_event_id": "handoff_01",
                "detach_event_id": "place_01",
                "final_owner_id": "actor_b" if variant not in {"reveal_elliptical_return", "subjectless_handoff_return"} else "actor_a",
                "final_support_id": "support_surface",
            }
        ],
        "camera_evidence": [
            {
                "shot_id": "shot_establish",
                "required_event_ids": camera_events[:2],
                "target_ids": [*actor_ids, *prop_ids],
                "visibility_predicates": {target_id: "visible" for target_id in [*actor_ids, *prop_ids]},
                "max_occlusion": 0.3,
                "innovation_intent_evidence": "wide context establishes the seed scene and both trajectory lanes",
            },
            {
                "shot_id": "shot_transfer",
                "required_event_ids": [event_id for event_id in camera_events if "handoff" in event_id or "return" in event_id],
                "target_ids": ["actor_a", "actor_b", prop_ids[0]],
                "visibility_predicates": {target_id: "visible" for target_id in ["actor_a", "actor_b", prop_ids[0]]},
                "max_occlusion": 0.25,
                "innovation_intent_evidence": "two-shot orbit makes ownership change and hand trajectory legible",
            },
            {
                "shot_id": "shot_final_support",
                "required_event_ids": camera_events[-2:],
                "target_ids": ["actor_b", *prop_ids],
                "visibility_predicates": {target_id: "visible" for target_id in ["actor_b", *prop_ids]},
                "max_occlusion": 0.3,
                "innovation_intent_evidence": "final dolly verifies support contact and object identity",
            },
        ],
        "negative_constraints": negative_constraints,
        "oracle_expectations": {
            "required_entity_ids": [*actor_ids, *prop_ids],
            "event_order": camera_events,
            "required_camera_events": camera_events,
            "required_negative_constraints": negative_constraints,
        },
        "proxy_scene": proxy_scene,
    }


def build_dataset(source_path: str | Path, output_root: str | Path) -> dict[str, Any]:
    source_file = Path(source_path)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    source_items = _load_source(source_file)
    records: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    ordinal = 0
    for source_dimension, category in SOURCE_DIMENSIONS:
        candidates = [(index, item) for index, item in enumerate(source_items) if source_dimension in item.get("dimension", [])]
        for local_index, (source_index, source_item) in enumerate(_pick_spread(candidates, 20)):
            split = "train" if local_index < 14 else "dev"
            record = _record(
                split=split,
                category=category,
                source_dimension=source_dimension,
                source_index=source_index,
                source_item=source_item,
                ordinal=ordinal,
            )
            records.append(record)
            selected.append({"case_id": record["case_id"], "source_index": source_index, "source_dimension": source_dimension})
            ordinal += 1
    splits = {
        "train": [record["case_id"] for record in records if record["split"] == "train"],
        "dev": [record["case_id"] for record in records if record["split"] == "dev"],
        "test": [],
    }
    labels = [
        {
            "case_id": record["case_id"],
            "split": record["split"],
            "category": record["category"],
            "source_prompt": record["source_prompt"],
            "source_dimension": record["source_dimension"],
            "source_index": record["source_index"],
            "event_graph": record["event_graph"],
            "interactions": record["interactions"],
            "camera_evidence": record["camera_evidence"],
            "negative_constraints": record["negative_constraints"],
            "oracle_expectations": record["oracle_expectations"],
        }
        for record in records
    ]
    specs = [{"case_id": record["case_id"], "split": record["split"], "proxy_scene": record["proxy_scene"]} for record in records]
    for name, payload in (("manifest.jsonl", records), ("labels.jsonl", labels), ("proxy_specs.jsonl", specs)):
        (destination / name).write_text("".join(_json_line(item) for item in payload), encoding="utf-8")
    (destination / "splits.json").write_text(json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "dataset_id": "vbench-derived-100-v1",
        "schema_version": "trajectory-dataset-v4-multi",
        "generator_version": "vbench-derived-builder-v1",
        "source_dataset": "VBench-2.0",
        "source_url": SOURCE_URL,
        "source_file": SOURCE_FILE_NAME,
        "source_sha256": _sha256(source_file),
        "cases": len(records),
        "splits": {split: len(ids) for split, ids in splits.items()},
        "category_counts": {category: sum(record["category"] == category for record in records) for _, category in SOURCE_DIMENSIONS},
        "split_policy": "Each source dimension contributes 14 train and 6 held-out dev prompts; dev is never used for Harness patch selection.",
        "adaptation_policy": "Source prompt is preserved verbatim in source_prompt; executable prompt adds a fixed two-actor/two-proxy contract so trajectories and camera choreography are observable by this Blender Harness.",
        "comparison_policy": "The exact same 100 records, render settings, Blender binary, evaluator version, and sampled-frame policy are used for pretraining baseline and current Harness.",
        "selected_sources": selected,
        "fingerprint": _fingerprint(records),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.source, args.out), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
