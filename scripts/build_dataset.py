"""Generate the checked-in deterministic 40-case benchmark records."""

from __future__ import annotations

import json
from pathlib import Path


CATEGORIES = [
    ("single_static", "Observe a still {object} on a plain support."),
    ("walk_to_target", "A character walks to the {object} on the table."),
    ("pickup_release", "A character picks up the {object} from the table and places it down."),
    ("contact_attachment", "A character grasps the {object} on the table and keeps it attached."),
    ("follow_orbit", "The camera follows a character orbiting the {object} on the table."),
    ("multi_entity", "A character moves between the {object} on the table and a second prop."),
    ("occlusion_visibility", "The character reveals the {object} on the table after an occlusion."),
    ("physics_support", "The {object} stays supported before the character grasps it."),
    ("event_camera", "Show the character reaching for the {object} on the table in a closeup."),
    ("long_composition", "A long shot shows the character walking, reaching, and grasping the {object} on the table."),
]
OBJECTS = ["red cup", "blue cube", "green ball", "yellow book"]


def build(root: Path = Path("dataset")) -> None:
    root.mkdir(parents=True, exist_ok=True)
    records = []
    splits = {"train": [], "dev": [], "test": []}
    labels = []
    for category_index, (category, template) in enumerate(CATEGORIES):
        for variant in range(4):
            case_number = category_index * 4 + variant + 1
            case_id = f"case-{case_number:02d}"
            object_name = OBJECTS[variant]
            prompt = template.format(object=object_name)
            object_id = object_name.replace(" ", "_")
            required_events = ["observe"] if category == "single_static" else ["walk", "reach", "grasp"]
            if category == "walk_to_target":
                required_events = ["walk"]
            record = {
                "case_id": case_id,
                "prompt": prompt,
                "category": category,
                "entities": [
                    {"id": "character", "kind": "character", "role": "actor"},
                    {"id": "table", "kind": "support", "role": "environment"},
                    {"id": object_id, "kind": "prop", "role": "target_object"},
                ],
                "required_events": required_events,
                "expected_relations": [{"type": "on", "subject": object_id, "object": "table"}],
                "duration_s": 10.0 if category != "long_composition" else 20.0,
                "fps": 24,
                "evaluator_version": "deterministic-v1",
            }
            records.append(record)
            split = "train" if variant < 2 else "dev" if variant == 2 else "test"
            splits[split].append(case_id)
            if variant == 0:
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
    (root / "manifest.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "splits.json").write_text(json.dumps(splits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "labels.jsonl").write_text(
        "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels),
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()
