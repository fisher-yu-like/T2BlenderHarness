"""Author assistant-session Director preauth for round-01 batch (20 cases).

The per-case semantics below (entities, camera cues, evidence claims) were
authored by the driving glm-5.3-flash session from each verbatim benchmark
prompt; this script only materializes them with programmatically verified
evidence spans (exact prompt substrings) and writes the preauth files.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "vbench2-agent-training-index-v1" / "manifest.jsonl"
SESSION = ROOT / "out" / "assistant-session" / "preauth" / "director"


def span(prompt: str, quote: str) -> tuple[int, int]:
    start = prompt.find(quote)
    assert start >= 0, f"{quote!r} not in prompt: {prompt!r}"
    return [start, start + len(quote)]


def camera_case(case_id: str, prompt: str, subject_label: str, visual_hint: str, cues: list[dict], extra_quotes: list[str] | None = None):
    """Camera-cue prompt: subject prop + support surface + observe + cues."""
    entities = [
        {
            "id": "prop_01_subject",
            "kind": "prop",
            "role": "scene_subject",
            "label": subject_label,
            "attributes": {"source_evidence_id": "ev_entity_subject", "visual_hint": visual_hint},
        },
        {
            "id": "support_surface",
            "kind": "support",
            "role": "environment",
            "label": "neutral support surface",
            "attributes": {"source_evidence_id": "ev_policy_support"},
        },
    ]
    evidence = [
        {
            "id": "ev_entity_subject",
            "source": "prompt",
            "prompt_span": span(prompt, subject_label.split(" ")[0] if subject_label == "Mount Fuji" else subject_label),
            "quoted_text": subject_label.split(" ")[0] if subject_label == "Mount Fuji" else subject_label,
            "claim": f"the scene subject is {subject_label}",
        }
    ]
    if subject_label == "Mount Fuji":
        evidence[0]["prompt_span"] = span(prompt, "Mount Fuji")
        evidence[0]["quoted_text"] = "Mount Fuji"
    for index, cue in enumerate(cues, start=1):
        cue_start, cue_end = span(prompt, cue["quote"])
        evidence.append(
            {
                "id": f"ev_camera_{index:02d}",
                "source": "prompt",
                "prompt_span": [cue_start, cue_end],
                "quoted_text": cue["quote"],
                "claim": cue["claim"],
            }
        )
    evidence.append(
        {
            "id": "ev_policy_support",
            "source": "policy",
            "prompt_span": None,
            "quoted_text": None,
            "claim": "a neutral support surface is used for grounded staging",
        }
    )
    directives = [
        {
            "id": "observe_01",
            "action": "observe",
            "prop_id": "prop_01_subject",
            "evidence_id": "ev_entity_subject",
        }
    ]
    camera_cues = [
        {
            "id": f"cue_{index:02d}",
            "action": cue["action"],
            "direction": cue.get("direction"),
            "evidence_id": f"ev_camera_{index:02d}",
        }
        for index, cue in enumerate(cues, start=1)
    ]
    return {
        "scene_id": case_id,
        "prompt": prompt,
        "authored_by": "glm-5.3-flash assistant session (Director LLM role)",
        "response": {
            "entities": entities,
            "directives": directives,
            "camera_cues": camera_cues,
            "evidence": evidence,
            "assumptions": [
                "The subject is staged as a centered diorama-like set piece so the requested camera move keeps it framed."
            ],
            "uncertainties": [
                {
                    "id": "unc_visual_style",
                    "description": "The benchmark prompt does not specify material, scale, or set dressing; a neutral visual prior is applied.",
                    "severity": "soft",
                    "resolved": False,
                }
            ],
        },
    }


CASES = [
    camera_case(
        "vbench2-train-01-01",
        "Garden, zoom out.",
        "Garden",
        "garden diorama with trees, hedges and a stone path",
        [{"quote": "zoom out", "action": "zoom", "direction": "out", "claim": "camera zoom out cue"}],
    ),
    camera_case(
        "vbench2-train-01-02",
        "The camera orbits around in a clockwise direction. Garden.",
        "Garden",
        "garden diorama with trees, hedges and a stone path",
        [
            {"quote": "orbits around", "action": "orbit", "direction": "clockwise", "claim": "camera orbits the subject"},
            {"quote": "clockwise", "action": "orbit", "direction": "clockwise", "claim": "orbit direction is clockwise"},
        ],
    ),
    camera_case(
        "vbench2-train-01-03",
        "Pyramid, tilt down.",
        "Pyramid",
        "stone pyramid with stepped faces on sandy ground",
        [{"quote": "tilt down", "action": "tilt", "direction": "down", "claim": "camera tilt down cue"}],
    ),
    camera_case(
        "vbench2-train-01-04",
        "Mount Fuji, zoom in.",
        "Mount Fuji",
        "broad volcanic cone with a snow-capped summit",
        [{"quote": "zoom in", "action": "zoom", "direction": "in", "claim": "camera zoom in cue"}],
    ),
    camera_case(
        "vbench2-train-01-05",
        "Mount Fuji, pan left.",
        "Mount Fuji",
        "broad volcanic cone with a snow-capped summit",
        [{"quote": "pan left", "action": "pan", "direction": "left", "claim": "camera pan left cue"}],
    ),
    camera_case(
        "vbench2-train-01-06",
        "Blue Lagoon, tilt up.",
        "Blue Lagoon",
        "turquoise lagoon water with a pale shore rim",
        [{"quote": "tilt up", "action": "tilt", "direction": "up", "claim": "camera tilt up cue"}],
    ),
    camera_case(
        "vbench2-train-01-07",
        "The camera movement is static. Blue Lagoon, static shot, the camera is fixed.",
        "Blue Lagoon",
        "turquoise lagoon water with a pale shore rim",
        [
            {"quote": "static", "action": "static", "direction": None, "claim": "the camera movement is static"},
            {"quote": "static shot", "action": "static", "direction": None, "claim": "static shot requested"},
            {"quote": "the camera is fixed", "action": "static", "direction": None, "claim": "the camera is fixed"},
        ],
    ),
    camera_case(
        "vbench2-train-01-08",
        "Table, pan left.",
        "Table",
        "wooden table with legs and set place settings",
        [{"quote": "pan left", "action": "pan", "direction": "left", "claim": "camera pan left cue"}],
    ),
    camera_case(
        "vbench2-train-01-09",
        "Alhambra, zoom out.",
        "Alhambra",
        "moorish palace facade with arched windows and crenellations",
        [{"quote": "zoom out", "action": "zoom", "direction": "out", "claim": "camera zoom out cue"}],
    ),
    camera_case(
        "vbench2-train-01-10",
        "Alhambra, pan right.",
        "Alhambra",
        "moorish palace facade with arched windows and crenellations",
        [{"quote": "pan right", "action": "pan", "direction": "right", "claim": "camera pan right cue"}],
    ),
    camera_case(
        "vbench2-dev-01-11",
        "Vase, tilt down.",
        "Vase",
        "ceramic vase with a narrow neck on a pedestal",
        [{"quote": "tilt down", "action": "tilt", "direction": "down", "claim": "camera tilt down cue"}],
    ),
    camera_case(
        "vbench2-dev-01-12",
        "The camera orbits around in a clockwise direction. Vase.",
        "Vase",
        "ceramic vase with a narrow neck on a pedestal",
        [
            {"quote": "orbits around", "action": "orbit", "direction": "clockwise", "claim": "camera orbits the subject"},
            {"quote": "clockwise", "action": "orbit", "direction": "clockwise", "claim": "orbit direction is clockwise"},
        ],
    ),
    camera_case(
        "vbench2-dev-01-13",
        "Burj Khalifa, pan left.",
        "Burj Khalifa",
        "ultra-tall tiered skyscraper with a spire",
        [{"quote": "pan left", "action": "pan", "direction": "left", "claim": "camera pan left cue"}],
    ),
    camera_case(
        "vbench2-dev-01-14",
        "Machu Picchu, tilt up.",
        "Machu Picchu",
        "terraced mountain citadel with stone tiers",
        [{"quote": "tilt up", "action": "tilt", "direction": "up", "claim": "camera tilt up cue"}],
    ),
    camera_case(
        "vbench2-dev-01-15",
        "The camera movement is static. Machu Picchu, static shot, the camera is fixed.",
        "Machu Picchu",
        "terraced mountain citadel with stone tiers",
        [
            {"quote": "static", "action": "static", "direction": None, "claim": "the camera movement is static"},
            {"quote": "static shot", "action": "static", "direction": None, "claim": "static shot requested"},
            {"quote": "the camera is fixed", "action": "static", "direction": None, "claim": "the camera is fixed"},
        ],
    ),
    camera_case(
        "vbench2-dev-01-16",
        "Forbidden City, pan left.",
        "Forbidden City",
        "imperial palace hall with tiered roofs and red walls",
        [{"quote": "pan left", "action": "pan", "direction": "left", "claim": "camera pan left cue"}],
    ),
    camera_case(
        "vbench2-dev-01-17",
        "Forbidden City, First-person perspective, oblique shot, airborne dolly movement.",
        "Forbidden City",
        "imperial palace hall with tiered roofs and red walls",
        [
            {"quote": "First-person perspective", "action": "follow", "direction": "first_person", "claim": "first-person perspective camera"},
            {"quote": "airborne dolly movement", "action": "dolly", "direction": "airborne", "claim": "airborne dolly movement cue"},
        ],
    ),
    camera_case(
        "vbench2-dev-01-18",
        "Laptop, pan right.",
        "Laptop",
        "open laptop with keyboard deck and glowing screen",
        [{"quote": "pan right", "action": "pan", "direction": "right", "claim": "camera pan right cue"}],
    ),
    camera_case(
        "vbench2-dev-01-19",
        "Watch, zoom out.",
        "Watch",
        "wristwatch with a round dial and strap, standing upright",
        [{"quote": "zoom out", "action": "zoom", "direction": "out", "claim": "camera zoom out cue"}],
    ),
    camera_case(
        "vbench2-dev-01-20",
        "The camera orbits around in a clockwise direction. Watch.",
        "Watch",
        "wristwatch with a round dial and strap, standing upright",
        [
            {"quote": "orbits around", "action": "orbit", "direction": "clockwise", "claim": "camera orbits the subject"},
            {"quote": "clockwise", "action": "orbit", "direction": "clockwise", "claim": "orbit direction is clockwise"},
        ],
    ),
]


def main() -> int:
    records = {}
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[record["case_id"]] = record
    SESSION.mkdir(parents=True, exist_ok=True)
    written = []
    for spec in CASES:
        record = records[spec["scene_id"]]
        assert spec["prompt"] == record["prompt"], f"prompt drift for {spec['scene_id']}"
        # Validate every prompt-span claim exactly as the contract will.
        for item in spec["response"]["evidence"]:
            if item.get("prompt_span") is not None:
                start, end = item["prompt_span"]
                observed = spec["prompt"][start:end]
                assert observed.casefold() == (item.get("quoted_text") or "").casefold(), (
                    f"{spec['scene_id']}: span mismatch {observed!r} != {item.get('quoted_text')!r}"
                )
        target = SESSION / f"{spec['scene_id']}.json"
        target.write_text(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(spec["scene_id"])
    print(json.dumps({"written": len(written), "cases": written}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
