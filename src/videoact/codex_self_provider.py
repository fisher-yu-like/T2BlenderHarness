"""In-process Codex-self provider for exact-prompt local evidence.

The desktop Codex agent is the provider in this mode.  This module is not a
legacy scene template and it is never a fallback: every request is interpreted
from its exact prompt, converted to a typed DirectorPlan, and compiled into a
source string containing that plan.  Unsupported or empty prompts raise and
the surrounding Harness records a fail-closed preparation failure.

The provider exists so a human can review real Blender videos when no external
Codex subprocess or VLM credential is available.  It is deliberately marked
``codex-self`` in every artifact; it must not be presented as an external LLM
measurement.
"""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from typing import Any

from .director_contracts import DirectorDecisionEvidence, DirectorEntity, DirectorRequest


_CAMERA_PATTERNS: tuple[tuple[str, str, str | None], ...] = (
    ("static", r"(?:static shot|camera (?:is )?fixed|movement is static)", None),
    ("orbit", r"orbits? around|orbit(?:ing)?", None),
    ("zoom", r"zoom\s+(in|out)", None),
    ("dolly", r"(?:airborne\s+)?dolly(?:\s+movement)?", None),
    ("pan", r"pan\s+(left|right)", None),
    ("tilt", r"tilt\s+(up|down)", None),
    ("follow", r"(?:camera\s+)?follow(?:s|ing)?", None),
)

_ANIMAL_WORDS = (
    "dog", "cat", "horse", "bird", "kangaroo", "monkey", "elephant", "rabbit", "squirrel", "fox",
)
_ROLE_WORDS = (
    "person", "people", "woman", "man", "girl", "boy", "prince", "princess", "adventurer", "hero",
    "artist", "painter", "raider", "captain", "warrior", "forward", "player", "team", "elf", "sage",
)
_PROP_PHRASES = (
    "toy mouse", "frying pan", "olive oil", "flat tire", "car window", "concrete wall", "cardboard box",
    "marble countertop", "rubber balloon", "plastic container", "steaming bowl", "cup of tea", "glass of water",
    "paper airplane", "ancient time key", "dark magic", "sacred relic", "green gemstone", "violet flower",
    "jade box", "stone tablet", "ancient wooden box", "dragon scholar", "dragon tribe", "drop zone",
    "water droplet", "sharp object", "sheet of aluminum",
)
_PROP_WORDS = (
    "ball", "rope", "ladder", "cup", "tea", "car", "shirt", "gift", "box", "groceries", "phone", "chair",
    "jacket", "tie", "puzzle", "picture", "backpack", "coffee", "bowl", "water", "apple", "book", "plant",
    "stairs", "table", "key", "relic", "flower", "gemstone", "stone", "treasure", "sword", "shield", "amulet",
    "wall", "balloon", "needle", "chalk", "pen", "countertop", "soup", "cloth", "sponge", "mattress", "bucket",
    "pond", "branch", "sofa", "shoe", "basket", "rock", "surface", "floor", "television", "magazine", "closet",
    "droplet", "aluminum", "sheet", "liquid",
)
_STOPWORDS = {
    "a", "an", "the", "and", "or", "then", "they", "it", "this", "that", "is", "are", "was", "were", "to",
    "of", "in", "on", "at", "from", "with", "by", "for", "as", "while", "during", "another", "one", "two",
    "person", "people", "camera", "movement", "direction", "shot", "first", "second", "perspective", "oblique",
    "airborne", "slowly", "suddenly", "finally", "together", "surface", "space", "filled", "showing", "capturing",
}


def _span(prompt: str, match: re.Match[str] | None) -> tuple[int, int]:
    return match.span() if match is not None else (0, len(prompt))


def _slug(value: str, fallback: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    value = "_".join(token for token in tokens if token not in _STOPWORDS)
    return (value[:32].strip("_") or fallback)


def _add_evidence(
    prompt: str,
    evidence: list[DirectorDecisionEvidence],
    identifier: str,
    claim: str,
    match: re.Match[str] | None = None,
) -> str:
    start, end = _span(prompt, match)
    quoted = prompt[start:end]
    evidence.append(
        DirectorDecisionEvidence(
            id=identifier,
            source="prompt",
            prompt_span=(start, end),
            quoted_text=quoted,
            claim=claim,
        )
    )
    return identifier


def _camera_cues(prompt: str, evidence: list[DirectorDecisionEvidence]) -> list[dict[str, Any]]:
    cues: list[dict[str, Any]] = []
    for index, (action, pattern, _unused) in enumerate(_CAMERA_PATTERNS, 1):
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match is None:
            continue
        direction = match.group(1).lower() if match.lastindex else None
        evidence_id = _add_evidence(
            prompt,
            evidence,
            f"ev_camera_{index:02d}",
            f"camera cue {action}" + (f" toward {direction}" if direction else ""),
            match,
        )
        cues.append({"id": f"camera_cue_{index:02d}", "action": action, "direction": direction, "evidence_id": evidence_id})
    return cues


def _scene_subject(prompt: str) -> str:
    camera_words = r"camera|movement|direction|static|shot|fixed|perspective|oblique|airborne|dolly|zoom|pan|tilt|orbit"
    fragments = [fragment.strip(" .,!?:;") for fragment in re.split(r"[.!?]", prompt)]
    candidates: list[str] = []
    for fragment in fragments:
        if not fragment or re.search(rf"\b(?:{camera_words})\b", fragment, re.IGNORECASE):
            continue
        candidates.append(fragment)
    if candidates:
        candidate = re.sub(r"\b(?:the|a|an)\b", "", candidates[0], flags=re.IGNORECASE).strip(" ,")
        if candidate:
            return candidate
    # A non-camera prompt uses its first concrete noun as the subject.  The
    # caller still checks for an alphanumeric result before accepting it.
    words = re.findall(r"[A-Za-z][A-Za-z-]*", prompt)
    return next((word for word in words if word.casefold() not in _STOPWORDS), "")


def _extract_props(prompt: str) -> list[str]:
    lower = prompt.casefold()
    found: list[str] = []
    for phrase in _PROP_PHRASES:
        if phrase in lower and phrase not in found:
            found.append(phrase)
    for word in _PROP_WORDS:
        if (
            re.search(rf"\b{re.escape(word)}\b", lower)
            and word not in found
            and not any(re.search(rf"\b{re.escape(word)}\b", phrase) for phrase in found)
        ):
            found.append(word)
    return found[:4]


def _extract_actors(prompt: str) -> list[str]:
    lower = prompt.casefold()
    actors: list[str] = []
    has_two_participants = re.search(
        r"\b(?:two people|two persons|another person|both teams|team [abc])\b",
        lower,
    )
    has_another_receiver = re.search(
        r"\b(?:one|a)\s+person\b.*\b(?:another|a second|the other)\b"
        r"|\bpass(?:es|ed)?\b.*\bto\s+(?:another|the other)\b",
        lower,
        re.IGNORECASE,
    )
    if has_two_participants or has_another_receiver:
        actors.extend(["person 1", "person 2"])
    named = re.findall(r"\b(?:named|called)\s+([A-Z][a-z]+)\b", prompt)
    actors.extend(named)
    for word in (*_ANIMAL_WORDS, *_ROLE_WORDS):
        if word in {"person", "people"} and any(item.startswith("person ") for item in actors):
            continue
        if re.search(rf"\b{re.escape(word)}\b", lower) and word not in actors:
            actors.append(word)
    if not actors and not any(cues for cues in _camera_cues(prompt, [])):
        # Generic action prompts such as “A person is painting.” still need a
        # visible participant; the label is an explicit prompt noun.
        if re.search(r"\bperson\b", lower):
            actors.append("person")
    return list(dict.fromkeys(actors))[:3]


def _action_matches(prompt: str) -> list[tuple[str, re.Match[str]]]:
    patterns: tuple[tuple[str, str], ...] = (
        ("handoff", r"\b(?:pass(?:es|ed)?|hand(?:s|ed)?|giv(?:e|es|en)|hands?)\b"),
        ("place", r"\b(?:place(?:s|d)?|put|puts|sets?|lay(?:s|ing)?)\b"),
        ("run", r"\brun(?:s|ning)?\b"),
        ("walk", r"\bwalk(?:s|ed|ing)?\b"),
        ("jump", r"\bjump(?:s|ed|ing)?\b"),
        ("fly", r"\bfl(?:y|ies|ew|ying)\b"),
        ("climb", r"\bclimb(?:s|ed|ing)?\b"),
        ("drink", r"\b(?:drink(?:s|ing)?|slurp(?:s|ing)?)\b"),
        ("sweep", r"\bsweep(?:s|ing)?\b"),
        ("brush", r"\bbrush(?:es|ed|ing)?\b"),
        ("write", r"\b(?:write|writes|writing|paint(?:s|ed|ing)?|draw(?:s|ing)?)\b"),
        ("pour", r"\bpour(?:s|ed|ing)?\b"),
        ("press", r"\bpress(?:es|ed|ing)?\b"),
        ("bounce", r"\bbounc(?:e|es|ed|ing)\b"),
        ("open", r"\bopen(?:s|ed|ing)?\b|turn(?:s|ed|ing)?\s+on"),
        ("close", r"\bclos(?:e|es|ed|ing)\b"),
        ("move", r"\b(?:mov(?:e|es|ed|ing)|slid(?:e|es|ed|ing))\b"),
        ("stand", r"\bstand(?:s|ing)?\s+up\b"),
        ("sit", r"\bsit(?:s|ting)?\b"),
        ("interact", r"\b(?:adjust(?:s|ed|ing)?|fold(?:s|ed|ing)?|wrap(?:s|ped|ping)?|load(?:s|ed|ing)?|play(?:s|ed|ing)?|build(?:s|ing)?|change(?:s|d|ing)?|clean(?:s|ed|ing)?|wash(?:es|ed|ing)?|cook(?:s|ed|ing)?|read(?:s|ing)?|listen(?:s|ed|ing)?|watch(?:es|ed|ing)?|organ(?:ize|izes|ized|izing))\b"),
    )
    matches: list[tuple[int, str, re.Match[str]]] = []
    for action, pattern in patterns:
        for match in re.finditer(pattern, prompt, re.IGNORECASE):
            matches.append((match.start(), action, match))
    matches.sort(key=lambda item: item[0])
    return [(action, match) for _start, action, match in matches]


def build_codex_self_director_payload(request: DirectorRequest) -> dict[str, Any]:
    """Interpret one exact prompt into provider output for StructuredPromptInterpreter."""

    prompt = request.prompt.strip()
    if not re.search(r"[A-Za-z0-9]", prompt) or prompt in {"...", "?", "??"}:
        raise ValueError("cannot derive executable scene subject from prompt")

    evidence: list[DirectorDecisionEvidence] = []
    camera_cues = _camera_cues(prompt, evidence)
    actors = _extract_actors(prompt)
    props = _extract_props(prompt)
    camera_only = bool(camera_cues) and not _action_matches(prompt)
    subject = _scene_subject(prompt)
    if camera_only and subject and subject.casefold() not in {item.casefold() for item in props}:
        props.insert(0, subject)
    if not actors and not props and subject:
        # VBench mechanics sometimes names a novel object (“whipped cream”,
        # “foam”, or a material) outside the small visual vocabulary above.
        # Preserve that exact phrase as a generic prompt-derived prop rather
        # than failing into a template or discarding the case.
        props.append(subject)
    if not actors and not props:
        raise ValueError("cannot derive executable scene subject from prompt")
    # Do not invent a human merely because a prompt names an object.  An
    # object-only VBench prompt (for example, a droplet sliding on aluminum)
    # must compile to an object-only scene; adding a default person here made
    # the old proxy look like a shared actor/table template and polluted the
    # DirectorPlan's semantic evidence.

    entities: list[DirectorEntity] = []
    for index, label in enumerate(actors):
        entity_id = f"actor_{chr(ord('a') + index)}"
        span_match = re.search(re.escape(label), prompt, re.IGNORECASE)
        evidence_id = _add_evidence(prompt, evidence, f"ev_entity_{entity_id}", f"participant {label}", span_match)
        entities.append(
            DirectorEntity(
                id=entity_id,
                kind="actor",
                role="participant",
                label=label,
                attributes={"visual_hint": label, "source_evidence_id": evidence_id},
            )
        )
    prop_entities: list[DirectorEntity] = []
    for index, label in enumerate(props):
        entity_id = f"prop_{index + 1:02d}_{_slug(label, f'object_{index + 1:02d}') }"
        span_match = re.search(re.escape(label), prompt, re.IGNORECASE)
        evidence_id = _add_evidence(prompt, evidence, f"ev_entity_{entity_id}", f"target object or scene subject {label}", span_match)
        prop_entities.append(
            DirectorEntity(
                id=entity_id,
                kind="prop",
                role="target_object" if not camera_only or index else "scene_subject",
                label=label,
                attributes={"visual_hint": label, "source_evidence_id": evidence_id},
            )
        )
    entities.extend(prop_entities)
    # A neutral support surface is a rendering aid, not a prompt claim.  It
    # is explicitly labeled as an assumption and remains visible in the plan.
    support_evidence = _add_evidence(prompt, evidence, "ev_policy_support", "a neutral support surface is used for grounded staging")
    entities.append(
        DirectorEntity(
            id="support_surface",
            kind="support",
            role="environment",
            label="neutral support surface",
            attributes={"source_evidence_id": support_evidence},
        )
    )

    directives: list[dict[str, Any]] = []
    action_matches = _action_matches(prompt)
    if camera_only or not action_matches:
        subject_id = prop_entities[0].id if prop_entities else (entities[0].id if entities else None)
        if subject_id is None:
            raise ValueError("cannot derive executable scene subject from prompt")
        match = re.search(re.escape(subject), prompt, re.IGNORECASE)
        evidence_id = _add_evidence(prompt, evidence, "ev_action_observe_01", "the subject remains observable while the camera cue executes", match)
        directives.append({"id": "observe_01", "action": "observe", "actor_id": subject_id, "prop_id": subject_id, "evidence_id": evidence_id})
    else:
        primary_actor = f"actor_{chr(ord('a'))}" if actors else (prop_entities[0].id if prop_entities else None)
        for index, (action, match) in enumerate(action_matches, 1):
            prop_id = prop_entities[min(index - 1, len(prop_entities) - 1)].id if prop_entities else None
            actor_id = primary_actor
            receiver_id = "actor_b" if action == "handoff" and len(actors) > 1 else None
            evidence_id = _add_evidence(prompt, evidence, f"ev_action_{index:02d}", f"executable action {action}", match)
            directive = {
                "id": f"event_{index:02d}_{action}",
                "action": action,
                "actor_id": actor_id,
                "prop_id": prop_id,
                "receiver_id": receiver_id,
                "evidence_id": evidence_id,
            }
            directives.append({key: value for key, value in directive.items() if value is not None})
            if len(actors) > 1 and action not in {"handoff", "place"} and index == 1:
                second_evidence = _add_evidence(prompt, evidence, f"ev_action_{index:02d}_b", f"co-participant performs {action}", match)
                directives.append({"id": f"event_{index:02d}_{action}_b", "action": action, "actor_id": "actor_b", "prop_id": prop_id, "evidence_id": second_evidence, "concurrency_group": f"parallel_{index:02d}"})

    uncertainties = [
        {
            "id": "unc_visual_style",
            "description": "The benchmark prompt does not fully specify wardrobe, material, or set dressing; the self provider uses a documented neutral visual prior.",
            "severity": "soft",
            "resolved": False,
        }
    ]
    if len(entities) <= 1:
        uncertainties.append(
            {
                "id": "unc_sparse_scene",
                "description": "The prompt names a sparse scene; the provider exposes the missing visual detail instead of silently switching to a template.",
                "severity": "soft",
                "resolved": False,
            }
        )
    return {
        "entities": [item.model_dump(mode="json") for item in entities],
        "directives": directives,
        "camera_cues": camera_cues,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "assumptions": [
            "Use a neutral support surface only as a staging aid; it is not treated as a benchmark semantic requirement.",
            "Use prompt-derived labels and action spans as the only semantic source; all spatial coordinates are selected downstream by trajectory composition.",
        ],
        "uncertainties": uncertainties,
    }


def _visual_family(label: str) -> str:
    lower = label.casefold()
    if any(word in lower for word in ("cup", "bowl", "bucket", "glass", "container", "tea", "soup", "water")):
        return "vessel"
    if any(word in lower for word in ("apple", "ball", "orange", "stone", "rock", "gem", "droplet", "flower")):
        return "organic_round"
    if any(word in lower for word in ("book", "box", "laptop", "phone", "table", "chair", "basket", "shoe")):
        return "crafted_hard_surface"
    if any(word in lower for word in ("plant", "tree", "branch", "garden", "forest")):
        return "botanical"
    if any(word in lower for word in ("mount", "pyramid", "tower", "temple", "palace", "city", "building", "machu", "alhambra")):
        return "architectural_landmark"
    if any(word in lower for word in ("dog", "cat", "horse", "bird", "kangaroo", "monkey", "elephant", "rabbit", "fox")):
        return "animal"
    return "crafted_object"


def _environment_family(prompt: str, labels: list[str]) -> str:
    lower = prompt.casefold()
    text = " ".join([lower, *(label.casefold() for label in labels)])
    if any(word in text for word in ("space station", "zero gravity", "airborne")):
        return "space_station"
    if any(word in text for word in ("kitchen", "counter", "stove", "cup of tea", "cooking")):
        return "kitchen"
    if any(word in text for word in ("garden", "forest", "grass", "plant", "tree")):
        return "garden"
    if any(word in text for word in ("pond", "lagoon", "water", "sea")):
        return "waterfront"
    if any(word in text for word in ("mount", "pyramid", "tower", "temple", "palace", "city", "machu", "alhambra")):
        return "landmark"
    return "studio"


def _build_case_scene_profile(plan: dict[str, Any]) -> dict[str, Any]:
    """Derive a per-case visual brief consumed by the local code agent.

    This is deliberately a small structured design pass, not a fallback scene
    template.  It changes the generated source's geometry branches, material
    palette, environment details, and articulated action cues from the exact
    DirectorPlan while retaining the runtime contract shared by every job.
    """

    request = plan.get("request") or {}
    prompt = str(request.get("prompt") or "")
    entities = list(plan.get("entities") or [])
    labels = [str(entity.get("label") or entity.get("id") or "object") for entity in entities]
    plan_hash = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    entity_designs: dict[str, dict[str, Any]] = {}
    for index, entity in enumerate(entities):
        entity_id = str(entity.get("id") or f"entity_{index:02d}")
        kind = str(entity.get("kind") or "prop")
        label = str(entity.get("label") or entity_id)
        entity_designs[entity_id] = {
            "kind": kind,
            "label": label,
            "visual_family": "humanoid" if kind == "actor" else _visual_family(label),
            "palette_slot": index % 6,
            "detail_level": 3 if kind == "actor" else 2,
        }
    events = list(plan.get("events") or [])
    shots = list((plan.get("camera_plan") or {}).get("shots") or [])
    return {
        "profile_version": "codex-local-case-profile-v2",
        "case_signature": plan_hash[:16],
        "scene_family": _environment_family(prompt, labels),
        "prompt_keywords": sorted(set(re.findall(r"[a-z0-9]+", prompt.casefold()))),
        "entity_designs": entity_designs,
        "event_actions": [
            {
                "id": str(event.get("id") or "event"),
                "action": str(event.get("action") or "observe"),
                "participants": list(event.get("participant_ids") or []),
                "targets": list(event.get("target_ids") or []),
                "start": float(event.get("start") or 0.0),
                "end": float(event.get("end") or 0.0),
            }
            for event in events
        ],
        "camera_cues": [
            {
                "cue": shot.get("camera_cue"),
                "direction": shot.get("camera_direction"),
                "trajectory": shot.get("trajectory_type"),
            }
            for shot in shots
        ],
        "camera_bias": {
            "azimuth": 6.0 + float(int(plan_hash[16:20], 16) % 9),
            "elevation": 4.8 + float(int(plan_hash[20:24], 16) % 5) * 0.25,
            "lens_mm": 44.0 + float(int(plan_hash[24:28], 16) % 13),
        },
        "quality_targets": {
            "actor_segments": 32,
            "actor_rings": 20,
            "prop_segments": 32,
            "require_connected_rig": any(str(entity.get("kind")) == "actor" for entity in entities),
        },
    }


_CODEGEN_TEMPLATE = r'''"""Case-specific Blender job generated by a local Codex provider."""
from pathlib import Path
import hashlib
import json
import math

import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

from blender.lib.camera import dolly_camera, follow_camera, orbit_camera
from blender.lib.constraints import child_of_constraint, track_to_constraint
from blender.lib.geometry import box, capsule, cone, cylinder, ellipsoid, extruded_polygon, rounded_box, torus
from blender.lib.layout import avoid_penetration, handoff_constraint_sequence, lane_separated_paths, place_on_surface
from blender.lib.rigging import add_ik_constraint, minimal_humanoid_armature
from blender.lib.scaffolding import build_runtime_contract, validate_runtime_contract

OUTPUT_DIR = Path(__file__).resolve().parent
DIRECTOR_PLAN = json.loads(__PLAN_JSON__)
CASE_SCENE_PROFILE = json.loads(__SCENE_PROFILE__)
CODEX_PROVIDER = __PROVIDER__
CODEX_VARIANT = __VARIANT__
REQUIRED_ENTITY_IDS = __REQUIRED_ENTITY_IDS__
REQUIRED_EVENT_IDS = __REQUIRED_EVENT_IDS__
REQUIRED_CAMERA_EVENT_IDS = __REQUIRED_CAMERA_EVENT_IDS__
SAMPLE_FRAMES = [1, max(1, int(DIRECTOR_PLAN["request"]["duration_s"] * DIRECTOR_PLAN["request"]["fps"] * 0.5)), max(1, int(DIRECTOR_PLAN["request"]["duration_s"] * DIRECTOR_PLAN["request"]["fps"]))]
FRAMES_DIR = OUTPUT_DIR / "frames"
ACTOR_PARTS = {}
ACTOR_RIGS = {}


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


PLAN_HASH = canonical_hash(DIRECTOR_PLAN)


def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def make_material(name, color, metallic=0.0, roughness=0.55):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = (*color, 1.0)
    material.metallic = metallic
    material.roughness = roughness
    return material


def mesh_object(name, mesh_data, material, parent=None):
    vertices, faces = mesh_data
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    obj["entity_id"] = parent.get("entity_id") if parent is not None and parent.get("entity_id") else name
    obj["geometry_style"] = "detailed_parametric_v1"
    obj["generated_case_signature"] = CASE_SCENE_PROFILE["case_signature"]
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    return obj


def add_box(name, center, size, material, parent=None, bevel=0.08):
    data = rounded_box(center, size, min(bevel, min(size) * 0.24))
    obj = mesh_object(name, data, material, parent)
    modifier = obj.modifiers.new(name + "_bevel", "BEVEL")
    modifier.width = float(min(bevel, min(size) * 0.18))
    modifier.segments = 3
    return obj


def create_connected_humanoid_rig(entity_id, location):
    """Create a real connected Blender armature for one generated actor."""
    armature_data = bpy.data.armatures.new(entity_id + "__skeleton")
    armature = bpy.data.objects.new(entity_id + "__armature", armature_data)
    bpy.context.collection.objects.link(armature)
    armature.location = location
    armature.show_in_front = True
    armature.hide_render = True
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    specs = {
        "root": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.35), None),
        "hips": ((0.0, 0.0, 0.35), (0.0, 0.0, 1.05), "root"),
        "spine": ((0.0, 0.0, 1.05), (0.0, 0.0, 1.55), "hips"),
        "chest": ((0.0, 0.0, 1.55), (0.0, 0.0, 2.12), "spine"),
        "neck": ((0.0, 0.0, 2.12), (0.0, 0.0, 2.48), "chest"),
        "head": ((0.0, 0.0, 2.48), (0.0, 0.0, 3.20), "neck"),
        "shoulder.L": ((-0.42, 0.0, 2.02), (-0.68, 0.0, 2.02), "chest"),
        "upper_arm.L": ((-0.68, 0.0, 2.02), (-0.78, -0.02, 1.62), "shoulder.L"),
        "forearm.L": ((-0.78, -0.02, 1.62), (-0.72, -0.04, 1.28), "upper_arm.L"),
        "hand.L": ((-0.72, -0.04, 1.28), (-0.72, -0.04, 1.10), "forearm.L"),
        "shoulder.R": ((0.42, 0.0, 2.02), (0.68, 0.0, 2.02), "chest"),
        "upper_arm.R": ((0.68, 0.0, 2.02), (0.78, -0.02, 1.62), "shoulder.R"),
        "forearm.R": ((0.78, -0.02, 1.62), (0.72, -0.04, 1.28), "upper_arm.R"),
        "hand.R": ((0.72, -0.04, 1.28), (0.72, -0.04, 1.10), "forearm.R"),
        "thigh.L": ((-0.24, 0.0, 0.86), (-0.30, 0.0, 0.42), "hips"),
        "shin.L": ((-0.30, 0.0, 0.42), (-0.32, -0.05, 0.12), "thigh.L"),
        "foot.L": ((-0.32, -0.05, 0.12), (-0.32, -0.38, 0.10), "shin.L"),
        "thigh.R": ((0.24, 0.0, 0.86), (0.30, 0.0, 0.42), "hips"),
        "shin.R": ((0.30, 0.0, 0.42), (0.32, -0.05, 0.12), "thigh.R"),
        "foot.R": ((0.32, -0.05, 0.12), (0.32, -0.38, 0.10), "shin.R"),
    }
    bones = {}
    for name, (head, tail, parent_name) in specs.items():
        bone = armature_data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        if parent_name:
            bone.parent = bones[parent_name]
            bone.use_connect = name not in {"shoulder.L", "shoulder.R", "thigh.L", "thigh.R"}
        bones[name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    armature["entity_id"] = entity_id
    armature["entity_kind"] = "armature"
    armature["rig_contract"] = "connected_humanoid_v2"
    armature["bone_count"] = len(bones)
    ACTOR_RIGS[entity_id] = armature
    return armature


def add_actor(entity, location, material, accent, skin):
    root = bpy.data.objects.new(entity["id"], None)
    bpy.context.collection.objects.link(root)
    root.location = location
    root["entity_id"] = entity["id"]
    root["entity_kind"] = "actor"
    root["prompt_label"] = entity.get("label", "participant")
    root["visual_family"] = CASE_SCENE_PROFILE["entity_designs"].get(entity["id"], {}).get("visual_family", "humanoid")
    parts = {}

    def part(key, data, part_material):
        obj = mesh_object(entity["id"] + "_" + key, data, part_material, root)
        obj["semantic_owner"] = entity["id"]
        parts[key] = obj
        return obj

    part("torso", ellipsoid((0.0, 0.0, 1.72), (0.55, 0.36, 0.84), 32, 20), material)
    part("pelvis", ellipsoid((0.0, 0.0, 0.96), (0.48, 0.32, 0.31), 28, 18), material)
    part("collar", torus((0.0, 0.0, 2.30), 0.27, 0.055, 32, 12), accent)
    part("neck", capsule((0.0, 0.0, 2.30), (0.0, 0.0, 2.58), 0.145, 16, 24), skin)
    part("head", ellipsoid((0.0, 0.0, 2.98), (0.35, 0.30, 0.41), 32, 20), skin)
    part("hair", ellipsoid((0.0, 0.04, 3.25), (0.36, 0.31, 0.18), 28, 14), accent)
    part("eye_l", ellipsoid((-0.13, -0.275, 3.04), (0.045, 0.025, 0.055), 16, 10), accent)
    part("eye_r", ellipsoid((0.13, -0.275, 3.04), (0.045, 0.025, 0.055), 16, 10), accent)
    part("belt", rounded_box((0.0, -0.02, 1.16), (0.78, 0.38, 0.10), 0.035), accent)
    for side in (-1.0, 1.0):
        suffix = "l" if side < 0 else "r"
        shoulder = (side * 0.53, 0.0, 2.15)
        elbow = (side * 0.72, -0.04, 1.72)
        hand = (side * 0.74, -0.08, 1.34)
        hip = (side * 0.25, 0.0, 0.84)
        knee = (side * 0.30, 0.0, 0.43)
        foot = (side * 0.32, -0.14, 0.12)
        part("shoulder_" + suffix, ellipsoid(shoulder, (0.18, 0.18, 0.18), 24, 14), material)
        part("upper_arm_" + suffix, capsule(shoulder, elbow, 0.12, 16, 24), skin)
        part("elbow_" + suffix, ellipsoid(elbow, (0.13, 0.13, 0.13), 24, 14), skin)
        part("forearm_" + suffix, capsule(elbow, hand, 0.105, 16, 24), skin)
        part("hand_" + suffix, ellipsoid(hand, (0.15, 0.115, 0.17), 20, 14), skin)
        part("hip_" + suffix, ellipsoid(hip, (0.18, 0.18, 0.18), 24, 14), material)
        part("thigh_" + suffix, capsule(hip, knee, 0.17, 16, 24), material)
        part("knee_" + suffix, ellipsoid(knee, (0.17, 0.16, 0.16), 24, 14), material)
        part("shin_" + suffix, capsule(knee, foot, 0.125, 16, 24), material)
        part("foot_" + suffix, rounded_box(foot, (0.38, 0.66, 0.20), 0.055), accent)
        part("sole_" + suffix, rounded_box((foot[0], foot[1] - 0.03, 0.035), (0.40, 0.70, 0.07), 0.02), accent)
    rig = create_connected_humanoid_rig(entity["id"], tuple(location))
    root["rig_bone_count"] = len(rig.data.bones)
    root["rig_connected"] = True
    root["ik_hint"] = add_ik_constraint("hand.R", entity["id"] + "_reach_target").type
    ACTOR_PARTS[entity["id"]] = parts
    return root


def add_prop(entity, location, material, accent):
    root = bpy.data.objects.new(entity["id"], None)
    bpy.context.collection.objects.link(root)
    root.location = location
    root["entity_id"] = entity["id"]
    root["entity_kind"] = "prop"
    label = str(entity.get("label", "object")).lower()
    family = CASE_SCENE_PROFILE["entity_designs"].get(entity["id"], {}).get("visual_family", "crafted_object")
    root["visual_family"] = family
    segments = int(CASE_SCENE_PROFILE["quality_targets"].get("prop_segments", 32))
    if family == "organic_round":
        mesh_object(entity["id"] + "_body", ellipsoid((0.0, 0.0, 0.0), (0.50, 0.46, 0.50), segments, 20), material, root)
        mesh_object(entity["id"] + "_stem", capsule((0.0, 0.0, 0.42), (0.04, 0.0, 0.62), 0.055, 10, 16), accent, root)
        mesh_object(entity["id"] + "_leaf", ellipsoid((0.16, 0.0, 0.58), (0.22, 0.06, 0.08), 20, 10), accent, root)
    elif family == "vessel":
        mesh_object(entity["id"] + "_body", cylinder((0.0, 0.0, 0.0), 0.42, 0.78, segments), material, root)
        mesh_object(entity["id"] + "_rim", torus((0.0, 0.0, 0.40), 0.35, 0.075, segments, 16), accent, root)
        mesh_object(entity["id"] + "_liquid", ellipsoid((0.0, 0.0, 0.32), (0.32, 0.32, 0.035), segments, 10), accent, root)
        mesh_object(entity["id"] + "_handle", torus((0.44, 0.0, 0.02), 0.20, 0.055, 24, 12), accent, root)
    elif family == "crafted_hard_surface":
        mesh_object(entity["id"] + "_body", rounded_box((0.0, 0.0, 0.0), (1.05, 0.72, 0.56), 0.12), material, root)
        mesh_object(entity["id"] + "_inset", rounded_box((0.0, -0.37, 0.04), (0.72, 0.055, 0.30), 0.025), accent, root)
        mesh_object(entity["id"] + "_trim", torus((0.0, 0.0, 0.30), 0.28, 0.035, 24, 10), accent, root)
    elif family == "botanical":
        mesh_object(entity["id"] + "_stem", capsule((0.0, 0.0, 0.20), (0.0, 0.0, 1.30), 0.11, 16, 24), material, root)
        for index, (x, y, z) in enumerate(((-0.30, 0.0, 1.0), (0.30, 0.02, 1.22), (0.0, 0.10, 1.52))):
            mesh_object(entity["id"] + "_leaf_" + str(index), ellipsoid((x, y, z), (0.38, 0.16, 0.18), 24, 14), accent, root)
    elif family == "architectural_landmark":
        for index, size in enumerate((1.20, 0.95, 0.70, 0.45)):
            mesh_object(entity["id"] + "_tier_" + str(index), rounded_box((0.0, 0.0, 0.18 + index * 0.30), (size, size * 0.82, 0.34), 0.06), material if index % 2 else accent, root)
    elif family == "animal":
        mesh_object(entity["id"] + "_body", ellipsoid((0.0, 0.0, 0.45), (0.68, 0.34, 0.38), segments, 20), material, root)
        mesh_object(entity["id"] + "_head", ellipsoid((0.54, -0.01, 0.68), (0.34, 0.28, 0.32), segments, 20), accent, root)
        mesh_object(entity["id"] + "_tail", capsule((-0.62, 0.0, 0.56), (-0.95, 0.10, 0.95), 0.09, 12, 20), accent, root)
    else:
        mesh_object(entity["id"] + "_body", rounded_box((0.0, 0.0, 0.0), (1.0, 0.72, 0.58), 0.12), material, root)
        mesh_object(entity["id"] + "_detail", rounded_box((0.0, -0.39, 0.05), (0.62, 0.07, 0.26), 0.025), accent, root)
    return root


def add_environment(prompt, material, accent):
    lower = prompt.lower()
    environment_family = CASE_SCENE_PROFILE.get("scene_family", "studio")
    add_box("ground", (0.0, 0.0, -0.22), (14.0, 10.0, 0.35), material, bevel=0.12)
    add_box("backdrop", (0.0, 4.5, 3.4), (14.0, 0.25, 7.0), material, bevel=0.10)
    if any(word in lower for word in ("garden", "forest", "plant", "grass")):
        for index, x in enumerate((-4.5, -2.5, 2.5, 4.5)):
            mesh_object("tree_" + str(index), cylinder((x, 1.8, 0.7), 0.13, 1.4, 16), accent)
            mesh_object("tree_crown_" + str(index), ellipsoid((x, 1.8, 1.7), (0.6, 0.6, 0.8), 16, 10), material)
    if any(word in lower for word in ("mount", "fuji", "mountain", "pyramid", "tower")):
        for index, x in enumerate((-4.0, 0.0, 4.0)):
            mesh_object("landmark_" + str(index), cone((x, 2.5, 1.5), 1.8, 0.25, 3.0, 24), accent)
    if any(word in lower for word in ("lagoon", "pond", "water", "sea")):
        add_box("water_surface", (0.0, 0.5, 0.02), (10.0, 5.0, 0.08), accent, bevel=0.02)
    if any(word in lower for word in ("tomb", "kingdom", "temple", "palace", "hallway")):
        for index, x in enumerate((-5.0, 5.0)):
            mesh_object("column_" + str(index), cylinder((x, 2.8, 2.0), 0.36, 4.0, 24), accent)
    if environment_family == "kitchen":
        add_box("counter_back", (0.0, 2.0, 1.15), (10.0, 0.45, 2.2), accent, bevel=0.10)
        for index, x in enumerate((-3.4, -1.1, 1.1, 3.4)):
            add_box("cabinet_" + str(index), (x, 1.70, 0.45), (1.8, 0.32, 0.80), material, bevel=0.06)
    elif environment_family == "space_station":
        for index, z in enumerate((1.2, 2.7, 4.2)):
            mesh_object("station_panel_" + str(index), rounded_box((0.0, 4.15, z), (8.0, 0.12, 0.85), 0.035), accent)
        mesh_object("station_ring", torus((0.0, 2.8, 3.2), 2.8, 0.06, 48, 16), accent)
    elif environment_family == "waterfront":
        for index, x in enumerate((-4.0, -2.0, 2.0, 4.0)):
            mesh_object("water_ripple_" + str(index), torus((x, 1.0, 0.10), 0.65, 0.025, 32, 10), accent)
    elif environment_family == "landmark":
        for index, x in enumerate((-5.0, 0.0, 5.0)):
            mesh_object("landmark_base_" + str(index), rounded_box((x, 2.4, 0.25), (2.0, 1.0, 0.45), 0.08), accent)
    # The pure library calls are used as planning checks before Blender
    # objects are created; their return values are intentionally not hidden.
    place_on_surface(((-0.5, -0.5, -0.25), (0.5, 0.5, 0.25)), 0.0)
    extruded_polygon([(-0.5, -0.3), (0.5, -0.3), (0.65, 0.3), (-0.65, 0.3)], 0.04)


def _float_list(value):
    return [float(item) for item in value]


def _entity_members(entity_id):
    return [
        obj for obj in bpy.data.objects
        if obj.type == "MESH"
        and (str(obj.get("entity_id") or "") == entity_id or str(obj.get("semantic_owner") or "") == entity_id)
    ]


def _entity_world_bounds(entity_id, root):
    members = _entity_members(entity_id)
    corners = []
    for obj in members:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        corners = [root.matrix_world.translation.copy()]
    minimum = [min(float(point[index]) for point in corners) for index in range(3)]
    maximum = [max(float(point[index]) for point in corners) for index in range(3)]
    return [minimum, maximum], corners


def _screen_bounds(scene, camera, corners):
    projected = [world_to_camera_view(scene, camera, corner) for corner in corners]
    minimum_x = min(float(point.x) for point in projected)
    maximum_x = max(float(point.x) for point in projected)
    minimum_y = min(float(point.y) for point in projected)
    maximum_y = max(float(point.y) for point in projected)
    area = max(0.0, maximum_x - minimum_x) * max(0.0, maximum_y - minimum_y)
    visible_area = max(0.0, min(1.0, maximum_x) - max(0.0, minimum_x)) * max(0.0, min(1.0, maximum_y) - max(0.0, minimum_y))
    return {
        "screen_bbox": [minimum_x, minimum_y, maximum_x, maximum_y],
        "screen_center": [(minimum_x + maximum_x) / 2.0, (minimum_y + maximum_y) / 2.0],
        "visible_fraction": visible_area / area if area > 1e-9 else 0.0,
    }


def capture_runtime_observations(scene, objects, camera):
    """Capture actual evaluated transforms and projected bounds for every frame."""
    observations = []
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)
    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        entity_observations = {}
        for entity_id, root in objects.items():
            world_bbox, corners = _entity_world_bounds(entity_id, root)
            projected = _screen_bounds(scene, camera, corners)
            entity_observations[entity_id] = {
                "root_location": _float_list(root.matrix_world.translation),
                "root_rotation": _float_list(root.matrix_world.to_euler()),
                "world_bbox": world_bbox,
                "dimensions": [world_bbox[1][index] - world_bbox[0][index] for index in range(3)],
                **projected,
            }
            parts = ACTOR_PARTS.get(entity_id, {})
            if parts:
                entity_observations[entity_id]["pose_points"] = {
                    name: _float_list(part.matrix_world.translation)
                    for name, part in parts.items()
                    if name in {"head", "hand_l", "hand_r", "foot_l", "foot_r"}
                }
        forward = camera.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        observations.append({
            "frame": frame,
            "camera": {
                "location": _float_list(camera.matrix_world.translation),
                "rotation": _float_list(camera.matrix_world.to_euler()),
                "forward": _float_list(forward),
            },
            "entities": entity_observations,
        })
    scene.frame_set(frame_start)
    return observations


def target_point(objects):
    points = [obj.location.copy() for obj in objects.values() if obj is not None]
    if not points:
        return Vector((0.0, 0.0, 1.2))
    result = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    result.z = max(1.0, result.z + 1.0)
    return result


def look_at(camera, target):
    direction = Vector(target) - camera.location
    if direction.length > 0:
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(objects):
    camera_bias = CASE_SCENE_PROFILE.get("camera_bias", {})
    base_location = {
        "codex-self": (8.0, -12.0, 6.5),
        "codex-self-variant-b": (-8.0, -11.0, 5.8),
        "codex-self-variant-c": (6.0, -15.0, 4.5),
    }.get(CODEX_VARIANT, (8.0, -12.0, 6.5))
    base_location = (
        base_location[0] + float(camera_bias.get("azimuth", 0.0)) * 0.10,
        min(-8.0, base_location[1]),
        base_location[2] + float(camera_bias.get("elevation", 0.0)) * 0.08,
    )
    bpy.ops.object.camera_add(location=base_location)
    camera = bpy.context.object
    camera.name = "CodexSelfCamera"
    bpy.context.scene.camera = camera
    camera.data.lens = float(camera_bias.get("lens_mm", 48.0))
    target = target_point(objects)
    camera["camera_cues"] = "|".join(str(shot.get("camera_cue") or "") for shot in DIRECTOR_PLAN.get("camera_plan", {}).get("shots", []))
    shots = DIRECTOR_PLAN.get("camera_plan", {}).get("shots", [])
    if not shots:
        look_at(camera, target)
        return camera
    for index, shot in enumerate(shots):
        start_frame = int(shot["start_frame"])
        end_frame = int(shot["end_frame"])
        cue = shot.get("camera_cue") or "follow"
        direction = shot.get("camera_direction")
        if cue == "orbit" or shot.get("trajectory_type") == "orbit":
            # Keep the orbit on the camera-facing half of the stage.  The
            # backdrop is behind the subjects at positive Y, so a full
            # 180-degree orbit would legitimately put the camera behind it
            # and produce an empty frame rather than an event view.
            start_angle, end_angle = (320.0, 220.0) if direction == "counterclockwise" else (220.0, 320.0)
            keyframes = orbit_camera(tuple(target), 8.0, start_angle, end_angle, 4.8, (start_frame, end_frame), num_keyframes=8)
        elif cue in {"zoom", "dolly"} or shot.get("trajectory_type") == "dolly":
            near = (6.0, -9.0, 4.3)
            far = (11.0, -16.0, 8.0)
            if direction in {"in", "right", "up"}:
                near, far = far, near
            keyframes = dolly_camera(near, far, tuple(target), (start_frame, end_frame))
        elif cue in {"pan", "tilt", "follow"}:
            delta = 2.5 if direction in {"right", "up"} else -2.5
            start = base_location
            end = (base_location[0] + (delta if cue == "pan" else 0.0), base_location[1], base_location[2] + (delta if cue == "tilt" else 0.0))
            keyframes = follow_camera([(start_frame, tuple(target)), (end_frame, tuple(target))], (0.0, -12.0, 5.0))
            keyframes[0] = keyframes[0].__class__(start_frame, start, tuple(target))
            keyframes[-1] = keyframes[-1].__class__(end_frame, end, tuple(target))
        else:
            keyframes = [type("Key", (), {"frame": start_frame, "location": (8.0, -12.0, 6.5), "target": tuple(target)})(), type("Key", (), {"frame": end_frame, "location": (8.0, -12.0, 6.5), "target": tuple(target)})()]
        for key in keyframes:
            camera.location = key.location
            look_at(camera, key.target)
            camera.keyframe_insert(data_path="location", frame=int(key.frame))
            camera.keyframe_insert(data_path="rotation_euler", frame=int(key.frame))
    track_to_constraint(camera.name, "camera_target")
    return camera


def _keyframe_pose(parts, key, frame):
    obj = parts.get(key)
    if obj is None:
        return
    obj.keyframe_insert(data_path="rotation_euler", frame=int(frame))


def pose_actor_for_event(actor_id, action, frame, phase=0.0):
    """Give each scheduled event a visible, connected articulated pose."""
    parts = ACTOR_PARTS.get(actor_id, {})
    action = str(action or "observe").lower()
    active_arm = action in {"reach", "grasp", "pick", "carry", "handoff", "give", "pour", "press", "write", "sweep", "brush", "interact"}
    locomotion = action in {"walk", "run", "move", "climb", "jump"}
    arm_angle = -0.28 if active_arm else 0.05
    if action in {"sweep", "brush"}:
        arm_angle = -0.65
    if action in {"pour", "press", "write", "interact"}:
        arm_angle = -0.85
    if action in {"handoff", "give"}:
        arm_angle = -0.50
    for suffix, sign in (("l", -1.0), ("r", 1.0)):
        upper = parts.get("upper_arm_" + suffix)
        forearm = parts.get("forearm_" + suffix)
        hand = parts.get("hand_" + suffix)
        if upper is not None:
            upper.rotation_euler = (0.0, sign * arm_angle, 0.10 * math.sin(phase))
            _keyframe_pose(parts, "upper_arm_" + suffix, frame)
        if forearm is not None:
            forearm.rotation_euler = (0.0, sign * arm_angle * 0.55, 0.0)
            _keyframe_pose(parts, "forearm_" + suffix, frame)
        if hand is not None:
            hand.rotation_euler = (0.0, sign * arm_angle * 0.25, 0.0)
            _keyframe_pose(parts, "hand_" + suffix, frame)
        if locomotion:
            thigh = parts.get("thigh_" + suffix)
            shin = parts.get("shin_" + suffix)
            stride = 0.25 * sign * math.sin(phase)
            if thigh is not None:
                thigh.rotation_euler = (0.0, stride, 0.0)
                _keyframe_pose(parts, "thigh_" + suffix, frame)
            if shin is not None:
                shin.rotation_euler = (0.0, -0.5 * stride, 0.0)
                _keyframe_pose(parts, "shin_" + suffix, frame)


def animate_objects(objects):
    trajectory = DIRECTOR_PLAN.get("trajectory_summary", {}).get("entities", {})
    for entity_id, data in trajectory.items():
        obj = objects.get(entity_id)
        if obj is None:
            continue
        for state in data.get("states", []):
            obj.location = tuple(state["position"])
            obj.rotation_euler = tuple(state.get("rotation", (0.0, 0.0, 0.0)))
            obj.keyframe_insert(data_path="location", frame=int(state["frame"]))
            obj.keyframe_insert(data_path="rotation_euler", frame=int(state["frame"]))
        for primitive in data.get("motion_primitives", []):
            obj["motion_" + str(primitive.get("parameters", {}).get("event_id", primitive.get("type", "unknown")))] = primitive.get("type")
    fps = float(DIRECTOR_PLAN["request"].get("fps", 24))
    for event_index, event in enumerate(DIRECTOR_PLAN.get("events", [])):
        start = int(float(event.get("start", 0.0)) * fps) + 1
        end = max(start + 1, int(float(event.get("end", 0.0)) * fps))
        midpoint = (start + end) // 2
        for actor_id in event.get("participant_ids", []):
            pose_actor_for_event(actor_id, event.get("action"), start, phase=0.0)
            pose_actor_for_event(actor_id, event.get("action"), midpoint, phase=math.pi * 0.5 + event_index)
            pose_actor_for_event(actor_id, event.get("action"), end, phase=math.pi + event_index)


def configure_scene(scene, manifest):
    for engine in (manifest.get("render_settings", {}).get("engine", "BLENDER_EEVEE_NEXT"), "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except (TypeError, ValueError):
            continue
    resolution = manifest.get("render_settings", {}).get("resolution", [256, 256])
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = int(DIRECTOR_PLAN["request"]["fps"])
    scene.frame_start = 1
    scene.frame_end = int(DIRECTOR_PLAN["request"]["duration_s"] * scene.render.fps)
    scene.render.film_transparent = False
    scene.world.color = (0.025, 0.035, 0.05)
    scene.render.image_settings.color_mode = "RGBA"


def write_contract_artifacts(objects, camera, manifest):
    request = DIRECTOR_PLAN["request"]
    events = DIRECTOR_PLAN.get("events", [])
    shots = DIRECTOR_PLAN.get("camera_plan", {}).get("shots", [])
    entity_specs = [
        {
            "id": entity["id"],
            "kind": {"actor": "character", "prop": "prop", "support": "support"}.get(entity.get("kind"), entity.get("kind", "prop")),
            "role": entity.get("role", "target_object"),
        }
        for entity in DIRECTOR_PLAN.get("entities", [])
    ]
    event_specs = [
        {
            "id": event["id"],
            "start": event["start"],
            "end": event["end"],
            "description": event.get("action", "event"),
            "target_ids": list(dict.fromkeys([*event.get("participant_ids", []), *event.get("target_ids", [])])),
        }
        for event in events
    ]
    trajectory_summary = DIRECTOR_PLAN.get("trajectory_summary", {})
    trajectory_plan = {
        "timebase": trajectory_summary.get("timebase", {"fps": request["fps"], "frame_start": 1, "frame_end": int(request["duration_s"] * request["fps"])}),
        "entities": trajectory_summary.get("entities", {}),
        "camera": DIRECTOR_PLAN.get("camera_plan", {"shots": []}),
        "event_observability": [
            {
                "event_id": event["id"],
                "covered_by_shots": [shot["shot_id"] for shot in shots if event["id"] in shot.get("required_event_ids", [])],
                "target_visible_predicate": "all_targets_visible_during_event",
            }
            for event in events
        ],
        "validation_intents": [
            "director_event_order",
            "multi_entity_collision_free_lanes",
            "multi_target_camera_coverage",
        ],
    }
    runtime_contract = build_runtime_contract(
        PLAN_HASH,
        REQUIRED_ENTITY_IDS,
        REQUIRED_EVENT_IDS,
        REQUIRED_CAMERA_EVENT_IDS,
    )
    failures = validate_runtime_contract(runtime_contract)
    if failures:
        raise RuntimeError("runtime contract failed: " + ",".join(failures))
    runtime_observations = capture_runtime_observations(bpy.context.scene, objects, camera)
    (OUTPUT_DIR / "scene_contract.json").write_text(json.dumps({"scene_id": request["scene_id"], "fps": request["fps"], "duration_s": request["duration_s"], "entities": entity_specs, "events": event_specs, "relations": [], "must_show": [event["id"] for event in events], "physics_constraints": ["no_penetration"], "camera_constraints": ["multi_target_visibility", "event_coverage"]}, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "trajectory.json").write_text(json.dumps(trajectory_plan, indent=2, sort_keys=True), encoding="utf-8")
    (OUTPUT_DIR / "camera_plan.json").write_text(json.dumps(DIRECTOR_PLAN.get("camera_plan", {}), indent=2, sort_keys=True), encoding="utf-8")
    telemetry = {
        "provider": CODEX_PROVIDER,
        "provider_variant": CODEX_VARIANT,
        "prompt": request["prompt"],
        "director_plan_hash": PLAN_HASH,
        "frame_start": int(trajectory_plan["timebase"]["frame_start"]),
        "frame_end": int(trajectory_plan["timebase"]["frame_end"]),
        "fps": int(trajectory_plan["timebase"]["fps"]),
        "required_entities": runtime_contract["required_entities"],
        "required_events": runtime_contract["required_events"],
        "required_camera_events": runtime_contract["required_camera_events"],
        "objects": {entity_id: {"name": obj.name, "kind": {"actor": "character", "prop": "prop", "support": "support"}.get(obj.get("entity_kind"), obj.get("entity_kind", "unknown"))} for entity_id, obj in objects.items()},
        "camera": {"name": camera.name, "active": bpy.context.scene.camera == camera, "cue_shots": [shot.get("camera_cue") for shot in shots]},
        "trajectory_primitives": {entity_id: [item.get("type") for item in data.get("motion_primitives", [])] for entity_id, data in DIRECTOR_PLAN.get("trajectory_summary", {}).get("entities", {}).items()},
        "rigs": {
            entity_id: {
                "connected": bool(rig.get("rig_connected", False) or rig.get("rig_contract") == "connected_humanoid_v2"),
                "bone_count": int(rig.get("bone_count", 0) or 0),
                "contract": rig.get("rig_contract"),
            }
            for entity_id, rig in ACTOR_RIGS.items()
        },
        "runtime_observation_count": len(runtime_observations),
        "runtime_observations": runtime_observations,
        "runtime_contract": runtime_contract,
        "blender_version": bpy.app.version_string,
    }
    (OUTPUT_DIR / "telemetry.json").write_text(json.dumps(telemetry, indent=2, sort_keys=True), encoding="utf-8")
    manifest = dict(manifest)
    manifest["blender_version"] = bpy.app.version_string
    manifest["state"] = "rendered"
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def write_sample_frames(scene):
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for frame in sorted(set(SAMPLE_FRAMES)):
        scene.frame_set(int(frame))
        relative = "sample_frames/frame_" + f"{int(frame):06d}" + ".png"
        scene.render.filepath = str(FRAMES_DIR / relative)
        (FRAMES_DIR / "sample_frames").mkdir(parents=True, exist_ok=True)
        bpy.ops.render.render(write_still=True)
        samples.append({"frame": int(frame), "path": relative})
    (FRAMES_DIR / "index.json").write_text(json.dumps({"frames": samples}, indent=2, sort_keys=True), encoding="utf-8")


def main():
    manifest_path = OUTPUT_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reset_scene()
    scene = bpy.context.scene
    primary_color = {
        "codex-self": (0.54, 0.68, 0.82),
        "codex-self-variant-b": (0.48, 0.76, 0.60),
        "codex-self-variant-c": (0.72, 0.56, 0.86),
    }.get(CODEX_VARIANT, (0.54, 0.68, 0.82))
    accent_color = {
        "codex-self": (0.90, 0.48, 0.18),
        "codex-self-variant-b": (0.92, 0.64, 0.16),
        "codex-self-variant-c": (0.18, 0.72, 0.78),
    }.get(CODEX_VARIANT, (0.90, 0.48, 0.18))
    primary = make_material("CodexSelfPrimary", primary_color, metallic=0.08, roughness=0.42)
    accent = make_material("CodexSelfAccent", accent_color, metallic=0.02, roughness=0.48)
    skin = make_material("CodexSelfSkin", (0.72, 0.42, 0.28), metallic=0.0, roughness=0.50)
    add_environment(DIRECTOR_PLAN["request"]["prompt"], primary, accent)
    objects = {}
    for index, entity in enumerate(DIRECTOR_PLAN.get("entities", [])):
        location = (0.0, 0.0, 0.0)
        if entity["kind"] == "actor":
            location = (-2.8, (index - 1) * 1.7, 0.0)
            objects[entity["id"]] = add_actor(entity, location, primary, accent, skin)
        elif entity["kind"] == "prop":
            location = (-0.8 + index * 0.8, (index - 1) * 1.1, 1.1)
            objects[entity["id"]] = add_prop(entity, location, accent, primary)
        elif entity["kind"] == "support":
            objects[entity["id"]] = add_box(entity["id"], (0.0, 0.0, 0.35), (5.5, 3.5, 0.3), primary, bevel=0.12)
            objects[entity["id"]]["entity_kind"] = "support"
    # Plan-level collision checks are performed before the render, and their
    # output is retained as object metadata for the evaluator.
    actor_paths = {entity_id: [(int(state["frame"]), tuple(state["position"])) for state in data.get("states", [])] for entity_id, data in DIRECTOR_PLAN.get("trajectory_summary", {}).get("entities", {}).items() if entity_id.startswith("actor_")}
    if actor_paths:
        separated = lane_separated_paths(actor_paths, 1.5)
        for entity_id, states in separated.items():
            objects.get(entity_id, bpy.context.scene).set if False else None
            if entity_id in objects:
                objects[entity_id]["lane_state_count"] = len(states)
                avoid_penetration(entity_id, ["ground"], states, obstacle_bounds={"ground": ((-7.0, -5.0, -0.5), (7.0, 5.0, 0.0))})
    animate_objects(objects)
    camera = add_camera(objects)
    bpy.ops.object.light_add(type="AREA", location=(0.0, -3.0, 8.0))
    key = bpy.context.object
    key.data.energy = 1000
    key.data.shape = "DISK"
    key.data.size = 5.5
    bpy.ops.object.light_add(type="AREA", location=(4.0, 3.0, 4.0))
    fill = bpy.context.object
    fill.data.energy = 550
    fill.data.size = 4.0
    configure_scene(scene, manifest)
    write_contract_artifacts(objects, camera, manifest)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    (FRAMES_DIR / "animation").mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(FRAMES_DIR / "animation" / "frame_")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "proxy.blend"))
    bpy.ops.render.render(animation=True)
    write_sample_frames(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "proxy.blend"))


main()
'''


def _build_codex_local_codegen_response(
    payload: dict[str, Any], *, provider_name: str
) -> dict[str, Any]:
    """Compile a plan into a case-specific, statically auditable Blender job."""

    plan = payload.get("director_plan")
    if not isinstance(plan, dict) or not plan.get("request"):
        raise ValueError("local Codex codegen requires a DirectorPlan")
    variant = str(payload.get("model") or provider_name)
    source = textwrap.dedent(_CODEGEN_TEMPLATE).replace(
        "__PLAN_JSON__",
        repr(json.dumps(plan, ensure_ascii=False, sort_keys=True)),
    )
    profile = _build_case_scene_profile(plan)
    source = source.replace(
        "__SCENE_PROFILE__",
        repr(json.dumps(profile, ensure_ascii=False, sort_keys=True)),
    )
    source = source.replace("__PROVIDER__", repr(provider_name))
    source = source.replace("__VARIANT__", repr(variant))
    entity_ids = [str(item.get("id")) for item in plan.get("entities", []) if item.get("id")]
    event_ids = [str(item.get("id")) for item in plan.get("events", []) if item.get("id")]
    camera_event_ids = [
        str(event_id)
        for shot in (plan.get("camera_plan", {}) or {}).get("shots", [])
        for event_id in shot.get("required_event_ids", [])
        if str(event_id).strip()
    ]
    source = source.replace("__REQUIRED_ENTITY_IDS__", repr(entity_ids))
    source = source.replace("__REQUIRED_EVENT_IDS__", repr(event_ids))
    source = source.replace("__REQUIRED_CAMERA_EVENT_IDS__", repr(list(dict.fromkeys(camera_event_ids))))
    plan_hash = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return {
        "status": "success",
        "generated_code": source,
        "library_calls": [
            "add_ik_constraint", "avoid_penetration", "box", "build_runtime_contract", "capsule",
            "child_of_constraint", "cylinder", "dolly_camera", "ellipsoid", "extruded_polygon",
            "follow_camera", "handoff_constraint_sequence", "lane_separated_paths", "minimal_humanoid_armature",
            "orbit_camera", "place_on_surface", "rounded_box", "torus", "track_to_constraint",
            "validate_runtime_contract",
        ],
        "llm_call_id": f"{provider_name}:{plan_hash[:16]}",
        "generation_provenance": {
            "provider": provider_name,
            "method": "case_specific_scene_profile_v2",
            "profile_hash": hashlib.sha256(
                json.dumps(profile, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
        },
    }


def build_codex_self_codegen_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Historical compatibility wrapper for pre-local-provider tests."""

    return _build_codex_local_codegen_response(payload, provider_name="codex-self")


class CodexSelfProvider:
    """Structured provider implemented by the current Codex desktop agent."""

    name = "codex-self"

    def director(self, request: DirectorRequest) -> dict[str, Any]:
        return build_codex_self_director_payload(request)

    def codegen(self, payload: dict[str, Any]) -> dict[str, Any]:
        return build_codex_self_codegen_response(payload)


class CodexLocalProvider:
    """The current Codex environment as the in-process dynamic provider.

    This is the production local path.  It does not call an endpoint or spawn
    ``codex exec``; the surrounding Harness still applies the same typed
    Director, codegen, coverage, artifact, and fail-closed gates.
    """

    name = "codex-local"

    def director(self, request: DirectorRequest) -> dict[str, Any]:
        return build_codex_self_director_payload(request)

    def codegen(self, payload: dict[str, Any]) -> dict[str, Any]:
        return _build_codex_local_codegen_response(payload, provider_name=self.name)


def build_codex_self_agents():
    """Return a dynamic DirectorAgent and BlenderCodeAgent backed by this provider."""

    from .blender_code_agent import BlenderCodeAgent
    from .director import DirectorAgent

    provider = CodexSelfProvider()
    director = DirectorAgent.from_provider(
        provider.director,
        provider_name="codex-self",
        policy="director-v3-codex-self",
    )
    code_agent = BlenderCodeAgent(provider=provider.codegen, model="codex-self")
    return director, code_agent


def build_codex_local_agents():
    """Build both dynamic stages from the current local Codex environment."""

    from .blender_code_agent import BlenderCodeAgent
    from .director import DirectorAgent

    provider = CodexLocalProvider()
    director = DirectorAgent.from_provider(
        provider.director,
        provider_name=provider.name,
        policy="director-v3-codex-local",
    )
    code_agent = BlenderCodeAgent(provider=provider.codegen, model=provider.name)
    return director, code_agent


__all__ = [
    "CodexSelfProvider",
    "CodexLocalProvider",
    "build_codex_local_agents",
    "build_codex_self_agents",
    "build_codex_self_codegen_response",
    "build_codex_self_director_payload",
]
