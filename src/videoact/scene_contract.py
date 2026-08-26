"""Deterministic prompt normalization into a validated SceneContract."""

from __future__ import annotations

import hashlib
import re

from .contracts import EntitySpec, EventSpec, RelationSpec, SceneContract, TrajectoryRequirement


class SceneContractBuilder:
    """Build a validated scene contract without generating Blender code."""

    def build(self, prompt: str, *, duration_s: float = 10.0, fps: int = 24) -> SceneContract:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if duration_s <= 0 or fps <= 0:
            raise ValueError("duration_s and fps must be positive")

        text = prompt.strip().lower()
        entities = self._entities(text)
        entity_ids = {entity.id for entity in entities}
        target_id = self._target_id(entity_ids)
        environment_ids = [entity.id for entity in entities if entity.role == "environment"]
        table_id = "table" if "table" in entity_ids else (environment_ids[0] if environment_ids else None)
        events = self._events(text, duration_s, table_id, target_id)

        relations = []
        if table_id and target_id:
            relations.append(RelationSpec(type="on", subject=target_id, object=table_id))

        camera_constraints = ["target_visible_before_grasp"] if target_id else []
        if re.search(r"\b(?:closeup|close-up|close up)\b", text):
            camera_constraints.append("grasp_in_closeup")
        if re.search(r"\b(?:follow|follows|following|track|tracking)\b", text):
            camera_constraints.append("camera_follow")
        if re.search(r"\b(?:orbit(?:s|ing)?|rotat(?:e|es|ed|ing)|arc(?:s|ing)?|circle(?:s|d|ing)?)\b", text):
            camera_constraints.append("camera_orbit")
        if re.search(r"\b(?:doll(?:y|ies|ied|ying)|push(?:es|ed|ing)?\s+in|move(?:s|d|ing)?\s+in)\b", text):
            camera_constraints.append("camera_dolly")
        if re.search(r"\b(?:occlusion|occluded|reveal|reveals|uncover)\b", text):
            camera_constraints.append("occlusion_reveal")
        if re.search(r"\breverse\s+(?:zigzag|path)\b", text):
            camera_constraints.append("reverse_path")

        physics_constraints = ["no_penetration"]
        if table_id and target_id:
            physics_constraints.extend(["support_before_grasp", "contact_before_attachment"])
        if any(event.id in {"lift", "carry", "place", "release"} for event in events):
            physics_constraints.append("attachment_lifecycle")

        event_ids = [event.id for event in events]
        actor_ids = [entity.id for entity in entities if entity.role in {"actor", "receiver"}]
        actor_phase_ids = [event_id for event_id in event_ids if event_id not in {"observe", "reveal"}]
        requirements = [
            TrajectoryRequirement(
                entity_id=actor_id,
                required_event_ids=actor_phase_ids,
                minimum_states=max(1, len(actor_phase_ids) + 1),
                require_phase_primitives=any(
                    event_id in {"lift", "carry", "place", "release"}
                    for event_id in actor_phase_ids
                ),
                required_attachment_actions=(
                    (["attach"] if "grasp" in event_ids else [])
                    + (["detach"] if "release" in event_ids else [])
                ) if actor_id == "character" and target_id else [],
            )
            for actor_id in actor_ids
            if actor_phase_ids
        ]
        if target_id:
            object_phase_ids = [
                event_id
                for event_id in event_ids
                if event_id in {"grasp", "lift", "carry", "place", "release"}
            ]
            if object_phase_ids:
                requirements.append(
                    TrajectoryRequirement(
                        entity_id=target_id,
                        required_event_ids=object_phase_ids,
                        minimum_states=max(2, len(object_phase_ids) + 1),
                    )
                )

        return SceneContract(
            scene_id=self._scene_id(prompt),
            duration_s=duration_s,
            fps=fps,
            entities=entities,
            events=events,
            relations=relations,
            must_show=[event.id for event in events],
            physics_constraints=physics_constraints,
            camera_constraints=camera_constraints,
            trajectory_requirements=requirements,
        )

    @staticmethod
    def _scene_id(prompt: str) -> str:
        digest = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:12]
        return f"scene-{digest}"

    @staticmethod
    def _entities(text: str) -> list[EntitySpec]:
        entities = [EntitySpec(id="character", kind="character", role="actor")]
        if re.search(r"\b(?:table|worktable|support|surface|platform)\b", text):
            entities.append(EntitySpec(id="table", kind="support", role="environment"))
        if re.search(r"\b(?:drop\s+zone|drop-zone|drop\s+platform|marked\s+destination)\b", text):
            entities.append(EntitySpec(id="drop_zone", kind="support", role="environment"))
        if re.search(r"\b(?:foreground\s+blocker|occluder|partition)\b", text):
            blocker_id = "partition" if "partition" in text else "foreground_blocker"
            entities.append(EntitySpec(id=blocker_id, kind="occluder", role="environment"))
        if re.search(r"\b(?:assistant|receiver|second\s+actor)\b", text):
            entities.append(EntitySpec(id="assistant", kind="character", role="receiver"))
        if re.search(r"\bopening\b", text):
            entities.append(EntitySpec(id="opening", kind="support", role="environment"))

        prop_pattern = r"\b(?:(red|blue|green|yellow)\s+)?(cup|cube|ball|book|object|prop)\b"
        seen: set[str] = set()
        for match in re.finditer(prop_pattern, text):
            color, kind = match.group(1), match.group(2)
            if color is None and kind in {"object", "prop"}:
                continue
            prop_id = f"{color}_{kind}" if color else kind
            if prop_id not in seen:
                entities.append(EntitySpec(id=prop_id, kind="prop", role="target_object"))
                seen.add(prop_id)
        return entities

    @staticmethod
    def _target_id(entity_ids: set[str]) -> str | None:
        for candidate in ("red_cup", "cup", "blue_cube", "green_ball", "yellow_book"):
            if candidate in entity_ids:
                return candidate
        for candidate in sorted(entity_ids):
            if candidate not in {"character", "table", "drop_zone"}:
                return candidate
        return None

    @staticmethod
    def _events(
        text: str,
        duration_s: float,
        table_id: str | None,
        target_id: str | None,
    ) -> list[EventSpec]:
        patterns = [
            ("walk", r"\bwalk(?:s|ed|ing)?\b|\bapproach(?:es|ed|ing)?\b|\bmove(?:s|d|ing)?\b|\bstroll(?:s|ed|ing)?\b|\badvance(?:s|d|ing)?\b"),
            ("reach", r"\breach(?:es|ed|ing)?\b|\bextend(?:s|ed|ing)?\b|\bextends?\s+(?:an?\s+)?arm\b"),
            ("grasp", r"pick(?:s)?\s*up|pickup|\bgrasp(?:s|ed|ing)?\b|\bgrab(?:s|bed|bing)?\b|\bseiz(?:e|es|ed|ing)\b|\bsnatch(?:es|ed|ing)?\b"),
            ("lift", r"\blift(?:s|ed|ing)?\b|\braise(?:s|d|ing)?\b|\bhoist(?:s|ed|ing)?\b|\belevat(?:e|es|ed|ing)\b|\braises?\b"),
            ("carry", r"\bcarr(?:y|ies|ied|ying)\b|\btransport(?:s|ed|ing)?\b|\bbring(?:s|ing)?\b|\bconvey(?:s|ed|ing)?\b"),
            ("place", r"\bplace(?:s|d|ing)?\b|\bsets?\s+(?:it\s+)?down\b|\bput\s+down\b|\blower(?:s|ed|ing)?\b|\bsettle(?:s|d|ing)?\b|\bsets?\s+(?:it\s+)?atop\b"),
            ("release", r"\brelease(?:s|d|ing)?\b|\bdetach(?:es|ed|ing)?\b|\blet\s+go\b|\bdrop(?:s|ped|ping)?\b(?!\s+zone)|\bunhand(?:s|ed|ing)?\b|\buncouple(?:s|d|ing)?\b"),
            ("reveal", r"\breveal(?:s|ed|ing)?\b|\buncover(?:s|ed|ing)?\b"),
        ]
        matches = []
        for action_id, pattern in patterns:
            match = re.search(pattern, text)
            if match:
                matches.append((match.start(), action_id))
        actions = [action_id for _, action_id in sorted(matches)]
        if "grasp" in actions and "reach" not in actions:
            actions.insert(actions.index("grasp"), "reach")
        if "grasp" in actions and "walk" not in actions and table_id:
            actions.insert(0, "walk")

        if len(actions) <= 3 and set(actions) <= {"walk", "reach", "grasp", "reveal"}:
            bounds = {"walk": (0.0, 0.4), "reach": (0.4, 0.6), "grasp": (0.6, 0.8), "reveal": (0.0, 1.0)}
        else:
            bounds = {
                "walk": (0.0, 0.25),
                "reach": (0.25, 0.4),
                "grasp": (0.4, 0.55),
                "lift": (0.55, 0.65),
                "carry": (0.65, 0.82),
                "place": (0.82, 0.92),
                "release": (0.92, 1.0),
                "reveal": (0.2, 0.35),
            }
        descriptions = {
            "walk": "character walks to the target support",
            "reach": "character reaches toward the target object",
            "grasp": "character grasps the target object",
            "lift": "character lifts the attached target object",
            "carry": "character carries the attached target object through the scene",
            "place": "character places the target object at the destination",
            "release": "character releases the target object",
            "reveal": "camera reveals the target object after occlusion",
        }
        events = []
        for action_id in actions:
            start_ratio, end_ratio = bounds.get(action_id, (0.0, 1.0))
            target_ids = [table_id] if action_id == "walk" and table_id else [target_id] if target_id else []
            events.append(
                EventSpec(
                    id=action_id,
                    start=round(duration_s * start_ratio, 4),
                    end=round(duration_s * end_ratio, 4),
                    description=descriptions[action_id],
                    target_ids=target_ids,
                )
            )
        if not events:
            events.append(EventSpec(id="observe", start=0.0, end=duration_s, description="scene remains observable"))
        return sorted(events, key=lambda event: event.start)
