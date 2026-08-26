"""Prompt interpretation for DirectorAgent.

This layer extracts actors, props, and narrative action directives. It does not
choose coordinates, trajectories, camera shots, or Blender implementation.
"""

from __future__ import annotations

import re
from typing_extensions import Literal

from pydantic import Field, model_validator

from .director_contracts import (
    ContractModel,
    DirectorDecisionEvidence,
    DirectorEntity,
    DirectorRequest,
    DirectorUncertainty,
)


DirectiveAction = Literal["carry", "handoff", "place", "pause", "return"]


class DirectorActionDirective(ContractModel):
    id: str = Field(min_length=1)
    action: DirectiveAction
    actor_id: str | None = None
    prop_id: str | None = None
    receiver_id: str | None = None
    concurrency_group: str | None = None
    evidence_id: str = Field(min_length=1)


class PromptInterpretation(ContractModel):
    request: DirectorRequest
    entities: list[DirectorEntity]
    directives: list[DirectorActionDirective]
    evidence: list[DirectorDecisionEvidence]
    assumptions: list[str] = Field(default_factory=list)
    uncertainties: list[DirectorUncertainty] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "PromptInterpretation":
        entity_ids = {entity.id for entity in self.entities}
        evidence_ids = {evidence.id for evidence in self.evidence}
        for directive in self.directives:
            for field_name in ("actor_id", "prop_id", "receiver_id"):
                reference = getattr(directive, field_name)
                if reference is not None and reference not in entity_ids:
                    raise ValueError(f"directive {directive.id} references unknown entity: {reference}")
            if directive.evidence_id not in evidence_ids:
                raise ValueError(f"directive {directive.id} references unknown evidence: {directive.evidence_id}")
        return self


class DeterministicPromptInterpreter:
    _ACTOR_LABELS = ("Alice", "Bob", "Carla", "Dana")
    _PROP_RE = re.compile(r"\b(red|blue|green|yellow)\s+(cube|cup|book|ball)\b", re.IGNORECASE)

    def interpret(self, request: DirectorRequest) -> PromptInterpretation:
        prompt = request.prompt.strip()
        actors = self._actors(prompt)
        props = self._props(prompt)
        actor_ids = {entity.label: entity.id for entity in actors}
        prop_ids = {entity.label.lower(): entity.id for entity in props}
        evidence: list[DirectorDecisionEvidence] = []
        directives: list[DirectorActionDirective] = []

        for entity in actors + props:
            span = self._find_span(prompt, entity.label)
            evidence.append(
                DirectorDecisionEvidence(
                    id=f"ev_{entity.id}",
                    source="prompt",
                    prompt_span=span,
                    claim=f"{entity.label} is a {entity.kind}.",
                )
            )

        while_match = re.search(
            r"\b(?P<a1>[A-Z][a-z]+)\s+carries\s+the\s+(?P<p1>(?:red|blue|green|yellow)\s+\w+)\s+while\s+"
            r"(?P<a2>[A-Z][a-z]+)\s+carries\s+the\s+(?P<p2>(?:red|blue|green|yellow)\s+\w+)",
            prompt,
        )
        if while_match:
            for index, actor_group, prop_group in (
                (1, "a1", "p1"),
                (2, "a2", "p2"),
            ):
                directives.append(
                    self._directive(
                        prompt,
                        evidence,
                        action="carry",
                        actor_id=actor_ids[while_match.group(actor_group)],
                        prop_id=prop_ids[while_match.group(prop_group).lower()],
                        concurrency_group="while_01",
                        evidence_label=f"parallel carry {index}",
                        span=(while_match.start(), while_match.end()),
                    )
                )
        else:
            carry_match = re.search(
                r"\b(?P<actor>[A-Z][a-z]+)\s+(?:carries|takes)\s+the\s+"
                r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)(?:\s+to\s+(?P<receiver>[A-Z][a-z]+))?",
                prompt,
            )
            if carry_match:
                directives.append(
                    self._directive(
                        prompt,
                        evidence,
                        action="carry",
                        actor_id=actor_ids[carry_match.group("actor")],
                        prop_id=prop_ids[carry_match.group("prop").lower()],
                        receiver_id=actor_ids.get(carry_match.group("receiver") or ""),
                        evidence_label="carry",
                        span=(carry_match.start(), carry_match.end()),
                    )
                )

        if re.search(r"\bpauses?\b", prompt, re.IGNORECASE):
            span = self._find_regex_span(prompt, r"\bpauses?\b")
            last = directives[-1] if directives else None
            directives.append(
                self._directive(
                    prompt,
                    evidence,
                    action="pause",
                    actor_id=last.actor_id if last else (actors[0].id if actors else None),
                    prop_id=last.prop_id if last else (props[0].id if props else None),
                    evidence_label="pause",
                    span=span,
                )
            )

        handoff_match = re.search(
            r"\b(?P<giver>[A-Z][a-z]+)\s+(?:hands|passes|gives)\s+the\s+"
            r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)\s+to\s+(?P<receiver>[A-Z][a-z]+)",
            prompt,
        )
        if handoff_match:
            directives.append(
                self._directive(
                    prompt,
                    evidence,
                    action="handoff",
                    actor_id=actor_ids[handoff_match.group("giver")],
                    prop_id=prop_ids[handoff_match.group("prop").lower()],
                    receiver_id=actor_ids[handoff_match.group("receiver")],
                    evidence_label="handoff",
                    span=(handoff_match.start(), handoff_match.end()),
                )
            )
        elif directives and directives[0].receiver_id and "returns" in prompt:
            first = directives[0]
            directives.append(
                self._directive(
                    prompt,
                    evidence,
                    action="handoff",
                    actor_id=first.actor_id,
                    prop_id=first.prop_id,
                    receiver_id=first.receiver_id,
                    evidence_label="implied handoff",
                    span=(0, min(len(prompt), prompt.find("returns") if "returns" in prompt else len(prompt))),
                )
            )

        return_match = re.search(
            r"\b(?P<actor>[A-Z][a-z]+)\s+returns\s+the\s+"
            r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)\s+to\s+(?P<receiver>[A-Z][a-z]+)",
            prompt,
        )
        if return_match:
            directives.append(
                self._directive(
                    prompt,
                    evidence,
                    action="return",
                    actor_id=actor_ids[return_match.group("actor")],
                    prop_id=prop_ids[return_match.group("prop").lower()],
                    receiver_id=actor_ids[return_match.group("receiver")],
                    evidence_label="return",
                    span=(return_match.start(), return_match.end()),
                )
            )

        place_match = re.search(
            r"\b(?P<actor>[A-Z][a-z]+)\s+places\s+the\s+"
            r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)",
            prompt,
        )
        if place_match:
            directives.append(
                self._directive(
                    prompt,
                    evidence,
                    action="place",
                    actor_id=actor_ids[place_match.group("actor")],
                    prop_id=prop_ids[place_match.group("prop").lower()],
                    evidence_label="place",
                    span=(place_match.start(), place_match.end()),
                )
            )

        uncertainties = [
            DirectorUncertainty(
                id="unc_visual_style",
                description="Prompt does not fully specify actor appearance, exact object scale, or room dressing.",
                severity="soft",
                resolved=False,
            )
        ]
        return PromptInterpretation(
            request=request,
            entities=actors + props,
            directives=directives,
            evidence=evidence,
            uncertainties=uncertainties,
        )

    def _actors(self, prompt: str) -> list[DirectorEntity]:
        seen: list[str] = []
        for match in re.finditer(r"\b[A-Z][a-z]+\b", prompt):
            label = match.group(0)
            if label in self._ACTOR_LABELS and label not in seen:
                seen.append(label)
        return [
            DirectorEntity(
                id=f"actor_{chr(ord('a') + index)}",
                kind="actor",
                role="participant",
                label=label,
            )
            for index, label in enumerate(seen)
        ]

    def _props(self, prompt: str) -> list[DirectorEntity]:
        seen: dict[str, DirectorEntity] = {}
        for match in self._PROP_RE.finditer(prompt):
            label = match.group(0).lower()
            prop_id = label.replace(" ", "_")
            if prop_id not in seen:
                seen[prop_id] = DirectorEntity(
                    id=prop_id,
                    kind="prop",
                    role="target_object",
                    label=label,
                )
        return list(seen.values())

    @staticmethod
    def _find_span(prompt: str, label: str) -> tuple[int, int] | None:
        index = prompt.lower().find(label.lower())
        if index < 0:
            return None
        return (index, index + len(label))

    @staticmethod
    def _find_regex_span(prompt: str, pattern: str) -> tuple[int, int]:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if not match:
            return (0, len(prompt))
        return (match.start(), match.end())

    @staticmethod
    def _directive(
        prompt: str,
        evidence: list[DirectorDecisionEvidence],
        *,
        action: DirectiveAction,
        actor_id: str | None,
        prop_id: str | None,
        evidence_label: str,
        span: tuple[int, int],
        receiver_id: str | None = None,
        concurrency_group: str | None = None,
    ) -> DirectorActionDirective:
        suffix = len([item for item in evidence if item.id.startswith(f"ev_action_{action}")]) + 1
        evidence_id = f"ev_action_{action}_{suffix:02d}"
        claim = prompt[span[0] : span[1]].strip()
        evidence.append(
            DirectorDecisionEvidence(
                id=evidence_id,
                source="prompt",
                prompt_span=span,
                claim=f"{evidence_label}: {claim}",
            )
        )
        directive_id = "_".join(part for part in (action, actor_id, receiver_id, prop_id) if part)
        return DirectorActionDirective(
            id=directive_id,
            action=action,
            actor_id=actor_id,
            prop_id=prop_id,
            receiver_id=receiver_id,
            concurrency_group=concurrency_group,
            evidence_id=evidence_id,
        )
