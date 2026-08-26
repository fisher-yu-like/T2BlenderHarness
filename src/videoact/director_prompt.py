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
            else:
                reveal_match = re.search(
                    r"\b(?P<actor>[A-Z][a-z]+)\s+reveals\s+the\s+"
                    r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)\b",
                    prompt,
                )
                implied_carry = None
                if reveal_match:
                    suffix = prompt[reveal_match.end() :]
                    suffix_match = re.search(
                        r"\bthen\s+carries\s+the\s+(?P<prop>(?:red|blue|green|yellow)\s+\w+)\b",
                        suffix,
                    )
                    if suffix_match:
                        implied_carry = (suffix_match, reveal_match.end())
                if reveal_match and implied_carry:
                    carry_match, carry_offset = implied_carry
                    directives.append(
                        self._directive(
                            prompt,
                            evidence,
                            action="carry",
                            actor_id=actor_ids[reveal_match.group("actor")],
                            prop_id=prop_ids[carry_match.group("prop").lower()],
                            evidence_label="implied carry after reveal",
                            span=(reveal_match.start(), carry_offset + carry_match.end()),
                        )
                    )

        if re.search(r"\bpauses?\b", prompt, re.IGNORECASE):
            span = self._find_regex_span(prompt, r"\bpauses?\b")
            last = directives[-1] if directives else None
            pause_actor_match = re.search(
                r"\b(?P<actor>[A-Z][a-z]+)\s+pauses?\b",
                prompt,
            )
            directives.append(
                self._directive(
                    prompt,
                    evidence,
                    action="pause",
                    actor_id=(
                        actor_ids.get(pause_actor_match.group("actor"))
                        if pause_actor_match
                        else last.actor_id
                        if last
                        else (actors[0].id if actors else None)
                    ),
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
        else:
            implied_handoff = re.search(
                r"\b(?:and\s+)?hands\s+the\s+(?P<prop>(?:red|blue|green|yellow)\s+\w+)\s+"
                r"to\s+(?P<receiver>[A-Z][a-z]+)",
                prompt,
            )
            if implied_handoff:
                prop_id = prop_ids[implied_handoff.group("prop").lower()]
                giver = next(
                    (
                        directive.actor_id
                        for directive in reversed(directives)
                        if directive.prop_id == prop_id and directive.action in {"carry", "handoff"}
                    ),
                    None,
                )
                if giver is None:
                    prior_actor_matches = list(
                        re.finditer(r"\b[A-Z][a-z]+\b", prompt[: implied_handoff.start()])
                    )
                    prior_actor = next(
                        (
                            match.group(0)
                            for match in reversed(prior_actor_matches)
                            if match.group(0) in actor_ids
                        ),
                        None,
                    )
                    giver = actor_ids.get(prior_actor or "")
                receiver = actor_ids.get(implied_handoff.group("receiver"))
                if giver and receiver:
                    directives.append(
                        self._directive(
                            prompt,
                            evidence,
                            action="handoff",
                            actor_id=giver,
                            prop_id=prop_id,
                            receiver_id=receiver,
                            evidence_label="implied handoff",
                            span=(implied_handoff.start(), implied_handoff.end()),
                        )
                    )
            elif directives and directives[0].receiver_id and "returns" in prompt:
                first = directives[0]
                return_anchor = re.search(r"\b[A-Z][a-z]+\s+returns?\b", prompt)
                inferred_start = return_anchor.start() if return_anchor else 0
                directives.append(
                    self._directive(
                        prompt,
                        evidence,
                        action="handoff",
                        actor_id=first.actor_id,
                        prop_id=first.prop_id,
                        receiver_id=first.receiver_id,
                        evidence_label="implied handoff",
                        span=(
                            inferred_start,
                            min(len(prompt), return_anchor.end() if return_anchor else len(prompt)),
                        ),
                    )
                )

        return_match = re.search(
            r"\b(?P<actor>[A-Z][a-z]+)\s+returns\s+the\s+"
            r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)\s+to\s+(?P<receiver>[A-Z][a-z]+)",
            prompt,
        )
        if return_match is None:
            subjectless_return = re.search(
                r"\b(?:and\s+)?returns\s+the\s+"
                r"(?P<prop>(?:red|blue|green|yellow)\s+\w+)\s+to\s+(?P<receiver>[A-Z][a-z]+)",
                prompt,
            )
            if subjectless_return:
                prop_id = prop_ids[subjectless_return.group("prop").lower()]
                prior_handoff = next(
                    (
                        directive
                        for directive in reversed(directives)
                        if directive.action == "handoff" and directive.prop_id == prop_id
                    ),
                    None,
                )
                return_actor = prior_handoff.receiver_id if prior_handoff else None
                if return_actor is None:
                    prior_pause = next(
                        (
                            directive
                            for directive in reversed(directives)
                            if directive.action == "pause" and directive.prop_id == prop_id
                        ),
                        None,
                    )
                    return_actor = prior_pause.actor_id if prior_pause else None
                receiver = actor_ids.get(subjectless_return.group("receiver"))
                if return_actor and receiver:
                    return_match = subjectless_return
                    directives.append(
                        self._directive(
                            prompt,
                            evidence,
                            action="return",
                            actor_id=return_actor,
                            prop_id=prop_id,
                            receiver_id=receiver,
                            evidence_label="implied return",
                            span=(return_match.start(), return_match.end()),
                        )
                    )
        if return_match:
            if return_match.groupdict().get("actor"):
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

        evidence_by_id = {item.id: item for item in evidence}
        directives.sort(
            key=lambda directive: (
                evidence_by_id[directive.evidence_id].prompt_span[0]
                if evidence_by_id[directive.evidence_id].prompt_span is not None
                else len(prompt),
                evidence_by_id[directive.evidence_id].prompt_span[1]
                if evidence_by_id[directive.evidence_id].prompt_span is not None
                else len(prompt),
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
