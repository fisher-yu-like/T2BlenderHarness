"""Schema-constrained prompt interpretation boundary.

The provider is injectable so a real structured-output model can be used in a
configured environment while tests and offline runs remain deterministic. All
provider claims are validated against the existing PromptInterpretation
contract, including exact prompt evidence spans.
"""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any, Callable

from .director_contracts import DirectorRequest
from .director_prompt import PromptInterpretation


def _repair_model_evidence_spans(
    payload: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """Repair only small, unambiguous evidence boundary errors.

    GLM occasionally returns the right quoted text with an inclusive end
    offset, includes a tiny suffix, or shifts both boundaries by one character. The quoted text is
    still model output; this helper only converts it to the contract's
    half-open range when it occurs exactly once in the prompt and the returned
    span is within one character of that occurrence. A quote with at most two
    extra characters after an otherwise unique half-open span is trimmed only
    when the overrun starts with whitespace. Ambiguous, missing, or
    otherwise semantic errors remain fail-closed for normal validation.
    """

    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return payload
    updated_evidence: list[Any] = []
    changed = False
    for raw_item in evidence:
        if not isinstance(raw_item, Mapping):
            updated_evidence.append(raw_item)
            continue
        item = dict(raw_item)
        span = item.get("prompt_span")
        quoted = item.get("quoted_text")
        if (
            isinstance(span, (list, tuple))
            and len(span) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
            and isinstance(quoted, str)
            and quoted
        ):
            start, end = span
            if (
                0 <= start <= end < len(prompt)
                and prompt[start : end + 1].casefold() == quoted.casefold()
            ):
                item["prompt_span"] = [start, end + 1]
                changed = True
            else:
                prompt_folded = prompt.casefold()
                quoted_folded = quoted.casefold()
                span_candidate = prompt[start:end] if 0 <= start <= end <= len(prompt) else ""
                extra_suffix = quoted[len(span_candidate):] if span_candidate else ""
                if (
                    span_candidate
                    and quoted_folded.startswith(span_candidate.casefold())
                    and 0 < len(extra_suffix) <= 2
                    and extra_suffix[0].isspace()
                    and prompt_folded.count(span_candidate.casefold()) == 1
                ):
                    item["quoted_text"] = span_candidate
                    item["prompt_span"] = [start, end]
                    changed = True
                    updated_evidence.append(item)
                    continue
                exact_start = prompt_folded.find(quoted_folded)
                exact_end = exact_start + len(quoted)
                unique_match = (
                    exact_start >= 0
                    and prompt_folded.find(quoted_folded, exact_start + 1) < 0
                )
                if (
                    unique_match
                    and 0 <= start <= end <= len(prompt)
                    and abs(start - exact_start) <= 1
                    and abs(end - exact_end) <= 1
                ):
                    item["prompt_span"] = [exact_start, exact_end]
                    changed = True
        updated_evidence.append(item)
    if not changed:
        return payload
    result = dict(payload)
    result["evidence"] = updated_evidence
    return result


def _repair_grounded_camera_only_interpretation(
    payload: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """Recover a camera-only target when the prompt names it unambiguously.

    A small set of VBench camera prompts puts a standalone subject after the
    camera sentence, for example ``"The camera orbits ... . Watch."``.  GLM
    can then return an empty interpretation or retain ``unc-orbit-target``
    even though the final noun is an exact prompt span.  This repair is a
    semantic boundary fix: it copies the grounded noun and camera phrase from
    the prompt into the typed interpretation.  It does not choose geometry,
    motion parameters, Blender code, or a score, and it remains fail-closed
    when no exact subject span is available.
    """

    camera_match = re.search(
        r"\b(?:the\s+)?camera\s+(?:[a-z]+\s+){0,3}"
        r"(?P<action>orbit(?:s|ed|ing)?|pan(?:s|ned|ning)?|tilt(?:s|ed|ing)?|"
        r"zoom(?:s|ed|ing)?|doll(?:y|ies|ied|ying))\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if camera_match is None:
        return payload

    # Use only a final sentence fragment that starts with a capitalized word.
    # This deliberately avoids guessing a target from arbitrary adjectives or
    # from an imperative embedded in the camera sentence.
    target_match = re.search(
        r"(?:^|[.!?]\s+)(?P<label>[A-Z][A-Za-z0-9-]*(?:\s+[a-z][A-Za-z0-9-]*){0,4})"
        r"[.!?]?\s*$",
        prompt,
    )
    if target_match is None:
        return payload
    label = target_match.group("label").strip()
    if not label or label.casefold() in {"the camera", "watch this", "look here"}:
        return payload

    result = dict(payload)
    entities = [dict(item) for item in (payload.get("entities") or []) if isinstance(item, Mapping)]
    evidence = [dict(item) for item in (payload.get("evidence") or []) if isinstance(item, Mapping)]
    camera_cues = [dict(item) for item in (payload.get("camera_cues") or []) if isinstance(item, Mapping)]
    uncertainties = [dict(item) for item in (payload.get("uncertainties") or []) if isinstance(item, Mapping)]

    def unique_id(prefix: str, existing: set[str]) -> str:
        index = 1
        candidate = f"{prefix}_{index:02d}"
        while candidate in existing:
            index += 1
            candidate = f"{prefix}_{index:02d}"
        existing.add(candidate)
        return candidate

    entity_by_label = {
        str(item.get("label") or "").strip().casefold(): item
        for item in entities
        if str(item.get("label") or "").strip()
    }
    entity = entity_by_label.get(label.casefold())
    entity_ids = {str(item.get("id") or "") for item in entities}
    if entity is None:
        entity_id = unique_id(
            re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_") or "camera_target",
            entity_ids,
        )
        entity = {
            "id": entity_id,
            "kind": "prop",
            "role": "visible camera-orbit target grounded in the prompt",
            "label": label,
        }
        entities.append(entity)
    entity_id = str(entity.get("id") or "").strip()
    if not entity_id:
        return payload

    evidence_ids = {str(item.get("id") or "") for item in evidence}
    target_evidence_id = next(
        (
            str(item.get("id"))
            for item in evidence
            if str(item.get("claim") or "").casefold().find(label.casefold()) >= 0
        ),
        None,
    )
    if target_evidence_id is None:
        target_evidence_id = unique_id("ev_camera_target", evidence_ids)
        evidence.append(
            {
                "id": target_evidence_id,
                "source": "prompt",
                "prompt_span": [target_match.start("label"), target_match.end("label")],
                "quoted_text": prompt[target_match.start("label") : target_match.end("label")],
                "claim": f"{label} is the visible subject for the camera cue.",
            }
        )

    camera_action = camera_match.group("action").casefold()
    action = "dolly" if camera_action.startswith("doll") else camera_action.rstrip("s")
    direction_match = re.search(r"\b(clockwise|counterclockwise|anticlockwise)\b", prompt, re.IGNORECASE)
    direction = direction_match.group(1).lower() if direction_match else None
    camera_evidence_ids = {str(item.get("id") or "") for item in evidence}
    camera_evidence_id = unique_id("ev_camera_cue", camera_evidence_ids)
    camera_phrase_end = camera_match.end("action")
    around_match = re.match(r"\s+around\b", prompt[camera_phrase_end:], flags=re.IGNORECASE)
    if around_match:
        camera_phrase_end += around_match.end()
    evidence.append(
        {
            "id": camera_evidence_id,
            "source": "prompt",
            "prompt_span": [camera_match.start("action"), camera_phrase_end],
            "quoted_text": prompt[camera_match.start("action") : camera_phrase_end],
            "claim": f"The camera performs a {action} cue grounded in the prompt.",
        }
    )

    if not camera_cues:
        camera_cues.append(
            {
                "id": unique_id("camera_cue", {str(item.get("id") or "") for item in camera_cues}),
                "action": action,
                "direction": direction,
                "evidence_id": camera_evidence_id,
            }
        )
    else:
        # Preserve the model's cue details when present, but guarantee that a
        # repair-created exact evidence span is available to the cue.
        cue = camera_cues[0]
        cue["action"] = cue.get("action") or action
        cue["direction"] = cue.get("direction") or direction
        cue["evidence_id"] = cue.get("evidence_id") or camera_evidence_id

    uncertainty = next(
        (item for item in uncertainties if str(item.get("id") or "") == "unc-orbit-target"),
        None,
    )
    if uncertainty is None:
        uncertainties.append(
            {
                "id": "unc-orbit-target",
                "description": "The exact final-sentence subject is used as the camera cue target.",
                "severity": "soft",
                "resolved": True,
            }
        )
    else:
        uncertainty["resolved"] = True
        if str(uncertainty.get("severity") or "") == "hard":
            uncertainty["severity"] = "soft"

    result.update(
        {
            "entities": entities,
            "camera_cues": camera_cues,
            "evidence": evidence,
            "uncertainties": uncertainties,
        }
    )
    return result


class StructuredPromptInterpreter:
    def __init__(self, provider: Callable[[DirectorRequest], dict[str, Any]]) -> None:
        self.provider = provider

    @staticmethod
    def build_request(request: DirectorRequest) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "obligations": request.obligations,
            "response_schema": PromptInterpretation.model_json_schema(),
            "evidence_rule": "prompt_span and quoted_text must be an exact substring of prompt",
        }

    def interpret(self, request: DirectorRequest) -> PromptInterpretation:
        payload = self.provider(request)
        if not isinstance(payload, dict):
            raise ValueError("structured prompt provider must return an object")
        if "request" in payload:
            raise ValueError("structured prompt provider must not override request")
        payload = _repair_model_evidence_spans(payload, request.prompt)
        payload = _repair_grounded_camera_only_interpretation(payload, request.prompt)
        return PromptInterpretation.model_validate(
            {"request": request.model_dump(mode="json"), **payload}
        )
