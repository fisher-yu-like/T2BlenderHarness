from __future__ import annotations

import pytest


def test_deterministic_fallback_handles_out_of_vocabulary_entities():
    from videoact.director_contracts import DirectorRequest
    from videoact.director_prompt import DeterministicPromptInterpreter

    request = DirectorRequest(
        prompt="Priya hands the ceramic mug to Wei.",
        scene_id="generic-entities",
        duration_s=8.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )

    interpretation = DeterministicPromptInterpreter().interpret(request)

    assert {entity.label for entity in interpretation.entities} >= {"Priya", "Wei", "ceramic mug"}
    assert any(directive.action == "handoff" for directive in interpretation.directives)


def test_structured_interpreter_rejects_provider_fake_evidence_span():
    from videoact.director_prompt_llm import StructuredPromptInterpreter
    from videoact.director_contracts import DirectorRequest

    request = DirectorRequest(
        prompt="Priya hands the ceramic mug to Wei.",
        scene_id="evidence-span",
        duration_s=8.0,
        fps=24,
        provider="test-provider",
        policy="director-v1",
    )

    def provider(_request):
        return {
            "entities": [],
            "directives": [],
            "evidence": [
                {
                    "id": "ev_bad",
                    "source": "prompt",
                    "prompt_span": [0, 5],
                    "quoted_text": "Wei",
                    "claim": "the provider's fabricated span",
                }
            ],
        }

    with pytest.raises(ValueError, match="evidence span"):
        StructuredPromptInterpreter(provider).interpret(request)


def test_structured_interpreter_rejects_provider_request_override():
    from videoact.director_prompt_llm import StructuredPromptInterpreter
    from videoact.director_contracts import DirectorRequest

    request = DirectorRequest(
        prompt="Alice carries the red cup.",
        scene_id="request-integrity",
        duration_s=8.0,
        fps=24,
        provider="test-provider",
        policy="director-v1",
    )

    def provider(_request):
        return {
            "request": request.model_copy(update={"prompt": "different prompt"}).model_dump(mode="json"),
            "entities": [],
            "directives": [],
            "evidence": [],
        }

    with pytest.raises(ValueError, match="must not override request"):
        StructuredPromptInterpreter(provider).interpret(request)
