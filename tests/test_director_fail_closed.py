from __future__ import annotations

import pytest


def _request(prompt: str):
    from videoact.director_contracts import DirectorRequest

    return DirectorRequest(
        prompt=prompt,
        scene_id="fail-closed-test",
        duration_s=10.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )


def test_deterministic_interpreter_marks_uncovered_actions_as_hard() -> None:
    from videoact.director_prompt import DeterministicPromptInterpreter

    interpretation = DeterministicPromptInterpreter().interpret(
        _request("Alice walks to the table, lifts the red cup, and looks toward the window.")
    )

    hard = {item.id for item in interpretation.uncertainties if item.severity == "hard" and not item.resolved}
    assert "unc_uncovered_prompt_action" in hard


def test_director_agent_rejects_uncovered_actions_before_projection() -> None:
    from videoact.director import DirectorAgent

    with pytest.raises(ValueError, match="unresolved hard uncertainty"):
        DirectorAgent().plan(
            "Alice walks to the table, lifts the red cup, and looks toward the window.",
            scene_id="uncovered-action",
            duration_s=10.0,
            fps=24,
        )


def test_reveal_is_preserved_as_an_explicit_event() -> None:
    from videoact.director import DirectorAgent

    result = DirectorAgent().plan(
        "Alice reveals the green book, then carries the green book and hands the green book to Carla.",
        scene_id="explicit-reveal",
        duration_s=12.0,
        fps=24,
    )

    assert [event.action for event in result.director_plan.events] == ["reveal", "carry", "handoff"]


def test_dynamic_interpreter_cannot_return_an_empty_event_graph() -> None:
    from videoact.director import DirectorAgent
    from videoact.director_prompt import DeterministicPromptInterpreter

    class EmptyInterpreter:
        def interpret(self, request):
            interpretation = DeterministicPromptInterpreter().interpret(
                request.model_copy(update={"prompt": "Alice carries the red cup."})
            )
            return interpretation.model_copy(
                update={"request": request, "directives": [], "uncertainties": []}
            )

    with pytest.raises(ValueError, match="empty event graph"):
        DirectorAgent(interpreter=EmptyInterpreter(), provider="dynamic").plan(
            "Alice carries the red cup.",
            scene_id="empty-graph",
            duration_s=10.0,
            fps=24,
        )


def test_dynamic_mode_requires_a_non_deterministic_interpreter() -> None:
    from videoact.director import DirectorAgent

    with pytest.raises(ValueError, match="dynamic mode requires"):
        DirectorAgent(mode="dynamic")


def test_director_agent_exposes_an_explicit_dynamic_provider_factory() -> None:
    from videoact.director import DirectorAgent
    from videoact.director_prompt import DeterministicPromptInterpreter

    def provider(request):
        interpretation = DeterministicPromptInterpreter().interpret(request)
        return {
            "entities": [item.model_dump(mode="json") for item in interpretation.entities],
            "directives": [item.model_dump(mode="json") for item in interpretation.directives],
            "evidence": [item.model_dump(mode="json") for item in interpretation.evidence],
            "assumptions": [],
            "uncertainties": [item.model_dump(mode="json") for item in interpretation.uncertainties if item.severity == "soft"],
        }

    agent = DirectorAgent.from_provider(provider, provider_name="codex-local")

    assert agent.mode == "dynamic"
    assert agent.provider == "codex-local"
