from __future__ import annotations


def _request(prompt: str):
    from videoact.director_contracts import DirectorRequest

    return DirectorRequest(
        prompt=prompt,
        scene_id="prompt-test",
        duration_s=12.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )


def test_interpreter_extracts_named_actors_repeated_props_roles_and_prompt_evidence():
    from videoact.director_prompt import DeterministicPromptInterpreter

    request = _request(
        "Alice carries the red cube while Bob carries the blue cube, "
        "then Alice hands the red cube to Bob."
    )
    interpretation = DeterministicPromptInterpreter().interpret(request)

    assert {entity.id: entity.label for entity in interpretation.entities} == {
        "actor_a": "Alice",
        "actor_b": "Bob",
        "red_cube": "red cube",
        "blue_cube": "blue cube",
    }
    transfer = next(directive for directive in interpretation.directives if directive.action == "handoff")
    assert transfer.actor_id == "actor_a"
    assert transfer.receiver_id == "actor_b"
    assert transfer.prop_id == "red_cube"
    assert transfer.evidence_id in {evidence.id for evidence in interpretation.evidence}

    evidence = {item.id: item for item in interpretation.evidence}
    assert evidence[transfer.evidence_id].source == "prompt"
    assert evidence[transfer.evidence_id].prompt_span is not None
    assert "red cube" in evidence[transfer.evidence_id].claim
    assert all("position" not in directive.model_dump() for directive in interpretation.directives)
    assert all("camera" not in directive.model_dump() for directive in interpretation.directives)


def test_interpreter_marks_pause_resume_and_return_without_inventing_hard_facts():
    from videoact.director_prompt import DeterministicPromptInterpreter

    request = _request(
        "Alice takes the red cube to Bob, pauses, then Bob returns the red cube to Alice."
    )
    interpretation = DeterministicPromptInterpreter().interpret(request)

    assert [directive.action for directive in interpretation.directives] == [
        "carry",
        "pause",
        "handoff",
        "return",
    ]
    assert interpretation.directives[-1].actor_id == "actor_b"
    assert interpretation.directives[-1].receiver_id == "actor_a"
    assert interpretation.uncertainties
    assert all(uncertainty.severity == "soft" for uncertainty in interpretation.uncertainties)


def test_interpreter_resolves_elliptical_reveal_carry_and_handoff():
    from videoact.director_prompt import DeterministicPromptInterpreter

    request = _request(
        "Alice reveals the green book, then carries the green book and hands the green book to Carla; "
        "Carla places the green book while the yellow ball remains visible."
    )
    interpretation = DeterministicPromptInterpreter().interpret(request)

    assert [(item.action, item.actor_id, item.prop_id, item.receiver_id) for item in interpretation.directives] == [
        ("carry", "actor_a", "green_book", None),
        ("handoff", "actor_a", "green_book", "actor_b"),
        ("place", "actor_b", "green_book", None),
    ]
