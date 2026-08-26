from __future__ import annotations

import pytest
from pydantic import ValidationError


def _valid_director_plan():
    from videoact.director_contracts import (
        DirectorAssumption,
        DirectorDecisionEvidence,
        DirectorEntity,
        DirectorEvent,
        DirectorPlan,
        DirectorRequest,
        DirectorUncertainty,
        InteractionLifecycle,
    )

    request = DirectorRequest(
        prompt=(
            "Alice carries the red cube while Bob carries the blue cube, "
            "then Alice hands the red cube to Bob."
        ),
        scene_id="multi-contract-01",
        duration_s=8.0,
        fps=24,
        provider="deterministic",
        policy="director-v1",
    )
    evidence = [
        DirectorDecisionEvidence(
            id="ev_alice",
            source="prompt",
            prompt_span=(0, 5),
            claim="Alice is an actor.",
        ),
        DirectorDecisionEvidence(
            id="ev_bob",
            source="prompt",
            prompt_span=(33, 36),
            claim="Bob is an actor.",
        ),
        DirectorDecisionEvidence(
            id="ev_handoff",
            source="prompt",
            prompt_span=(64, 94),
            claim="Alice transfers the red cube to Bob.",
        ),
    ]
    return DirectorPlan(
        id="director-plan-contract-01",
        request=request,
        entities=[
            DirectorEntity(id="actor_a", kind="actor", role="giver", label="Alice"),
            DirectorEntity(id="actor_b", kind="actor", role="receiver", label="Bob"),
            DirectorEntity(id="red_cube", kind="prop", role="handoff_prop", label="red cube"),
            DirectorEntity(id="blue_cube", kind="prop", role="carried_prop", label="blue cube"),
        ],
        events=[
            DirectorEvent(
                id="carry_red",
                action="carry",
                participant_ids=["actor_a"],
                target_ids=["red_cube"],
                concurrency_group="parallel_carry",
                start=0.5,
                end=3.5,
            ),
            DirectorEvent(
                id="carry_blue",
                action="carry",
                participant_ids=["actor_b"],
                target_ids=["blue_cube"],
                concurrency_group="parallel_carry",
                start=0.5,
                end=3.5,
            ),
            DirectorEvent(
                id="handoff_red",
                action="handoff",
                participant_ids=["actor_a", "actor_b"],
                target_ids=["red_cube"],
                depends_on=["carry_red", "carry_blue"],
                start=3.5,
                end=5.0,
            ),
            DirectorEvent(
                id="place_red",
                action="place",
                participant_ids=["actor_b"],
                target_ids=["red_cube"],
                depends_on=["handoff_red"],
                start=5.0,
                end=7.0,
            ),
        ],
        interactions=[
            InteractionLifecycle(
                id="red_cube_handoff",
                prop_id="red_cube",
                giver_id="actor_a",
                receiver_id="actor_b",
                attach_event_id="carry_red",
                transfer_event_id="handoff_red",
                detach_event_id="place_red",
                final_owner_id="actor_b",
                final_support_id="table",
            )
        ],
        assumptions=[
            DirectorAssumption(
                id="assume_table",
                statement="A neutral table may be added as the final support.",
                supported_by_evidence_ids=["ev_handoff"],
            )
        ],
        uncertainties=[
            DirectorUncertainty(
                id="unc_actor_style",
                description="Prompt does not specify actor costume detail.",
                severity="soft",
                resolved=False,
            )
        ],
        evidence=evidence,
        provider_fingerprint="provider:deterministic",
        policy_fingerprint="policy:director-v1",
    )


def test_director_plan_accepts_two_actors_two_props_concurrency_handoff_and_fingerprints():
    from videoact.director_contracts import DirectorResult

    plan = _valid_director_plan()
    first_hash = plan.content_hash()
    second_hash = plan.model_copy(deep=True).content_hash()

    assert plan.request.prompt.startswith("Alice carries")
    assert plan.entity_ids == {"actor_a", "actor_b", "red_cube", "blue_cube"}
    assert plan.event_ids == {"carry_red", "carry_blue", "handoff_red", "place_red"}
    assert plan.events[0].concurrency_group == "parallel_carry"
    assert plan.interactions[0].final_owner_id == "actor_b"
    assert len(first_hash) == 64
    assert first_hash == second_hash

    result = DirectorResult(director_plan=plan)
    assert result.scene_id == "multi-contract-01"
    assert result.director_plan_hash == first_hash


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda plan: setattr(plan.events[0], "participant_ids", ["missing_actor"]),
            "unknown participant",
        ),
        (
            lambda plan: plan.entities.append(plan.entities[0].model_copy()),
            "entity IDs must be unique",
        ),
        (
            lambda plan: setattr(plan.assumptions[0], "supported_by_evidence_ids", ["missing_evidence"]),
            "unsupported assumption",
        ),
        (
            lambda plan: setattr(plan.events[0], "depends_on", ["place_red"]),
            "dependency cycle",
        ),
        (
            lambda plan: setattr(plan.interactions[0], "final_owner_id", "red_cube"),
            "final owner",
        ),
    ],
)
def test_director_plan_rejects_invalid_references_and_graphs(mutate, message: str):
    from videoact.director_contracts import DirectorPlan

    plan = _valid_director_plan()
    mutate(plan)

    with pytest.raises(ValidationError, match=message):
        DirectorPlan.model_validate(plan.model_dump())
