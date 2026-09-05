from __future__ import annotations

import pytest

from videoact.case_coverage import validate_case_coverage
from videoact.director_contracts import DirectorPlan
from videoact.obligations import (
    ObligationCompiler,
    attach_obligation_ids,
    bind_obligations_to_plan,
    compile_obligations,
    validate_obligation_completeness,
)


def _handoff_plan() -> DirectorPlan:
    return DirectorPlan.model_validate(
        {
            "id": "plan-handoff-01",
            "request": {
                "prompt": "Alice hands the red cup to Bob and places it on the drop zone. Use an orbit camera during the handoff.",
                "scene_id": "handoff-scene-01",
                "duration_s": 6.0,
                "fps": 24,
                "provider": "fixture-provider",
                "policy": "fixture-policy",
                "obligations": {
                    "required_entity_ids": ["actor_a", "actor_b", "red_cup", "table", "drop_zone"],
                    "required_event_ids": ["approach", "grasp", "handoff", "place", "release"],
                },
            },
            "entities": [
                {"id": "actor_a", "kind": "actor", "role": "giver", "label": "Alice"},
                {"id": "actor_b", "kind": "actor", "role": "receiver", "label": "Bob"},
                {"id": "red_cup", "kind": "prop", "role": "transferred_prop", "label": "red cup"},
                {"id": "table", "kind": "support", "role": "source_support", "label": "table"},
                {"id": "drop_zone", "kind": "support", "role": "final_support", "label": "drop zone"},
            ],
            "events": [
                {"id": "approach", "action": "walk", "participant_ids": ["actor_a"], "target_ids": ["table"], "start": 0.0, "end": 1.0},
                {"id": "grasp", "action": "grasp", "participant_ids": ["actor_a"], "target_ids": ["red_cup"], "depends_on": ["approach"], "start": 1.0, "end": 2.0},
                {"id": "handoff", "action": "handoff", "participant_ids": ["actor_a", "actor_b"], "target_ids": ["red_cup"], "depends_on": ["grasp"], "start": 2.0, "end": 3.0},
                {"id": "place", "action": "place", "participant_ids": ["actor_b"], "target_ids": ["red_cup", "drop_zone"], "depends_on": ["handoff"], "start": 3.0, "end": 4.0},
                {"id": "release", "action": "release", "participant_ids": ["actor_b"], "target_ids": ["red_cup"], "depends_on": ["place"], "start": 4.0, "end": 5.0},
            ],
            "interactions": [
                {
                    "id": "handoff-01",
                    "prop_id": "red_cup",
                    "giver_id": "actor_a",
                    "receiver_id": "actor_b",
                    "attach_event_id": "grasp",
                    "transfer_event_id": "handoff",
                    "detach_event_id": "release",
                    "final_owner_id": "actor_b",
                    "final_support_id": "drop_zone",
                }
            ],
            "camera_plan": {
                "shots": [
                    {
                        "shot_id": "shot_handoff",
                        "start_frame": 49,
                        "end_frame": 73,
                        "target_ids": ["actor_a", "actor_b", "red_cup"],
                        "intent": "three-way handoff coverage",
                        "lens_mm": 50.0,
                        "distance_range": [4.0, 8.0],
                        "required_event_ids": ["handoff"],
                        "trajectory_type": "orbit",
                        "visibility_predicates": {"actor_a": "visible", "actor_b": "visible", "red_cup": "visible"},
                        "camera_cue": "orbit",
                    }
                ]
            },
            "provider_fingerprint": "provider-hash",
            "policy_fingerprint": "policy-hash",
        }
    )


def _handoff_record() -> dict:
    return {
        "case_id": "handoff-case-01",
        "prompt": "Alice hands the red cup to Bob and places it on the drop zone. Use an orbit camera during the handoff.",
        "oracle_expectations": {
            "required_entity_ids": ["actor_a", "actor_b", "red_cup", "table", "drop_zone"],
            "event_order": ["approach", "grasp", "handoff", "place", "release"],
            "required_camera_events": ["handoff"],
        },
        "required_artifacts": ["candidate.blend", "proxy.mp4", "telemetry.json"],
    }


def test_handoff_compiles_ownership_window_and_three_way_camera_obligations() -> None:
    plan = _handoff_plan()
    first = compile_obligations(record=_handoff_record(), director_plan=plan)
    second = ObligationCompiler().compile(_handoff_record(), plan)

    assert first.obligation_ids == second.obligation_ids
    assert len(first.obligation_ids) == len(set(first.obligation_ids))
    assert {item.case_id for item in first.obligations} == {"handoff-case-01"}

    ownership = [item for item in first.obligations if item.kind == "ownership_transition"]
    assert ownership
    assert any(item.expected["giver_id"] == "actor_a" and item.expected["receiver_id"] == "actor_b" for item in ownership)
    assert any(item.expected["final_owner_id"] == "actor_b" for item in ownership)
    assert any(item.kind == "transfer_window" and item.expected["event_id"] == "handoff" for item in first.obligations)
    assert any(
        item.kind == "camera_coverage"
        and set(item.expected["target_ids"]) == {"actor_a", "actor_b", "red_cup"}
        for item in first.obligations
    )
    assert all(item.required == item.applicable for item in first.obligations if not item.applicable)


def test_object_only_does_not_invent_character_obligations() -> None:
    plan = DirectorPlan.model_validate(
        {
            "id": "plan-object-01",
            "request": {
                "prompt": "A red ball rolls across a table while the static camera holds.",
                "scene_id": "object-scene-01",
                "duration_s": 4.0,
                "fps": 24,
                "provider": "fixture-provider",
                "policy": "fixture-policy",
            },
            "entities": [
                {"id": "red_ball", "kind": "prop", "role": "moving_object", "label": "red ball"},
                {"id": "table", "kind": "support", "role": "support", "label": "table"},
            ],
            "events": [
                {"id": "roll", "action": "roll", "participant_ids": ["red_ball"], "target_ids": ["table"], "start": 0.0, "end": 3.0}
            ],
            "camera_plan": {"shots": []},
            "provider_fingerprint": "provider-hash",
            "policy_fingerprint": "policy-hash",
        }
    )
    compilation = compile_obligations(
        record={"case_id": "object-case-01", "prompt": plan.request.prompt},
        director_plan=plan,
    )

    assert any(item.kind == "trajectory" and item.expected["entity_id"] == "red_ball" for item in compilation.obligations)
    assert not any(item.expected.get("entity_kind") == "actor" for item in compilation.obligations if isinstance(item.expected, dict))
    assert not any(item.kind == "participant" and item.expected.get("entity_id") == "character" for item in compilation.obligations)


def test_static_camera_does_not_create_camera_innovation_obligation() -> None:
    plan = _handoff_plan().model_copy(
        deep=True,
        update={
            "request": _handoff_plan().request.model_copy(
                update={"prompt": "A red cup rests on a table; the static camera holds the composition."}
            ),
            "camera_plan": {"shots": []},
        },
    )
    compilation = compile_obligations(
        record={"case_id": "static-case-01", "prompt": plan.request.prompt},
        director_plan=plan,
    )

    assert not any(item.kind in {"camera_innovation", "camera_motion"} and item.applicable for item in compilation.obligations)
    assert not any(item.kind == "camera_innovation" for item in compilation.obligations)


def test_required_obligation_deletion_fails_closed_and_ids_trace_across_boundaries() -> None:
    plan = _handoff_plan()
    compilation = compile_obligations(record=_handoff_record(), director_plan=plan)
    removed = [item for item in compilation.obligations if item.kind != "ownership_transition"]

    with pytest.raises(ValueError, match="missing required obligation"):
        validate_obligation_completeness(removed, expected_ids=compilation.obligation_ids)

    bound_plan = bind_obligations_to_plan(plan, compilation)
    assert bound_plan.obligation_ids == compilation.obligation_ids

    coverage = validate_case_coverage(
        record=_handoff_record(),
        director_plan=bound_plan,
        director_trajectories={"entities": {"actor_a": {}, "actor_b": {}, "red_cup": {}, "table": {}, "drop_zone": {}}},
        director_camera={"shots": [{"shot_id": "shot_handoff", "required_event_ids": ["handoff"]}]},
        generated_code="actor_a actor_b red_cup table drop_zone approach grasp handoff place release",
        obligations=compilation,
    )
    assert coverage.obligation_ids == compilation.obligation_ids
    assert coverage.covered_obligation_ids == compilation.obligation_ids

    telemetry = attach_obligation_ids({"status": "pass"}, compilation, stage="trusted_observer")
    evaluation = attach_obligation_ids({"status": "scored"}, compilation, stage="evaluator")
    assert telemetry["obligation_ids"] == compilation.obligation_ids
    assert evaluation["obligation_ids"] == compilation.obligation_ids
    assert telemetry["obligation_trace"]["stage"] == "trusted_observer"
    assert evaluation["obligation_trace"]["stage"] == "evaluator"
