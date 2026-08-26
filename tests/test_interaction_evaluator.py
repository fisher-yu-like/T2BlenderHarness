from __future__ import annotations


def _result():
    from videoact.director import DirectorAgent

    return DirectorAgent().plan(
        "Alice carries the red cube, then Alice hands the red cube to Bob, then Bob places the red cube.",
        scene_id="interaction-eval",
        duration_s=12.0,
        fps=24,
    )


def test_interaction_evaluator_accepts_complete_handoff_lifecycle():
    from evaluator.interaction_metrics import evaluate_interactions

    result = _result()
    findings = evaluate_interactions(result.director_plan, result.trajectory_plan)

    assert findings == []


def test_interaction_evaluator_accepts_independent_carry_without_transfer():
    from evaluator.interaction_metrics import evaluate_interactions
    from videoact.director import DirectorAgent

    result = DirectorAgent().plan(
        "Alice carries the red cube while Bob carries the blue cup, then Alice hands the red cube to Bob and Bob places the red cube.",
        scene_id="independent-carry",
        duration_s=20.0,
        fps=24,
    )

    findings = evaluate_interactions(result.director_plan, result.trajectory_plan)

    assert findings == []


def test_interaction_evaluator_flags_incomplete_contact_with_precise_root():
    from evaluator.interaction_metrics import evaluate_interactions

    result = _result()
    prop = result.trajectory_plan.entities["red_cube"]
    broken_prop = prop.model_copy(
        update={
            "attachment_events": [
                event for event in prop.attachment_events if event.action != "transfer"
            ]
        }
    )
    broken_plan = result.trajectory_plan.model_copy(
        update={"entities": {**result.trajectory_plan.entities, "red_cube": broken_prop}}
    )

    findings = evaluate_interactions(result.director_plan, broken_plan)
    finding = next(item for item in findings if item.failure_id == "interaction_handoff_incomplete")

    assert finding.owner == "director_trajectory"
    assert finding.root_cause_id.startswith("attachment_lifecycle:actor_b:red_cube:")


def test_interaction_evaluator_flags_wrong_final_owner_from_telemetry():
    from evaluator.interaction_metrics import evaluate_interactions

    result = _result()
    transfer_id = result.director_plan.interactions[0].transfer_event_id
    telemetry = {"current_owner_by_event": {f"{transfer_id}:red_cube": "actor_a"}}

    findings = evaluate_interactions(result.director_plan, result.trajectory_plan, telemetry=telemetry)
    finding = next(item for item in findings if item.failure_id == "interaction_final_owner_mismatch")

    assert finding.owner == "director_trajectory"
    assert "actor_b" in finding.evidence
    assert "actor_a" in finding.evidence
