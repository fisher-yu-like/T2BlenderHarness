import json

import pytest


def test_real_run_state_machine_requires_ordered_transitions(tmp_path):
    from videoact.real_pipeline import RealRunStateMachine

    machine = RealRunStateMachine(tmp_path, case_id="case-01")
    assert machine.state == "prepared"

    machine.transition("executing", {"request_id": "mcp-1"})
    machine.transition("rendered", {"blender_version": "4.3.0"})
    machine.transition("artifact_valid", {"readable_frames": 3})
    machine.transition("evaluated", {"deterministic_score": 92})

    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["state"] == "evaluated"
    assert [entry["state"] for entry in saved["history"]] == [
        "prepared",
        "executing",
        "rendered",
        "artifact_valid",
        "evaluated",
    ]


def test_real_run_state_machine_rejects_skipping_artifact_gate(tmp_path):
    from videoact.real_pipeline import RealRunStateMachine

    machine = RealRunStateMachine(tmp_path, case_id="case-01")

    with pytest.raises(ValueError, match="invalid transition"):
        machine.transition("evaluated")


def test_mcp_response_failure_is_recorded_as_terminal_failed(tmp_path):
    from videoact.real_pipeline import RealRunStateMachine

    machine = RealRunStateMachine(tmp_path, case_id="case-01")
    machine.transition("executing")
    machine.record_mcp_response({"isError": True, "message": "Blender error"})

    assert machine.state == "failed"
    response = json.loads((tmp_path / "mcp_response.json").read_text(encoding="utf-8"))
    assert response["isError"] is True
