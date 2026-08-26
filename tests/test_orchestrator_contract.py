import json

import pytest

from videoact.contracts import ExecutionResult


class SuccessAdapter:
    def __init__(self):
        self.calls = 0

    def run(self, script_path, run_dir, *, prefer="mcp", timeout_s=300):
        self.calls += 1
        return ExecutionResult(status="success", backend="fake")


def case(prompt="Observe a table."):
    return {"case_id": "case-001", "prompt": prompt, "duration_s": 10.0, "fps": 24}


def snapshot():
    return {"version": "h1", "evaluator_version": "e1"}


def test_orchestrator_exposes_ordered_stages(tmp_path):
    from videoact.orchestrator import Orchestrator

    orchestrator = Orchestrator(adapter=SuccessAdapter())
    result = orchestrator.run(case(), snapshot(), tmp_path)

    assert result.status == "success"
    assert orchestrator.stage_order == [
        "contract",
        "plan",
        "execute",
        "render",
        "evaluate",
        "repair",
        "finalize",
    ]


def test_orchestrator_validates_contract_before_execution(tmp_path):
    from videoact.orchestrator import Orchestrator

    adapter = SuccessAdapter()
    with pytest.raises(ValueError, match="prompt"):
        Orchestrator(adapter=adapter).run(case("  "), snapshot(), tmp_path)

    assert adapter.calls == 0


def test_orchestrator_resumes_existing_final_selection_without_execution(tmp_path):
    from videoact.orchestrator import Orchestrator

    adapter = SuccessAdapter()
    first = Orchestrator(adapter=adapter).run(case(), snapshot(), tmp_path)
    second = Orchestrator(adapter=adapter).run(case(), snapshot(), tmp_path, resume=True)

    assert first.status == second.status == "success"
    assert adapter.calls == 1


def test_orchestrator_fails_closed_when_resume_prompt_changes(tmp_path):
    from videoact.orchestrator import Orchestrator

    Orchestrator(adapter=SuccessAdapter()).run(case(), snapshot(), tmp_path)

    with pytest.raises(ValueError, match="resume fingerprint"):
        Orchestrator(adapter=SuccessAdapter()).run(
            case("Observe a red cup."), snapshot(), tmp_path, resume=True
        )
