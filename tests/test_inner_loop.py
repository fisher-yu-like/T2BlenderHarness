import json

from videoact.contracts import ExecutionResult


class SequenceAdapter:
    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.calls = 0

    def run(self, script_path, run_dir, *, prefer="mcp", timeout_s=300):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        return ExecutionResult(
            status=status,
            backend="fake",
            artifact_paths={"proxy": "proxy.mp4"} if status == "success" else {},
            error=None if status == "success" else "fake failure",
        )


def case():
    return {
        "case_id": "case-001",
        "prompt": "A character walks to the table, picks up the red cup, and shows the grasp closeup.",
        "duration_s": 10.0,
        "fps": 24,
    }


def snapshot():
    return {"version": "h1", "evaluator_version": "e1"}


def test_inner_loop_promotes_first_valid_candidate(tmp_path):
    from videoact.inner_loop import run_inner_loop

    result = run_inner_loop(case(), snapshot(), tmp_path, adapter=SequenceAdapter(["success"]))

    assert result.status == "success"
    assert result.selected_attempt == 1
    attempt = tmp_path / "attempts" / "01"
    for name in [
        "plan.json",
        "trajectory.json",
        "camera_plan.json",
        "blender_script.py",
        "deterministic_report.json",
        "attempt_manifest.json",
    ]:
        assert (attempt / name).exists(), name
    assert (tmp_path / "final" / "selection.json").exists()


def test_inner_loop_preserves_failed_attempt_and_records_repair_route(tmp_path):
    from videoact.inner_loop import run_inner_loop

    adapter = SequenceAdapter(["failed", "success"])
    result = run_inner_loop(case(), snapshot(), tmp_path, adapter=adapter, max_attempts=2)

    assert result.status == "success"
    assert result.selected_attempt == 2
    assert adapter.calls == 2
    report = json.loads(
        (tmp_path / "attempts" / "01" / "deterministic_report.json").read_text(encoding="utf-8")
    )
    assert any(finding["failure_id"] == "incomplete_proxy" for finding in report["findings"])
    assert (tmp_path / "attempts" / "01" / "attempt_manifest.json").exists()
    assert (tmp_path / "attempts" / "02" / "attempt_manifest.json").exists()


def test_inner_loop_exhausts_after_bounded_attempts(tmp_path):
    from videoact.inner_loop import run_inner_loop

    adapter = SequenceAdapter(["failed"])
    result = run_inner_loop(case(), snapshot(), tmp_path, adapter=adapter, max_attempts=6)

    assert result.status == "exhausted"
    assert result.selected_attempt is None
    assert adapter.calls == 6
    assert not (tmp_path / "final" / "selection.json").exists()
