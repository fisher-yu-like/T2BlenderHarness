from __future__ import annotations

import json


def test_stage_retry_preserves_upstream_and_retries_only_failed_stage(tmp_path):
    from videoact.real_inner_loop import run_stage_aware_inner_loop

    calls: list[str] = []
    counts = {"director": 0, "codegen": 0, "executor": 0, "observer": 0, "evaluator": 0}

    def director(case_id, state, attempt):
        calls.append("director")
        counts["director"] += 1
        return {"status": "success", "plan_hash": "plan-1"}

    def codegen(case_id, state, attempt):
        calls.append("codegen")
        counts["codegen"] += 1
        return {"status": "success", "source_hash": "source-1"}

    def executor(case_id, state, attempt):
        calls.append("executor")
        counts["executor"] += 1
        if counts["executor"] == 1:
            return {"status": "timeout", "reason": "blender timeout"}
        assert state["plan_hash"] == "plan-1"
        assert state["source_hash"] == "source-1"
        return {"status": "success", "blend_hash": "blend-1"}

    def observer(case_id, state, attempt):
        calls.append("observer")
        counts["observer"] += 1
        assert state["blend_hash"] == "blend-1"
        return {"status": "success", "telemetry_hash": "telemetry-1"}

    def evaluator(case_id, state, attempt):
        calls.append("evaluator")
        counts["evaluator"] += 1
        assert state["telemetry_hash"] == "telemetry-1"
        return {"status": "success", "execution_status": "valid", "score": 90}

    result = run_stage_aware_inner_loop(
        ["train-stage-1"],
        tmp_path,
        director=director,
        codegen=codegen,
        executor=executor,
        observer=observer,
        evaluator=evaluator,
        max_attempts=4,
    )

    assert result["status"] == "completed"
    assert calls == ["director", "codegen", "executor", "executor", "observer", "evaluator"]
    assert counts["director"] == counts["codegen"] == counts["observer"] == counts["evaluator"] == 1
    assert counts["executor"] == 2
    retry = result["cases"]["train-stage-1"]["attempts"][0]
    assert retry["retry_stage"] == "executor"
    assert retry["input_hashes"]["plan_hash"] == "plan-1"
    assert retry["input_hashes"]["source_hash"] == "source-1"
    assert json.loads((tmp_path / "stage_retry_progress.jsonl").read_text())["retry_stage"] == "executor"


def test_evaluator_transport_retry_does_not_rerender_and_semantic_failure_is_terminal(tmp_path):
    from videoact.real_inner_loop import run_stage_aware_inner_loop

    calls: list[str] = []

    def stage(name, **extra):
        def callback(*_args):
            calls.append(name)
            return {"status": "success", **extra}

        return callback

    evaluator_calls = 0

    def evaluator(_case_id, state, _attempt):
        nonlocal evaluator_calls
        calls.append("evaluator")
        evaluator_calls += 1
        if evaluator_calls == 1:
            return {"status": "timeout", "reason": "judge transport"}
        return {"status": "fail", "execution_status": "valid", "semantic_status": "failed_required_event"}

    result = run_stage_aware_inner_loop(
        ["train-stage-2"],
        tmp_path,
        director=stage("director", plan_hash="p"),
        codegen=stage("codegen", source_hash="s"),
        executor=stage("executor", blend_hash="b"),
        evaluator=evaluator,
        max_attempts=3,
    )

    assert result["status"] == "completed"
    assert result["cases"]["train-stage-2"]["status"] == "semantic_failed"
    assert calls == ["director", "codegen", "executor", "evaluator", "evaluator"]
    assert result["cases"]["train-stage-2"]["stage_call_counts"]["executor"] == 1

