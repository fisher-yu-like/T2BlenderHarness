from __future__ import annotations

import json


def test_real_inner_loop_regenerates_after_plan_or_render_failure_and_selects_third_success(tmp_path):
    from videoact.real_inner_loop import run_real_inner_loop

    split_root = tmp_path / "real"
    calls: list[tuple[str, int, tuple[str, ...]]] = []
    outcomes = {
        ("case-a", 1): {"kind": "plan", "reason": "coverage_failed"},
        ("case-a", 2): {"kind": "render", "reason": "blender_exit_1"},
        ("case-a", 3): {"kind": "success"},
    }

    def prepare(case_ids, attempt):
        calls.append(("prepare", attempt, tuple(case_ids)))
        prepared = []
        failures = {}
        for case_id in case_ids:
            outcome = outcomes[(case_id, attempt)]
            case_dir = split_root / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "attempt.json").write_text(json.dumps({"attempt": attempt}), encoding="utf-8")
            if outcome["kind"] == "plan":
                failures[case_id] = {"status": "coverage_failed", "reason": outcome["reason"]}
            else:
                prepared.append(case_id)
        return {"prepared_ids": prepared, "failures": failures}

    def render(case_ids, attempt):
        calls.append(("render", attempt, tuple(case_ids)))
        results = {}
        for case_id in case_ids:
            outcome = outcomes[(case_id, attempt)]
            results[case_id] = {
                "status": "success" if outcome["kind"] == "success" else "failed",
                "reason": outcome.get("reason"),
            }
        return {"results": results}

    def evaluate(case_id, attempt):
        calls.append(("evaluate", attempt, (case_id,)))
        return {"status": "pass", "score": 91.0, "attempt": attempt}

    result = run_real_inner_loop(
        ["case-a"],
        split_root,
        prepare=prepare,
        render=render,
        evaluate=evaluate,
        max_attempts=3,
    )

    assert result["status"] == "completed"
    assert result["cases"]["case-a"]["selected_attempt"] == 3
    assert [item[1] for item in calls if item[0] == "prepare"] == [1, 2, 3]
    assert [item[1] for item in calls if item[0] == "render"] == [2, 3]
    assert [item[1] for item in calls if item[0] == "evaluate"] == [3]
    assert (split_root / "inner_attempts" / "case-a" / "attempt-01").is_dir()
    assert (split_root / "inner_attempts" / "case-a" / "attempt-02").is_dir()
    assert (split_root / "case-a" / "attempt.json").read_text(encoding="utf-8") == '{"attempt": 3}'


def test_real_inner_loop_fail_closed_after_three_attempts_and_keeps_each_failure(tmp_path):
    from videoact.real_inner_loop import run_real_inner_loop

    split_root = tmp_path / "real"

    def prepare(case_ids, attempt):
        case_dir = split_root / case_ids[0]
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "attempt.txt").write_text(str(attempt), encoding="utf-8")
        return {
            "prepared_ids": case_ids,
            "failures": {},
        }

    def render(case_ids, attempt):
        return {"results": {case_id: {"status": "failed", "reason": f"render-{attempt}"} for case_id in case_ids}}

    def evaluate(case_id, attempt):
        raise AssertionError("failed render must not reach evaluator")

    result = run_real_inner_loop(
        ["case-b"],
        split_root,
        prepare=prepare,
        render=render,
        evaluate=evaluate,
        max_attempts=3,
    )

    assert result["status"] == "exhausted"
    assert result["cases"]["case-b"]["selected_attempt"] is None
    assert len(result["cases"]["case-b"]["attempts"]) == 3
    assert [item["reason"] for item in result["cases"]["case-b"]["attempts"]] == ["render-1", "render-2", "render-3"]
    for attempt in range(1, 4):
        assert (split_root / "inner_attempts" / "case-b" / f"attempt-{attempt:02d}").is_dir()
    assert not (split_root / "case-b").exists()


def test_real_inner_loop_turns_prepare_and_render_exceptions_into_retryable_failures(tmp_path):
    from videoact.real_inner_loop import run_real_inner_loop

    split_root = tmp_path / "real"
    calls: list[str] = []

    def prepare(case_ids, attempt):
        calls.append(f"prepare-{attempt}")
        case_dir = split_root / case_ids[0]
        case_dir.mkdir(parents=True, exist_ok=True)
        if attempt == 1:
            raise RuntimeError("provider crashed")
        return {"prepared_ids": case_ids, "failures": {}}

    def render(case_ids, attempt):
        calls.append(f"render-{attempt}")
        if attempt == 2:
            raise RuntimeError("blender launcher crashed")
        return {"results": {case_id: {"status": "success"} for case_id in case_ids}}

    def evaluate(case_id, attempt):
        calls.append(f"evaluate-{attempt}")
        return {"status": "pass", "score": 90.0}

    result = run_real_inner_loop(
        ["case-c"],
        split_root,
        prepare=prepare,
        render=render,
        evaluate=evaluate,
        max_attempts=3,
    )

    assert result["status"] == "completed"
    assert result["cases"]["case-c"]["selected_attempt"] == 3
    assert calls == ["prepare-1", "prepare-2", "render-2", "prepare-3", "render-3", "evaluate-3"]
    assert result["cases"]["case-c"]["attempts"][0]["status"] == "prepare_failed"
    assert result["cases"]["case-c"]["attempts"][1]["status"] == "render_failed"
