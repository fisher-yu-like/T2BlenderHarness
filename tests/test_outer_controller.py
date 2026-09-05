from __future__ import annotations

import hashlib
import json

import pytest


def _finding(case_id: str, *, root: str = "camera_visibility", owner: str = "director_camera") -> dict:
    return {
        "case_id": case_id,
        "failure_id": "handoff_not_visible",
        "root_cause_id": root,
        "owner": owner,
        "category": "camera_coverage",
        "severity": "hard",
        "message": "the required event is out of frame",
        "evidence": [f"{case_id}/observer_report.json"],
        "repair_route": "camera_repair",
    }


def _records(*case_ids: str) -> list[dict]:
    return [
        {"case_id": case_id, "split": "train", "findings": [_finding(case_id)]}
        for case_id in case_ids
    ]


def test_no_repeated_finding_is_one_explicit_no_patch_attempt(tmp_path):
    from videoact.outer_controller import OuterTransitionController

    calls: list[dict] = []
    controller = OuterTransitionController(output_dir=tmp_path)
    result = controller.run(
        _records("train-one"),
        coding_agent=lambda payload: calls.append(payload),
    )

    assert result["action"] == "no_patch"
    assert result["attempt_count"] == 1
    assert calls == []
    assert len(controller.transitions) == 1
    assert json.loads((tmp_path / "outer_transitions.jsonl").read_text())[
        "event"
    ] == "outer_transition"


def test_proposal_without_coding_agent_is_blocked_not_patch(tmp_path):
    from videoact.outer_controller import OuterTransitionController

    result = OuterTransitionController(output_dir=tmp_path).run(_records("train-one", "train-two"))

    assert result["action"] == "blocked"
    assert result["proposal"]["owner"] == "director_camera"
    assert "Coding Agent" in result["reason"]
    assert result["transition"]["append_only"] is True


def test_empty_or_unhashed_diff_can_never_return_patch(tmp_path):
    from videoact.outer_controller import OuterTransitionController

    for response in ({}, {"diff": "real diff", "changed_files": ["src/videoact/director_camera.py"]}):
        result = OuterTransitionController(output_dir=tmp_path).run(
            _records("train-one", "train-two"),
            coding_agent=lambda _proposal, response=response: response,
        )
        assert result["action"] == "blocked"
        assert result["action"] != "patch"


def test_verified_one_owner_diff_returns_patch_and_hashes_manifest(tmp_path):
    from videoact.outer_controller import OuterTransitionController

    diff = "diff --git a/src/videoact/director_camera.py b/src/videoact/director_camera.py\n+@@ -1 +1 @@\n-old\n+new\n"
    result = OuterTransitionController(output_dir=tmp_path).run(
        _records("train-one", "train-two"),
        coding_agent=lambda proposal: {
            "diff": diff,
            "manifest": {
                "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "changed_files": proposal["affected_files"],
                "production_call_sites": ["compose_camera_plan"],
            },
        },
    )

    assert result["action"] == "patch"
    assert result["diff_sha256"] == hashlib.sha256(diff.encode()).hexdigest()
    assert result["changed_files"] == ["src/videoact/director_camera.py"]
    assert result["patch_manifest"]["source_diff_present"] is True
    assert result["transition"]["action"] == "patch"


@pytest.mark.parametrize(
    "changed_file",
    [
        "src/videoact/evaluator_policy.py",
        "src/videoact/observer_contract.py",
        "dataset/train.jsonl",
        "tests/test_camera.py",
    ],
)
def test_controller_rejects_frozen_component_changes(tmp_path, changed_file):
    from videoact.outer_controller import OuterTransitionController

    diff = "diff --git a/frozen b/frozen\n+@@ -1 +1 @@\n-old\n+new\n"
    result = OuterTransitionController(output_dir=tmp_path).run(
        _records("train-one", "train-two"),
        coding_agent=lambda _proposal: {
            "diff": diff,
            "changed_files": [changed_file],
            "diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
        },
    )

    assert result["action"] == "blocked"
    assert "evaluator, dataset, test, or observer" in result["reason"]


def test_controller_rejects_dev_test_context_and_excess_attempts(tmp_path):
    from videoact.outer_controller import OuterTransitionController

    controller = OuterTransitionController(output_dir=tmp_path, max_attempts=5)
    with pytest.raises(ValueError, match="train-only"):
        controller.run([{"case_id": "dev-one", "split": "dev", "findings": []}])
    with pytest.raises(ValueError, match="between 1 and 5"):
        controller.run(_records("train-one"), attempt=6)


def test_run_attempts_stops_after_first_no_patch_without_hidden_attempt(tmp_path):
    from videoact.outer_controller import OuterTransitionController

    seen: list[int] = []
    controller = OuterTransitionController(output_dir=tmp_path)
    result = controller.run_attempts(
        lambda attempt: seen.append(attempt) or _records("train-one"),
    )

    assert seen == [1]
    assert result["attempt_count"] == 1
    assert result["action"] == "no_patch"


def test_formal_runner_blocks_without_controller_before_any_render(tmp_path, monkeypatch):
    import scripts.train_real_harness as training

    monkeypatch.setattr(training, "require_experiment_contract", lambda _path: None)
    result = training.run_six_round_protocol(
        tmp_path / "formal",
        dataset_root="dataset/vbench2-agent-training-index-v1",
        test_dataset_root="dataset/vbench2-agent-test-100-v1",
        harness_version="harness-rsi-test",
        evaluator_version="evaluator-test",
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        vlm_model="judge-v1",
        director_agent=object(),
        code_agent=object(),
        formal=True,
    )

    assert result["status"] == "blocked"
    assert result["formal_training_allowed"] is False
    assert result["rounds"] == []
