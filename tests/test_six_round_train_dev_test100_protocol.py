from __future__ import annotations

import json
from pathlib import Path


def test_glm_protocol_uses_ten_train_ten_dev_and_full_test100_after_each_round() -> None:
    from scripts.train_real_harness import build_glm_six_round_manifest

    train = [f"train-{family:02d}-{item:02d}" for family in range(1, 7) for item in range(1, 11)]
    dev = [f"dev-{family:02d}-{item:02d}" for family in range(1, 7) for item in range(1, 11)]
    test = [f"vbench2-test100-{family:02d}-{item:02d}" for family in range(1, 11) for item in range(1, 11)]

    manifest = build_glm_six_round_manifest(
        train,
        dev,
        test,
        dataset_fingerprint="training-fp",
        test_dataset_id="vbench2-agent-test-100-v1",
        test_dataset_fingerprint="test100-fp",
    )

    assert manifest["round_count"] == 6
    assert manifest["attempts_per_round_max"] == 5
    assert manifest["batch_case_count"] == 20
    assert manifest["test_case_count_per_round"] == 100
    assert all(len(item["train"]) == 10 and len(item["dev"]) == 10 for item in manifest["rounds"])
    assert all("overall_evaluation" not in item for item in manifest["rounds"])
    assert all(item["test_evaluation"]["test_cases"] == test for item in manifest["rounds"])
    assert all(item["test_evaluation"]["selection_excluded"] is True for item in manifest["rounds"])
    assert all(item["test_evaluation"]["patch_selection_excluded"] is True for item in manifest["rounds"])


def test_six_round_runner_does_not_run_full_training_overall_and_uses_test100_each_round(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.train_real_harness as training

    training_root = Path("dataset/vbench2-agent-training-index-v1")
    test_root = Path("dataset/vbench2-agent-test-100-v1")
    training_split = json.loads((training_root / "splits.json").read_text(encoding="utf-8"))
    test_split = json.loads((test_root / "splits.json").read_text(encoding="utf-8"))
    test_calls: list[tuple[int, list[str], str]] = []
    overall_calls: list[int] = []

    def fake_attempt(output_root, *, round_number, attempt_number, **kwargs):
        empty = {"aggregate": {}, "real_video_count": 0, "vlm_scored_count": 0}
        return {"round": round_number, "attempt": attempt_number, "splits": {"train": empty, "dev": empty}}

    def fake_overall(*args, **kwargs):
        overall_calls.append(kwargs["round_number"])
        return {"round": kwargs["round_number"], "splits": {"train": {}, "dev": {}}}

    def fake_test(output_root, *, round_number, test_case_ids, test_dataset_root, **kwargs):
        test_calls.append((round_number, list(test_case_ids), str(test_dataset_root)))
        return {
            "round": round_number,
            "scope": "vbench2_test100_after_every_round",
            "split": "test",
            "case_ids": list(test_case_ids),
        }

    def transition(attempt_number, reports):
        assert all("test" not in report for report in reports)
        return {"action": "stop", "status": "round_complete", "reason": "test is post-round only"}

    monkeypatch.setattr(training, "run_outer_attempt", fake_attempt)
    monkeypatch.setattr(training, "run_outer_overall", fake_overall)
    monkeypatch.setattr(training, "run_round_test", fake_test)

    result = training.run_six_round_protocol(
        tmp_path / "rounds",
        dataset_root=training_root,
        test_dataset_root=test_root,
        harness_version="test-harness",
        evaluator_version="test-evaluator",
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        vlm_model="gpt-5.6-luna",
        markdown_path=tmp_path / "memory.md",
        director_agent=object(),
        code_agent=object(),
        provider_mode="glm",
        outer_transition=transition,
    )

    assert not overall_calls
    assert [round_number for round_number, _, _ in test_calls] == [1, 2, 3, 4, 5, 6]
    assert all(len(case_ids) == 100 for _, case_ids, _ in test_calls)
    assert all(Path(dataset).resolve() == test_root.resolve() for _, _, dataset in test_calls)
    assert [round_report["test"]["round"] for round_report in result["rounds"]] == [1, 2, 3, 4, 5, 6]


def test_six_round_formal_schedule_runs_frozen_test_only_at_baseline_and_final(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.train_real_harness as training

    training_root = Path("dataset/vbench2-agent-training-index-v1")
    test_root = Path("dataset/vbench2-agent-test-100-v1")
    test_calls: list[int] = []

    def fake_attempt(output_root, *, round_number, attempt_number, **kwargs):
        empty = {"aggregate": {}, "real_video_count": 0, "vlm_scored_count": 0}
        return {"round": round_number, "attempt": attempt_number, "splits": {"train": empty, "dev": empty}}

    def fake_test(output_root, *, round_number, test_case_ids, **kwargs):
        test_calls.append(round_number)
        return {"round": round_number, "scope": "frozen_test", "split": "test", "case_ids": list(test_case_ids)}

    monkeypatch.setattr(training, "run_outer_attempt", fake_attempt)
    monkeypatch.setattr(training, "run_round_test", fake_test)

    result = training.run_six_round_protocol(
        tmp_path / "formal-like-rounds",
        dataset_root=training_root,
        test_dataset_root=test_root,
        harness_version="test-harness",
        evaluator_version="test-evaluator",
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        vlm_model="gpt-5.6-luna",
        markdown_path=tmp_path / "memory.md",
        director_agent=object(),
        code_agent=object(),
        provider_mode="glm",
        outer_transition=lambda _attempt, _reports: {"action": "accept", "status": "accepted", "reason": "ok"},
        test_schedule="baseline_final_only",
    )

    assert test_calls == [0, 6]
    assert result["test_schedule"] == "baseline_final_only"
    assert result["baseline_test"]["round"] == 0
    assert result["final_test"]["round"] == 6
    assert all("test" not in item for item in result["rounds"][:-1])
    assert "test" in result["rounds"][-1]


def test_diagnostic_rounds_never_call_visual_provider(monkeypatch, tmp_path: Path) -> None:
    import scripts.train_real_harness as training

    seen: list[dict] = []
    sentinel = object()

    def fake_attempt(output_root, *, round_number, attempt_number, **kwargs):
        seen.append(kwargs)
        empty = {"aggregate": {}, "real_video_count": 0, "vlm_scored_count": 0}
        return {"round": round_number, "attempt": attempt_number, "splits": {"train": empty, "dev": empty}}

    def fake_test(output_root, *, round_number, **kwargs):
        seen.append(kwargs)
        return {"round": round_number, "scope": "frozen_test", "split": "test", "case_ids": []}

    monkeypatch.setattr(training, "run_outer_attempt", fake_attempt)
    monkeypatch.setattr(training, "run_round_test", fake_test)

    training.run_six_round_protocol(
        tmp_path / "diagnostic-rounds",
        dataset_root=Path("dataset/vbench2-agent-training-index-v1"),
        test_dataset_root=Path("dataset/vbench2-agent-test-100-v1"),
        harness_version="test-harness",
        evaluator_version="test-evaluator",
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        vlm_model="gpt-5.6-luna",
        markdown_path=tmp_path / "memory.md",
        director_agent=object(),
        code_agent=object(),
        provider_mode="glm",
        outer_transition=lambda _attempt, _reports: {
            "action": "stop",
            "status": "round_complete",
            "reason": "diagnostic",
        },
        diagnostic_only=True,
        visual_provider=sentinel,
        test_schedule="baseline_final_only",
    )

    assert seen
    assert all(item.get("visual_provider") is None for item in seen)
    assert all(item.get("assistant_local") is True for item in seen)
