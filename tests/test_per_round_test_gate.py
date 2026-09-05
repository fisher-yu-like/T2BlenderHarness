from __future__ import annotations

import json
from pathlib import Path


def test_six_round_manifest_declares_the_frozen_test_for_every_round() -> None:
    from scripts.train_real_harness import build_protocol_manifest

    def family(prefix: str) -> list[str]:
        return [f"{prefix}-{group:02d}-{item:02d}" for group in range(1, 7) for item in range(1, 11)]

    train = family("train")
    dev = family("dev")
    test = [f"test-{item:02d}" for item in range(1, 21)]

    manifest = build_protocol_manifest(train, dev, test, dataset_fingerprint="fp")

    assert len(manifest["rounds"]) == 6
    assert all(round_spec["test_evaluation"]["test_cases"] == test for round_spec in manifest["rounds"])
    assert all(round_spec["test_evaluation"]["selection_excluded"] is True for round_spec in manifest["rounds"])


def test_six_round_protocol_runs_test_after_each_round_without_test_leaking_to_transition(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.train_real_harness as training

    dataset_root = Path("dataset/vbench2-agent-training-index-v1")
    test_dataset_root = Path("dataset/vbench2-agent-test-100-v1")
    test_split_payload = json.loads((test_dataset_root / "splits.json").read_text(encoding="utf-8"))
    calls: list[tuple[str, int]] = []
    transitions: list[int] = []

    def fake_attempt(output_root, *, round_number, attempt_number, **kwargs):
        empty = {"aggregate": {}, "real_video_count": 0, "vlm_scored_count": 0}
        return {"round": round_number, "attempt": attempt_number, "splits": {"train": empty, "dev": empty}}

    def fake_overall(output_root, *, round_number, **kwargs):
        empty = {"aggregate": {}, "real_video_count": 0, "vlm_scored_count": 0}
        return {"round": round_number, "splits": {"train": empty, "dev": empty}}

    def fake_test(output_root, *, round_number, test_case_ids, test_dataset_root, **kwargs):
        calls.append(("test", round_number))
        assert test_case_ids == test_split_payload["test"]
        assert Path(test_dataset_root).resolve() == Path("dataset/vbench2-agent-test-100-v1").resolve()
        return {"round": round_number, "scope": "frozen_test", "split": "test", "case_ids": test_case_ids}

    def transition(attempt_number, reports):
        transitions.append(attempt_number)
        assert all("test" not in report for report in reports)
        return {"action": "accept", "status": "accepted", "reason": "test remains outside patch selection"}

    monkeypatch.setattr(training, "run_outer_attempt", fake_attempt)
    monkeypatch.setattr(training, "run_outer_overall", fake_overall)
    monkeypatch.setattr(training, "run_round_test", fake_test, raising=False)

    result = training.run_six_round_protocol(
        tmp_path,
        dataset_root=dataset_root,
        test_dataset_root=test_dataset_root,
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

    assert [round_number for _, round_number in calls] == [1, 2, 3, 4, 5, 6]
    assert transitions == [1, 1, 1, 1, 1, 1]
    assert [round_report["test"]["round"] for round_report in result["rounds"]] == [1, 2, 3, 4, 5, 6]
