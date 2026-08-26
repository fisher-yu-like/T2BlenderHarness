import json


def test_training_pipeline_runs_all_splits_and_persists_memory(tmp_path):
    from scripts.train_harness_with_memory import run_training

    report = run_training("dataset/trajectory-v2", tmp_path, harness_version="h-trajectory-v2")

    assert report["dataset_id"] == "trajectory-v2"
    assert report["benchmarks"]["train"]["case_count"] == 50
    assert report["benchmarks"]["dev"]["case_count"] == 20
    assert report["benchmarks"]["test"]["case_count"] == 10
    assert report["test_policy"] == "frozen_final_only"
    memory_lines = (tmp_path / "memory" / "harness_updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert memory_lines
    assert all(json.loads(line)["event"] in {"proposal", "train_evaluated", "dev_evaluated", "test_evaluated", "no_patch", "rejected", "accepted", "rollback"} for line in memory_lines)


def test_training_pipeline_does_not_select_test_cases_for_patch(tmp_path):
    from scripts.train_harness_with_memory import run_training

    report = run_training("dataset/trajectory-v2", tmp_path, harness_version="h-trajectory-v2")

    assert not set(report["outer_loop"].get("proposal_case_ids", [])) & set(report["test_case_ids"])
