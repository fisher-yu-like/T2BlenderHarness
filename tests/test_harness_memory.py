import json

import pytest


def test_harness_memory_preserves_update_lifecycle_and_acceptance(tmp_path):
    from training.harness_memory import HarnessMemoryStore

    store = HarnessMemoryStore(tmp_path / "memory")
    memory_id = store.begin_update(
        parent_version="h1",
        candidate_version="h2",
        owner="camera_planner",
        dataset_fingerprint="dataset-v2",
        evaluator_fingerprint="deterministic-v1",
        affected_case_ids=["case-01", "case-02"],
    )
    store.append_event(memory_id, "patch_applied", files=["src/videoact/camera.py"])
    store.append_event(
        memory_id,
        "train_evaluated",
        train_before=70.0,
        train_after=76.0,
        evidence=["train.json"],
    )
    store.append_event(
        memory_id,
        "dev_evaluated",
        dev_before=68.0,
        dev_after=68.0,
        hard_regression=False,
        evidence=["dev.json"],
    )
    store.append_event(
        memory_id,
        "accepted",
        train_before=70.0,
        train_after=76.0,
        dev_before=68.0,
        dev_after=68.0,
        hard_regression=False,
    )

    history = store.history(memory_id)
    assert [event["event"] for event in history] == [
        "proposal",
        "patch_applied",
        "train_evaluated",
        "dev_evaluated",
        "accepted",
    ]
    assert all(event["owner"] == "camera_planner" for event in history)
    lines = (tmp_path / "memory" / "harness_updates.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == [
        "proposal",
        "patch_applied",
        "train_evaluated",
        "dev_evaluated",
        "accepted",
    ]


def test_harness_memory_rejects_test_leakage_and_bad_acceptance(tmp_path):
    from training.harness_memory import HarnessMemoryStore

    store = HarnessMemoryStore(tmp_path / "memory")
    with pytest.raises(ValueError, match="test"):
        store.begin_update(
            parent_version="h1",
            candidate_version="h2",
            owner="camera_planner",
            dataset_fingerprint="dataset-v2",
            evaluator_fingerprint="deterministic-v1",
            affected_case_ids=["case-test-01"],
            forbidden_case_ids={"case-test-01"},
        )

    memory_id = store.begin_update(
        parent_version="h1",
        candidate_version="h2",
        owner="camera_planner",
        dataset_fingerprint="dataset-v2",
        evaluator_fingerprint="deterministic-v1",
        affected_case_ids=["case-01", "case-02"],
    )
    with pytest.raises(ValueError, match="train"):
        store.append_event(
            memory_id,
            "accepted",
            train_before=70.0,
            train_after=70.0,
            dev_before=68.0,
            dev_after=68.0,
            hard_regression=False,
        )
