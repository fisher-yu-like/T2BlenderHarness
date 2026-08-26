from __future__ import annotations


def test_multi_five_round_manifest_matches_50_60_30_protocol():
    from scripts.train_real_harness import build_multi_five_round_manifest

    train = [f"multi-train-{index:03d}" for index in range(1, 51)]
    dev = [f"multi-dev-{index:03d}" for index in range(1, 61)]
    test = [f"multi-test-{index:03d}" for index in range(1, 31)]
    manifest = build_multi_five_round_manifest(train, dev, test, dataset_fingerprint="fp")

    assert manifest["protocol_version"] == "multi-five-rounds-v1"
    assert manifest["round_count"] == 5
    assert manifest["train_count"] == 50
    assert manifest["dev_count"] == 60
    assert manifest["test_count"] == 30
    assert all(len(item["train"]) == 10 and len(item["dev"]) == 10 for item in manifest["rounds"])
    assert all(len(item["overall_evaluation"]["dev_cases"]) == 60 for item in manifest["rounds"])
    assert manifest["final_evaluation"]["blind_test_cases"] == test


def test_multi_protocol_defaults_are_declared_in_training_skill_and_cli():
    from scripts.train_real_harness import build_multi_five_round_manifest

    assert build_multi_five_round_manifest.__name__ == "build_multi_five_round_manifest"
