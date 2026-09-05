from __future__ import annotations

import pytest


def _record(case_id, prompt, *, split, dimension="Camera_Motion", index=1):
    return {
        "case_id": case_id,
        "prompt": prompt,
        "source_prompt": prompt,
        "split": split,
        "source_dataset": "VBench-2.0",
        "source_dimension": dimension,
        "source_index": index,
    }


def test_split_policy_rejects_exact_normalized_near_duplicate_and_source_family():
    from videoact.split_access import SplitAccessPolicy

    train = _record("train-1", "A red ball rolls across the floor.", split="train", dimension="Mechanics", index=1)
    dev = _record("dev-1", "A RED ball rolls across the floor!", split="dev", dimension="Camera", index=2)
    policy = SplitAccessPolicy.from_records([train], [dev], [], near_duplicate_threshold=0.8)

    with pytest.raises(ValueError, match="prompt"):
        policy.validate_records([_record("train-new", "a red ball rolls across the floor", split="train", dimension="Other", index=99)])

    with pytest.raises(ValueError, match="source family"):
        policy.validate_records([_record("train-family", "a different prompt", split="train", dimension="Camera", index=99)])


def test_controller_policy_blocks_dev_test_case_ids_before_proposal(tmp_path):
    from videoact.outer_controller import OuterTransitionController
    from videoact.split_access import SplitAccessPolicy

    policy = SplitAccessPolicy.from_records(
        [_record("train-1", "train prompt", split="train", dimension="train", index=1)],
        [_record("dev-1", "dev prompt", split="dev", dimension="dev", index=2)],
        [_record("test-1", "test prompt", split="test", dimension="test", index=3)],
    )
    controller = OuterTransitionController(output_dir=tmp_path, split_policy=policy)
    with pytest.raises(ValueError, match="split leakage"):
        controller.run([{"case_id": "test-1", "split": "train", "findings": []}])


def test_meta_ranking_accepts_only_train_policy_context(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer
    from videoact.split_access import SplitAccessPolicy

    policy = SplitAccessPolicy.from_records(
        [_record("train-1", "train one", split="train", dimension="train", index=1)],
        [_record("dev-1", "dev one", split="dev", dimension="dev", index=2)],
        [_record("test-1", "test one", split="test", dimension="test", index=3)],
    )
    finding = {
        "failure_id": "camera_failure",
        "root_cause_id": "camera_visibility",
        "owner": "director_camera",
        "category": "camera",
        "severity": "hard",
        "message": "not visible",
        "evidence": ["observer.json"],
        "repair_route": "camera_repair",
    }
    with pytest.raises(ValueError, match="split leakage"):
        MetaHarnessOptimizer(output_dir=tmp_path).rank_proposals(
            [{"case_id": "test-1", "split": "train", "findings": [finding]}],
            split_policy=policy,
        )
