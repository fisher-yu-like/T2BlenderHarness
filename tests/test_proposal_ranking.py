from __future__ import annotations


def _record(case_id: str, owner: str, root: str, evidence: str, split: str = "train"):
    route = "camera_repair" if "camera" in owner else "trajectory_repair"
    return {
        "case_id": case_id,
        "split": split,
        "findings": [
            {
                "failure_id": root,
                "root_cause_id": root,
                "owner": owner,
                "category": "camera_coverage" if "camera" in owner else "trajectory",
                "severity": "hard",
                "message": root,
                "evidence": [evidence],
                "repair_route": route,
            }
        ],
    }


def test_ranked_owner_proposals_select_one_and_keep_other_owner_as_backlog(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    records = [
        _record("train-camera-1", "director_camera", "camera_visibility", "camera-1.json"),
        _record("train-camera-2", "director_camera", "camera_visibility", "camera-2.json"),
        _record("train-trajectory-1", "director_trajectory", "trajectory_execution", "trajectory-1.json"),
        _record("train-trajectory-2", "director_trajectory", "trajectory_execution", "trajectory-2.json"),
    ]
    optimizer = MetaHarnessOptimizer(output_dir=tmp_path)
    ranked = optimizer.rank_proposals(records)
    payload = optimizer.build_ranked_proposals(records)

    assert len(ranked) == 2
    assert ranked[0].owner == "director_camera"
    assert len(payload["backlog"]) == 1
    assert payload["backlog"][0]["owner"] == "director_trajectory"


def test_ranking_is_deterministic_and_train_only(tmp_path):
    import pytest

    from videoact.meta_harness import MetaHarnessOptimizer

    records = [
        _record("train-1", "director_camera", "camera_visibility", "a.json"),
        _record("train-2", "director_camera", "camera_visibility", "b.json"),
    ]
    optimizer = MetaHarnessOptimizer(output_dir=tmp_path)
    first = optimizer.propose_ranked(records).model_dump(mode="json")
    second = optimizer.propose_ranked(list(reversed(records))).model_dump(mode="json")
    assert first == second
    with pytest.raises(ValueError, match="train-only"):
        optimizer.rank_proposals([_record("dev-1", "director_camera", "camera_visibility", "dev.json", split="dev")])
