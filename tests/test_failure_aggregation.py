from videoact.contracts import Finding


def finding(owner="camera_planner", failure_id="missing_required_event"):
    return Finding(
        failure_id=failure_id,
        owner=owner,
        category="camera_coverage",
        severity="hard",
        message="grasp is not visible",
        evidence=["attempts/01/deterministic_report.json"],
        repair_route="camera_repair",
    )


def test_aggregate_failures_groups_by_normalized_failure_and_owner():
    from videoact.evolution import aggregate_failures

    summary = aggregate_failures(
        [
            {"case_id": "case-1", "findings": [finding()]},
            {"case_id": "case-2", "findings": [finding()]},
            {"case_id": "case-3", "findings": [finding(owner="trajectory_planner", failure_id="velocity_spike")]},
        ]
    )

    assert summary.total_cases == 3
    assert len(summary.groups) == 2
    camera_group = next(group for group in summary.groups if group.owner == "camera_planner")
    assert camera_group.count == 2
    assert camera_group.affected_case_ids == ["case-1", "case-2"]


def test_build_patch_brief_selects_one_owner_and_exact_rerun_command():
    from videoact.evolution import aggregate_failures, build_patch_brief

    summary = aggregate_failures([{"case_id": "case-1", "findings": [finding()]}])
    brief = build_patch_brief(summary)

    assert brief.owner == "camera_planner"
    assert brief.affected_files == ["src/videoact/camera.py"]
    assert "missing_required_event" in brief.observed_failure_pattern
    assert brief.rerun_command == "python scripts/run_harness_eval.py --split train"
