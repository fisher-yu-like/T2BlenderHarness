from __future__ import annotations


def test_group_case_ids_caps_each_render_batch_at_twelve_and_preserves_order() -> None:
    from scripts.render_evaluate_groups import group_case_ids

    case_ids = [f"case-{index:02d}" for index in range(25)]
    groups = group_case_ids(case_ids, group_size=12)

    assert [len(group) for group in groups] == [12, 12, 1]
    assert [case_id for group in groups for case_id in group] == case_ids


def test_group_case_ids_rejects_invalid_group_size() -> None:
    import pytest

    from scripts.render_evaluate_groups import group_case_ids

    with pytest.raises(ValueError, match="group_size"):
        group_case_ids(["case-01"], group_size=0)


def test_grouped_pipeline_allows_a_lower_bounded_worker_count(tmp_path) -> None:
    from scripts.render_evaluate_groups import run_grouped_pipeline

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.jsonl").write_text("", encoding="utf-8")
    result = run_grouped_pipeline(
        [tmp_path / "empty-run"],
        dataset_root=dataset,
        blender_bin="missing-blender",
        workers=4,
    )

    assert result["render_workers"] == 4
    assert result["group_count"] == 0
