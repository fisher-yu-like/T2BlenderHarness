def test_complete_assistant_local_reviews_exposes_a_non_vlm_scoring_mode(tmp_path, monkeypatch):
    import scripts.complete_assistant_local_reviews as module

    monkeypatch.setattr(module, "evaluate_real_split", lambda *args, **kwargs: [{"case_id": "case-01", "score": 90}])
    monkeypatch.setattr(
        module,
        "evaluate_split",
        lambda *args, **kwargs: [{"case_id": "case-01", "status": "scored", "review_source": "assistant_local_review", "aggregate": {"final_score": 89.0}}],
    )
    monkeypatch.setattr(
        module,
        "merge_real_scores",
        lambda **kwargs: {
            "scoring_mode": "real_blender_video_assistant_local_review",
            "real_video_count": 1,
            "vlm_scored_count": 1,
            "cases": [],
        },
    )
    captured = {}
    monkeypatch.setattr(module, "write_unified_outputs", lambda report, **kwargs: captured.update(report))

    result = module.complete_reviews(
        tmp_path,
        dataset_root=tmp_path,
        assistant_review_dir=tmp_path / "reviews",
    )

    assert result["status"] == "complete"
    assert result["review_source"] == "assistant_local_review"
    assert captured["scoring_mode"] == "real_blender_video_assistant_local_review"
