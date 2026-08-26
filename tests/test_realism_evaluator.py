from evaluator.realism import score_realism


def test_geometry_compliance_does_not_claim_realism_without_independent_review():
    result = score_realism({
        "coverage_score": 100,
        "topology_score": 100,
        "primitive_score": 100,
        "semantic_score": 100,
        "structural_score": 100,
        "hard_gate_failed": False,
    })
    assert result["score"] == 48.0
    assert result["score_kind"] == "artifact_only_proxy"
    assert result["score"] <= result["artifact_only_ceiling"]
    assert result["requires_independent_review"] is True
    assert result["realism_claim"] == "not_established"
    assert result["evaluator_version"] == "realism-v4-shared-visual-review"


def test_realism_score_exposes_primitive_failure_without_rewriting_deterministic_score():
    result = score_realism({
        "coverage_score": 100,
        "topology_score": 0,
        "primitive_score": 0,
        "semantic_score": 60,
        "hard_gate_failed": True,
    })
    assert result["score"] == 9.6
    assert result["band"] == "artifact_only_weak"
    assert result["evaluator_version"] == "realism-v4-shared-visual-review"


def test_even_perfect_render_metrics_cannot_saturate_artifact_only_score_at_100():
    result = score_realism({
        "coverage_score": 100,
        "topology_score": 100,
        "primitive_score": 100,
        "semantic_score": 100,
        "structural_score": 100,
        "hard_gate_failed": False,
    }, {"status": "complete", "score": 100})
    assert result["score"] == 80.0
    assert result["artifact_only_unbounded_score"] == 100.0
    assert result["requires_independent_review"] is True


def test_independent_review_unlocks_fused_score_without_faking_review():
    result = score_realism({
        "coverage_score": 100,
        "topology_score": 100,
        "primitive_score": 100,
        "semantic_score": 100,
        "structural_score": 100,
        "hard_gate_failed": False,
    }, {"status": "complete", "score": 80}, {
        "status": "complete",
        "source": "gpt-5.6-luna",
        "confidence": 0.9,
        "scores": {
            "appearance_detail": 80,
            "physical_realism": 80,
            "spatial_consistency": 80,
            "motion_naturalness": 80,
            "visual_presentation": 80,
        },
    })
    assert result["score_kind"] == "independent_review_fused"
    assert result["requires_independent_review"] is False
    assert result["score"] == 83.0
