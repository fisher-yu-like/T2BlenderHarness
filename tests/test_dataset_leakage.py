from __future__ import annotations


def _record(case_id, prompt, *, dimension="Camera_Motion", source_index=1):
    return {
        "case_id": case_id,
        "prompt": prompt,
        "source_prompt": prompt,
        "prompt_hash": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
        "source_dataset": "VBench-2.0",
        "source_index": source_index,
        "source_dimension": dimension,
    }


def test_leakage_audit_detects_normalized_prompt_collision() -> None:
    from videoact.dataset_leakage import audit_leakage

    current = [_record("current", "A red ball rolls across the floor.")]
    reference = [_record("reference", "  a RED ball rolls across the floor! ", source_index=999)]

    report = audit_leakage(current, {"reference": reference})

    assert report["status"] == "fail"
    assert "normalized_prompt" in report["collision_counts"]


def test_leakage_audit_detects_semantic_near_duplicate_and_source_family() -> None:
    from videoact.dataset_leakage import audit_leakage

    current = [_record("current", "A woman walks slowly toward a wooden table and lifts a red cup.", dimension="Mechanics", source_index=1)]
    reference = [_record("reference", "A woman walks slowly toward the wooden table and lifts the red cup.", dimension="Mechanics", source_index=999)]

    report = audit_leakage(current, {"reference": reference}, near_duplicate_threshold=0.8)

    assert report["status"] == "fail"
    assert report["collision_counts"]["semantic_near_duplicate"] >= 1
    assert report["collision_counts"]["source_family"] >= 1


def test_leakage_audit_passes_independent_source_family() -> None:
    from videoact.dataset_leakage import audit_leakage

    current = [_record("current", "A camera reveals a mountain landscape.", dimension="Complex_Landscape", source_index=1)]
    reference = [_record("reference", "Two people hand a box to one another.", dimension="Human_Interaction", source_index=999)]

    report = audit_leakage(current, {"reference": reference})

    assert report["status"] == "pass"
    assert report["collisions"] == []
