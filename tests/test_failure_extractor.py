from __future__ import annotations

import pytest


def _visual_record(case_id: str, *, camera_score: float = 42.0, confidence: float = 0.92) -> dict:
    return {
        "case_id": case_id,
        "split": "train",
        "status": "pass",
        "deterministic_report": {
            "terminal_status": "pass",
            "hard_gate_failed": False,
            "score": 100.0,
            "findings": [],
        },
        "obligations": {
            "obligation_ids": ["camera_coverage.handoff.abc123"],
            "obligations": [
                {
                    "obligation_id": "camera_coverage.handoff.abc123",
                    "kind": "camera_coverage",
                    "required": True,
                    "applicable": True,
                }
            ],
        },
        "vlm_report": {
            "status": "scored",
            "review_source": "gpt-5.6-luna",
            "visual_primary": {
                "status": "scored",
                "camera_coverage": camera_score,
                "confidence": confidence,
                "dimension_evidence": {
                    "camera_coverage": {
                        "evidence_completeness": 1.0,
                        "evidence_refs": ["frames/frame_0042.png"],
                    }
                },
            },
            "sampled_frames": ["frames/frame_0042.png"],
        },
    }


def test_pass_status_still_emits_actionable_repeated_visibility_bottleneck() -> None:
    from videoact.failure_extractor import FailureExtractor

    extractor = FailureExtractor(visual_failure_threshold=60.0)
    first = extractor.extract(_visual_record("train-01"))
    second = extractor.extract(_visual_record("train-02"))

    assert first and second
    assert first[0].actionable is True
    assert first[0].abstain is False
    assert first[0].first_divergence_stage == "visible"
    assert first[0].owner_candidate == "director_camera"
    assert first[0].root_cause_id == second[0].root_cause_id == "camera_visibility"
    assert first[0].affected_obligation_ids == ["camera_coverage.handoff.abc123"]


@pytest.mark.parametrize(
    "patch",
    [
        {"status": "unavailable"},
        {"status": "needs_human_review", "confidence": 0.41},
        {"status": "scored", "confidence": 0.91, "dimension_evidence": {}},
    ],
)
def test_unavailable_low_confidence_or_incomplete_evidence_abstains(patch: dict) -> None:
    from videoact.failure_extractor import FailureExtractor

    record = _visual_record("train-abstain")
    record["vlm_report"]["visual_primary"].update(patch)
    evidence = FailureExtractor().extract(record)

    assert evidence
    assert evidence[0].abstain is True
    assert evidence[0].actionable is False
    assert evidence[0].owner_confidence < 0.6


def test_natural_language_visibility_findings_share_normalized_root_cause() -> None:
    from videoact.failure_extractor import FailureExtractor

    base = {
        "case_id": "train-normalize",
        "split": "train",
        "deterministic_report": {
            "terminal_status": "fail",
            "hard_gate_failed": True,
            "score": 50.0,
            "findings": [
                {
                    "failure_id": "handoff_occluded",
                    "owner": "director_camera",
                    "category": "camera_coverage",
                    "severity": "hard",
                    "message": "the handoff is occluded in the shot",
                    "root_cause_id": "handoff_occlusion_a",
                    "evidence": ["deterministic_report.json"],
                    "repair_route": "camera_repair",
                },
                {
                    "failure_id": "handoff_not_visible",
                    "owner": "director_camera",
                    "category": "visibility",
                    "severity": "error",
                    "message": "visibility of the handoff is blocked",
                    "root_cause_id": "different-natural-language-label",
                    "evidence": ["visual_evidence.json"],
                    "repair_route": "camera_repair",
                },
            ],
        },
    }

    evidence = FailureExtractor().extract(base)

    assert len(evidence) == 2
    assert {item.root_cause_id for item in evidence} == {"camera_visibility"}


def test_observer_and_artifact_sources_are_normalized_without_trusting_generated_findings() -> None:
    from videoact.failure_extractor import FailureExtractor

    record = {
        "case_id": "train-runtime-01",
        "split": "train",
        "artifact_report": {
            "artifact_status": "incomplete",
            "hard_failures": ["missing_artifact:proxy.mp4"],
        },
        "observer_report": {
            "status": "fail",
            "trusted": False,
            "failures": ["telemetry_hash_mismatch"],
        },
        "telemetry": {"handoff_success": True},
    }

    evidence = FailureExtractor().extract(record)

    assert {item.root_cause_id for item in evidence} == {"artifact_completeness", "runtime_telemetry"}
    assert all("telemetry" not in item.evidence_refs for item in evidence if item.root_cause_id == "runtime_telemetry")


def test_test_or_dev_input_is_rejected_before_extraction() -> None:
    from videoact.failure_extractor import FailureExtractor

    with pytest.raises(ValueError, match="train-only"):
        FailureExtractor().extract({"case_id": "dev-01", "split": "dev"})

    with pytest.raises(ValueError, match="forbidden case"):
        FailureExtractor().extract(
            {"case_id": "train-01", "split": "train", "note": "test-99"},
            forbidden_case_ids={"test-99"},
        )


def test_failure_extractor_exposes_report_and_only_actionable_findings() -> None:
    from videoact.failure_extractor import FailureExtractor

    extractor = FailureExtractor(visual_failure_threshold=60.0)
    report = extractor.extract_report(_visual_record("train-report"))

    assert report.status == "actionable"
    assert report.actionable_count == 1
    assert report.abstain_count == 0
    assert report.to_patch_findings()[0]["root_cause_id"] == "camera_visibility"


def test_obligation_matrix_primary_failure_is_the_only_patchable_record_for_a_row() -> None:
    from videoact.failure_extractor import FailureExtractor

    record = {
        "case_id": "train-matrix-01",
        "split": "train",
        "obligations": {
            "obligation_ids": ["event-01"],
            "obligations": [{"obligation_id": "event-01", "kind": "event", "required": True, "applicable": True}],
        },
        "obligation_matrix": {
            "primary_failures": [
                {
                    "failure_id": "obligation_failed:event-01",
                    "obligation_id": "event-01",
                    "root_cause_id": "obligation_planning",
                    "first_divergence_stage": "planned",
                    "owner_candidate": "director_event_scheduler",
                    "severity": "hard",
                    "expected": {"event_id": "event-01"},
                    "evidence_refs": ["obligation_matrix.json:event-01"],
                }
            ]
        },
    }

    evidence = FailureExtractor().extract(record)

    assert len(evidence) == 1
    assert evidence[0].actionable is True
    assert evidence[0].first_divergence_stage == "planned"
    assert evidence[0].affected_obligation_ids == ["event-01"]
