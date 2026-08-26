from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RETAINED_RUN_ROOT = Path(
    r"C:\Users\sy\Desktop\T2BlenderCode\out\training\single-five-rounds-v1"
)
RETAINED_REPORT = RETAINED_RUN_ROOT / "round-01" / "attempt_report.json"


def test_index_retained_single_entity_baseline_truthfully_records_unavailable_review(
    tmp_path: Path,
):
    from scripts.index_training_baseline import index_baseline

    output = tmp_path / "single-v1-round01-attempt03.json"
    original_report = RETAINED_REPORT.read_bytes()
    summary = index_baseline(RETAINED_RUN_ROOT, output)

    assert summary["run_root"].endswith("out\\training\\single-five-rounds-v1")
    assert summary["round"] == 1
    assert summary["attempt"] == 3
    assert summary["splits"]["train"]["case_count"] == 10
    assert summary["splits"]["dev"]["case_count"] == 10
    assert summary["splits"]["train"]["aggregate"]["mean_deterministic_score"] == 100.0
    assert summary["splits"]["dev"]["aggregate"]["mean_deterministic_score"] == 100.0
    assert summary["splits"]["train"]["aggregate"]["mean_artifact_only_realism_score"] == 68.6689
    assert summary["splits"]["dev"]["aggregate"]["mean_artifact_only_realism_score"] == 68.3419
    assert summary["visual_review_status"] == "unavailable"
    assert summary["task_score"] is None
    assert len(summary["source_report_sha256"]) == 64
    assert summary["source_report_sha256"] == hashlib.sha256(RETAINED_REPORT.read_bytes()).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == summary
    assert RETAINED_REPORT.read_bytes() == original_report


def test_indexer_fails_closed_for_missing_source_evidence(tmp_path: Path):
    from scripts.index_training_baseline import BaselineEvidenceError, index_baseline

    with pytest.raises(BaselineEvidenceError, match="source report"):
        index_baseline(tmp_path / "single-five-rounds-v1", tmp_path / "baseline.json")


def test_indexer_fails_closed_for_identity_or_count_mismatch(tmp_path: Path):
    from scripts.index_training_baseline import BaselineEvidenceError, index_baseline

    run_root = tmp_path / "single-five-rounds-v1"
    report_path = run_root / "round-01" / "attempt_report.json"
    report_path.parent.mkdir(parents=True)
    report = json.loads(RETAINED_REPORT.read_text(encoding="utf-8"))
    report["attempt"] = 2
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(BaselineEvidenceError, match="attempt"):
        index_baseline(run_root, tmp_path / "baseline.json")

    report["attempt"] = 3
    report["splits"]["dev"]["case_count"] = 9
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(BaselineEvidenceError, match="dev"):
        index_baseline(run_root, tmp_path / "baseline.json")


def test_indexer_rejects_completed_visual_scores(tmp_path: Path):
    from scripts.index_training_baseline import BaselineEvidenceError, index_baseline

    run_root = tmp_path / "single-five-rounds-v1"
    report_path = run_root / "round-01" / "attempt_report.json"
    report_path.parent.mkdir(parents=True)
    report = json.loads(RETAINED_REPORT.read_text(encoding="utf-8"))
    report["splits"]["train"]["status"] = "completed"
    report["splits"]["train"]["vlm_scored_count"] = 10
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(BaselineEvidenceError, match="visual"):
        index_baseline(run_root, tmp_path / "baseline.json")
