from __future__ import annotations

import json


def test_outer_loop_records_attribution_before_root_cause_distillation(tmp_path):
    from videoact.outer_loop import record_patch_attribution

    verdict = record_patch_attribution(
        tmp_path / "optimization_record.jsonl",
        {
            "edit_id": "edit-01",
            "affected_files": ["src/videoact/director_camera.py"],
            "predicted_fixes": ["case-a"],
            "predicted_regressions": [],
        },
        {"case-a": -1.0},
    )

    record = json.loads((tmp_path / "optimization_record.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert verdict.verdict == "refuted"
    assert record["event"] == "patch_attribution"
    assert record["ordering"] == "before_root_cause_distillation"
    assert record["verdict"]["rollback_files"] == ["src/videoact/director_camera.py"]

