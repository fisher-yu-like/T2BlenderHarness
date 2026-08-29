from __future__ import annotations

import json
from pathlib import Path


def _records(path: Path):
    finding = {
        "failure_id": "camera_event_uncovered",
        "owner": "camera_planner",
        "category": "camera_coverage",
        "severity": "error",
        "message": "coverage fails",
        "evidence": ["run-a/frame.png"],
    }
    path.write_text(
        "\n".join(json.dumps({"case_id": case, "findings": [finding]}) for case in ("case-a", "case-b")) + "\n",
        encoding="utf-8",
    )


def test_proposal_marks_executable_change_type(tmp_path: Path):
    from skills.t2blendercodeharness.scripts.propose_skill_update import build_update_proposal

    records = tmp_path / "records.jsonl"
    skill = tmp_path / "SKILL.md"
    _records(records)
    skill.write_text("skill", encoding="utf-8")

    result = build_update_proposal(records, skill, change_type="function_library")

    assert result["proposals"][0]["change_type"] == "function_library"
    assert result["proposals"][0]["runtime_change"] is True
    assert "warning" not in result["proposals"][0]


def test_prose_guidance_requires_explicit_reason_and_is_not_training_gain(tmp_path: Path):
    from skills.t2blendercodeharness.scripts.propose_skill_update import build_update_proposal

    records = tmp_path / "records.jsonl"
    skill = tmp_path / "SKILL.md"
    _records(records)
    skill.write_text("skill", encoding="utf-8")

    result = build_update_proposal(records, skill, change_type="prose_guidance")

    proposal = result["proposals"][0]
    assert proposal["runtime_change"] is False
    assert proposal["requires_explicit_reason"] is True
    assert "not counted as Harness training" in proposal["warning"]

