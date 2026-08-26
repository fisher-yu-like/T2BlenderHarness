import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = ROOT / "skills" / "t2blendercodeharness"


def load_skill_script(name: str):
    path = SKILL_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"skill_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def finding(case_id: str, owner: str, failure_id: str) -> dict:
    return {
        "case_id": case_id,
        "status": "fail",
        "findings": [
            {
                "failure_id": failure_id,
                "owner": owner,
                "category": "camera_coverage" if owner == "camera_planner" else "scene_semantics",
                "severity": "hard",
                "message": f"{failure_id} observed",
                "evidence": ["deterministic_report.json"],
            }
        ],
    }


def test_capability_check_passes_current_project():
    module = load_skill_script("capability_check")

    report = module.run_capability_check(ROOT)

    assert report["status"] == "pass"
    assert all(check["status"] == "pass" for check in report["checks"])


def test_capability_check_fails_closed_when_components_are_missing(tmp_path):
    module = load_skill_script("capability_check")

    report = module.run_capability_check(tmp_path)

    assert report["status"] == "fail"
    assert any(check["name"] == "required_components" for check in report["checks"])


def test_self_evolution_splits_mixed_owners_and_ignores_vlm_unavailable(tmp_path):
    module = load_skill_script("propose_skill_update")
    records_path = tmp_path / "records.jsonl"
    records = [
        finding("case-01", "camera_planner", "camera_event_uncovered"),
        finding("case-02", "camera_planner", "camera_event_uncovered"),
        finding("case-03", "scene_parser", "telemetry_entity_kind_mismatch"),
        finding("case-04", "scene_parser", "telemetry_entity_kind_mismatch"),
        {"case_id": "case-vlm", "status": "unavailable", "source": "vlm", "findings": []},
    ]
    records_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    skill_path = SKILL_ROOT / "SKILL.md"
    before = hashlib.sha256(skill_path.read_bytes()).hexdigest()

    result = module.build_update_proposal(records_path, skill_path)

    after = hashlib.sha256(skill_path.read_bytes()).hexdigest()
    assert result["status"] == "proposal_ready"
    assert result["requires_human_review"] is True
    assert {proposal["owner"] for proposal in result["proposals"]} == {"camera_planner", "scene_parser"}
    assert result["ignored_vlm_unavailable"] == 1
    assert before == after


def test_self_evolution_requires_repeated_distinct_cases(tmp_path):
    module = load_skill_script("propose_skill_update")
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        json.dumps(finding("case-01", "camera_planner", "camera_event_uncovered")) + "\n",
        encoding="utf-8",
    )

    result = module.build_update_proposal(records_path, SKILL_ROOT / "SKILL.md")

    assert result["status"] == "no_action"
    assert result["proposals"] == []
