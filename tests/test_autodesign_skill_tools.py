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
    assert report["skill_version"] == "t2blendercodeharness-v5-executable-director"
    assert all(check["status"] == "pass" for check in report["checks"])


def test_capability_check_covers_agent_codegen_and_frozen_eval_boundaries():
    module = load_skill_script("capability_check")

    report = module.run_capability_check(ROOT)
    names = {check["name"] for check in report["checks"]}

    assert "agent_codegen_fail_closed" in names
    assert "frozen_eval_boundary" in names
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


def test_self_evolution_targets_director_owner_and_current_skill_version(tmp_path):
    module = load_skill_script("propose_skill_update")
    records_path = tmp_path / "records.jsonl"
    records = [
        finding("multi-train-041", "director_prompt_interpreter", "implicit_event_order_not_preserved"),
        finding("multi-train-042", "director_prompt_interpreter", "implicit_event_order_not_preserved"),
    ]
    records_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    result = module.build_update_proposal(records_path, SKILL_ROOT / "SKILL.md")

    assert result["skill_version"] == "t2blendercodeharness-v5-executable-director"
    assert result["proposals"][0]["target_section"] == "Director prompt interpretation and event scheduling"


def test_historical_self_evolution_records_keep_round_and_case_evidence(tmp_path):
    module = load_skill_script("build_self_evolution_records")
    round_root = tmp_path / "round-root"
    round_dir = round_root / "round-04"
    round_dir.mkdir(parents=True)
    (round_dir / "patch_manifest.json").write_text(
        json.dumps(
            {
                "decision": "accepted",
                "owner": "director_prompt_interpreter",
                "patch_id": "round-04-director-prompt-elliptical-v1",
                "detected_problem": "implicit reveal and return order was not preserved",
                "fix_location": "src/videoact/director_prompt.py",
                "fix_method": "preserve evidence-backed event order",
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "attempt_report.json").write_text(
        json.dumps({"batch": {"train": ["multi-train-041", "multi-train-042"]}}),
        encoding="utf-8",
    )

    records = module.build_historical_records(round_root)

    assert [record["case_id"] for record in records] == ["multi-train-041", "multi-train-042"]
    assert all(record["round"] == 4 for record in records)
    assert all(record["findings"][0]["failure_id"] == "implicit_event_order_not_preserved" for record in records)
    assert all("implicit event order" in record["findings"][0]["message"] for record in records)
    assert all("patch_manifest.json" in record["findings"][0]["evidence"][0] for record in records)
