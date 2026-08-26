from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_director_skill_is_t2blendercodeharness_protocol_not_autodesign():
    skill = _read("skills/director-agent/SKILL.md")

    assert "name: director-agent" in skill
    assert "DirectorRequest" in skill
    assert "DirectorPlan" in skill
    assert "actor_a" in skill
    assert "evidence" in skill and "uncertainty" in skill
    assert "provider-assisted" in skill
    assert "scene_contract_repair" in skill
    assert "director_trajectory" in skill
    assert "autodesign" not in skill.lower()


def test_multi_training_skill_matches_frozen_protocol():
    skill = _read("skills/t2blendercodeharness-training/SKILL.md")

    for phrase in (
        "trajectory-v4-multi",
        "multi-five-rounds-v1",
        "10 train",
        "10 paired dev",
        "five attempts",
        "12 workers",
        "gpt-5.6-Luna",
        "gpt-5.6-Terra",
        "director_plan_score",
        "task_score",
        "realism_score",
        "NOT_RENDERED",
        "proxy video",
    ):
        assert phrase in skill


def test_component_skills_declare_director_handoff_and_one_owner_rules():
    harness = _read("skills/t2blendercodeharness/SKILL.md")
    scene = _read("skills/scene-contract/SKILL.md")
    trajectory = _read("skills/trajectory-planner/SKILL.md")
    proxy = _read("skills/blender-proxy-executor/SKILL.md")
    evolution = _read("skills/harness-evolution/SKILL.md")

    assert "DirectorAgent" in harness
    assert "DirectorAgent" in scene
    assert "DirectorAgent" in trajectory
    assert "DirectorPlan" in proxy
    assert "director_plan_score" in evolution
    assert "one-owner" in evolution


def test_capability_check_declares_director_and_multi_dataset_checks():
    capability = _read("skills/t2blendercodeharness/scripts/capability_check.py")

    assert "director_agent" in capability
    assert "director_metrics" in capability
    assert "validate_multi_entity_dataset.py" in capability
