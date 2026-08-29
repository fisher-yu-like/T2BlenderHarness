from __future__ import annotations

import hashlib
from pathlib import Path


def test_training_provider_wiring_uses_codex_local_agents_without_template_fallback():
    from scripts.train_real_harness import build_dynamic_codex_agents

    director, code_agent = build_dynamic_codex_agents(
        codex_command="codex-test", timeout_s=31, provider_mode="external"
    )

    assert director.mode == "dynamic"
    assert director.provider == "codex-local"
    assert code_agent.provider.command == "codex-test"
    assert code_agent.provider.timeout_s == 31


def test_training_defaults_to_the_current_codex_environment_without_external_exec():
    from scripts.train_real_harness import build_dynamic_codex_agents

    director, code_agent = build_dynamic_codex_agents()

    assert director.mode == "dynamic"
    assert director.provider == "codex-local"
    assert director.policy == "director-v3-codex-local"
    assert code_agent.model == "codex-local"
    assert code_agent.provider.__self__.__class__.__name__ == "CodexLocalProvider"


def test_dynamic_agent_index_rejects_explicit_template_mode(tmp_path: Path) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    report = audit_dynamic_agent_index(
        {"generation_mode": "template_baseline", "jobs": []},
        run_root=tmp_path,
        expected_case_ids=["case-a"],
    )

    assert report["status"] == "fail"
    assert "template" in report["reason"]


def test_dynamic_agent_index_requires_case_specific_sources(tmp_path: Path) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    source = tmp_path / "shared.py"
    source.write_text(
        'CASE_SCENE_PROFILE = {"profile_version": "codex-local-case-profile-v2"}\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    jobs = [
        {
            "case_id": case_id,
            "status": "prepared",
            "codegen_call_id": f"codex-local:{case_id}",
            "job_path": str(source),
            "code_hash": digest,
        }
        for case_id in ("case-a", "case-b")
    ]

    report = audit_dynamic_agent_index(
        {"generation_mode": "agent", "jobs": jobs},
        run_root=tmp_path,
        expected_case_ids=["case-a", "case-b"],
    )

    assert report["status"] == "fail"
    assert "all_cases_reuse_one_generated_source" in report["failures"]


def test_dynamic_agent_index_resolves_relative_job_paths_from_run_root(tmp_path: Path, monkeypatch) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    monkeypatch.chdir(tmp_path.parent)
    relative_root = tmp_path.name
    case_dir = tmp_path / "case-a"
    case_dir.mkdir()
    source = case_dir / "blender_job.py"
    source.write_text(
        'CASE_SCENE_PROFILE = {"profile_version": "codex-local-case-profile-v2"}\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    report = audit_dynamic_agent_index(
        {
            "generation_mode": "agent",
            "jobs": [
                {
                    "case_id": "case-a",
                    "status": "prepared",
                    "codegen_call_id": "codex-local:case-a",
                    "job_path": f"{relative_root}/case-a/blender_job.py",
                    "code_hash": digest,
                }
            ],
        },
        run_root=relative_root,
        expected_case_ids=["case-a"],
    )

    assert report["status"] == "pass"


def test_dynamic_agent_index_rejects_unique_but_generic_generated_source(tmp_path: Path) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    source = tmp_path / "generic.py"
    source.write_text("# unique but generic source\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    report = audit_dynamic_agent_index(
        {
            "generation_mode": "agent",
            "jobs": [
                {
                    "case_id": "case-a",
                    "status": "prepared",
                    "codegen_call_id": "codex-local:case-a",
                    "job_path": str(source),
                    "code_hash": digest,
                }
            ],
        },
        run_root=tmp_path,
        expected_case_ids=["case-a"],
    )

    assert report["status"] == "fail"
    assert "case-a:missing_case_specific_generation_profile" in report["failures"]


def test_dynamic_agent_index_rejects_profile_not_bound_to_plan_hash(tmp_path: Path) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    source = tmp_path / "wrong-profile.py"
    source.write_text(
        'CASE_SCENE_PROFILE = {"profile_version": "codex-local-case-profile-v2", "case_signature": "wrong"}\n',
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    plan_hash = "a" * 64
    report = audit_dynamic_agent_index(
        {
            "generation_mode": "agent",
            "jobs": [
                {
                    "case_id": "case-a",
                    "status": "prepared",
                    "codegen_call_id": "codex-local:case-a",
                    "director_plan_hash": plan_hash,
                    "job_path": str(source),
                    "code_hash": digest,
                }
            ],
        },
        run_root=tmp_path,
        expected_case_ids=["case-a"],
    )

    assert report["status"] == "fail"
    assert "case-a:case_profile_not_bound_to_director_plan" in report["failures"]
