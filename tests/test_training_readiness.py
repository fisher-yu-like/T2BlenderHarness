from __future__ import annotations

from pathlib import Path


def _automated_passes() -> dict[str, str]:
    return {
        "full_test": "pass",
        "capability": "pass",
        "dataset": "pass",
        "frozen_eval": "pass",
    }


def test_readiness_blocks_template_smoke_and_missing_external_gates() -> None:
    from scripts.check_training_readiness import build_training_readiness

    report = build_training_readiness(
        automated_checks=_automated_passes(),
        real_blender_smoke={"status": "pass", "generation_mode": "template_baseline"},
        golden_review="pending",
        dynamic_agent_provider="blocked",
        paired_gate="pending",
    )

    assert report["training_allowed"] is False
    assert report["gates"]["real_blender_smoke"]["status"] == "blocked"
    assert report["gates"]["golden_review"]["status"] == "pending"
    assert report["gates"]["dynamic_agent_provider"]["status"] == "blocked"
    assert report["gates"]["paired_gate"]["status"] == "pending"
    assert report["numeric_substitutions"] == []


def test_readiness_allows_only_explicitly_passing_agent_evidence() -> None:
    from scripts.check_training_readiness import build_training_readiness

    report = build_training_readiness(
        automated_checks=_automated_passes(),
        real_blender_smoke={"status": "pass", "generation_mode": "agent", "artifact_status": "complete"},
        golden_review={"status": "pass", "annotators_per_sample": 2},
        dynamic_agent_provider={"status": "pass", "director": "pass", "blender_code": "pass"},
        paired_gate={"status": "pass", "case_count": 20},
    )

    assert report["training_allowed"] is True
    assert all(gate["status"] == "pass" for gate in report["gates"].values())


def test_readiness_rejects_unknown_or_numeric_gate_substitutes() -> None:
    from scripts.check_training_readiness import build_training_readiness

    report = build_training_readiness(
        automated_checks={**_automated_passes(), "full_test": 100},
        real_blender_smoke={"status": "pass", "generation_mode": "agent"},
        golden_review=100,
        dynamic_agent_provider="pass",
        paired_gate="pass",
    )

    assert report["training_allowed"] is False
    assert report["numeric_substitutions"]
    assert report["gates"]["golden_review"]["status"] == "blocked"


def test_readiness_reads_nested_pytest_junit_totals(tmp_path: Path) -> None:
    from scripts.check_training_readiness import _read_report

    report_path = tmp_path / "pytest.xml"
    report_path.write_text(
        '<testsuites><testsuite tests="375" failures="0" errors="0" skipped="2" /></testsuites>',
        encoding="utf-8",
    )

    report = _read_report(report_path)

    assert report is not None
    assert report["status"] == "pass"
    assert report["tests"] == 375
    assert report["skipped"] == 2


def test_readiness_requires_complete_agent_artifact_and_both_provider_stages() -> None:
    from scripts.check_training_readiness import build_training_readiness

    report = build_training_readiness(
        automated_checks=_automated_passes(),
        real_blender_smoke={"status": "pass", "generation_mode": "agent"},
        golden_review={"status": "pass", "annotators_per_sample": 2},
        dynamic_agent_provider={"status": "pass", "director": "pass", "blender_code": "fail"},
        paired_gate={"status": "pass", "case_count": 20},
    )

    assert report["training_allowed"] is False
    assert report["gates"]["real_blender_smoke"]["status"] == "blocked"
    assert report["gates"]["dynamic_agent_provider"]["status"] == "blocked"


def test_formal_model_readiness_requires_distinct_generator_and_judge_snapshots() -> None:
    from scripts.check_training_readiness import build_training_readiness

    base = build_training_readiness(
        automated_checks=_automated_passes(),
        real_blender_smoke={"status": "pass", "generation_mode": "agent", "artifact_status": "complete"},
        golden_review={"status": "pass", "annotators_per_sample": 2},
        dynamic_agent_provider={
            "status": "pass",
            "provider_mode": "model",
            "director": "pass",
            "blender_code": "pass",
            "generator_model_id": "codex-v1",
            "primary_judge_model_id": "codex-v1",
            "audit_judge_model_id": "terra-v1",
        },
        paired_gate={"status": "pass", "case_count": 20},
    )

    assert base["training_allowed"] is False
    assert base["gates"]["dynamic_agent_provider"]["reason"] == "generator_and_judge_model_snapshots_must_be_distinct"


def test_explicit_rule_template_provider_cannot_satisfy_readiness() -> None:
    from scripts.check_training_readiness import build_training_readiness

    report = build_training_readiness(
        automated_checks=_automated_passes(),
        real_blender_smoke={"status": "pass", "generation_mode": "agent", "artifact_status": "complete"},
        golden_review={"status": "pass", "annotators_per_sample": 2},
        dynamic_agent_provider={
            "status": "pass",
            "provider_mode": "rule_template_baseline",
            "director": "pass",
            "blender_code": "pass",
        },
        paired_gate={"status": "pass", "case_count": 20},
    )

    assert report["training_allowed"] is False
    assert report["gates"]["dynamic_agent_provider"]["reason"] == "rule_template_baseline_is_diagnostic_only"


def test_readiness_rejects_historical_self_built_dataset() -> None:
    from scripts.check_training_readiness import _dataset_evidence

    report = _dataset_evidence("dataset/trajectory-v5-agent-codegen")

    assert report["status"] == "fail"
    assert "benchmark_prompt_index" in report["reason"]


def test_dataset_readiness_ignores_self_reference() -> None:
    from scripts.check_training_readiness import _dataset_evidence

    root = Path("dataset/vbench2-agent-training-index-v1")
    report = _dataset_evidence(root, [root])

    assert report["status"] == "pass"


def test_readiness_reports_incomplete_golden_annotations_as_pending(monkeypatch, tmp_path: Path) -> None:
    import scripts.check_training_readiness as readiness

    bundle = tmp_path / "golden"
    bundle.mkdir()
    monkeypatch.setattr(
        readiness,
        "_read_report",
        lambda _path: None,
    )
    monkeypatch.setattr(
        "scripts.validate_golden_review_set.validate_golden_review_set",
        lambda _path: {
            "status": "fail",
            "errors": [
                "videos without human scores: [('case-001', 'sample_a')]",
                "each blind video needs at least two independent annotators: [('case-001', 'sample_a')]",
            ],
        },
    )

    report = readiness.build_training_readiness_from_project(
        project_root=tmp_path,
        golden_root=bundle,
    )

    assert report["gates"]["golden_review"]["status"] == "pending"
    assert report["gates"]["golden_review"]["reason"] == "golden_annotations_pending"


def test_readiness_reports_comparison_only_golden_bundle_as_pending(monkeypatch, tmp_path: Path) -> None:
    import scripts.check_training_readiness as readiness

    bundle = tmp_path / "golden"
    bundle.mkdir()
    monkeypatch.setattr(
        "scripts.validate_golden_review_set.validate_golden_review_set",
        lambda _path: {
            "status": "fail",
            "errors": ["render prompt differs from the displayed source prompt; bundle is comparison-only"],
        },
    )

    report = readiness.build_training_readiness_from_project(project_root=tmp_path, golden_root=bundle)

    assert report["gates"]["golden_review"]["status"] == "pending"
    assert report["gates"]["golden_review"]["reason"] == "golden_bundle_requires_exact_prompt_rerender"


def test_training_entry_requires_a_passing_readiness_report(tmp_path: Path) -> None:
    from scripts.train_real_harness import require_training_readiness

    blocked = tmp_path / "blocked.json"
    blocked.write_text('{"training_allowed": false}', encoding="utf-8")
    try:
        require_training_readiness(blocked)
    except ValueError as exc:
        assert "training_allowed=true" in str(exc)
    else:  # pragma: no cover - assertion documents the fail-closed contract
        raise AssertionError("blocked readiness report unexpectedly enabled training")

    allowed = tmp_path / "allowed.json"
    allowed.write_text('{"training_allowed": true}', encoding="utf-8")
    assert require_training_readiness(allowed)["training_allowed"] is True


def test_formal_training_entry_requires_verified_g0_to_g3_release_report(tmp_path: Path) -> None:
    from scripts.train_real_harness import require_formal_training_release
    from videoact.release_gates import build_formal_release_report, seal_report

    readiness = tmp_path / "readiness.json"
    readiness.write_text('{"training_allowed": true}', encoding="utf-8")
    g0 = seal_report({"status": "pass", "gate_id": "G0"})
    g1 = seal_report({"status": "pass", "gate_id": "G1"})
    pilot = seal_report(
        {
            "status": "pass",
            "gate_id": "G2",
            "case_count": 20,
            "split_case_counts": {"train": 10, "dev": 10},
            "all_artifacts_complete": True,
            "trusted_observer_complete": True,
            "blind_review_complete": True,
            "disagreement_audit_complete": True,
            "paired_outcome_registered": True,
            "baseline_arm": "rule_template_baseline",
            "candidate_arm": "model_driven_candidate",
            "primary_outcome": "task_score",
            "noninferiority_margin": -1.0,
            "hard_failure_rule": "no_regression",
        }
    )
    shadow = seal_report(
        {
            "status": "pass",
            "gate_id": "G3",
            "case_count": 120,
            "split_case_counts": {"train": 60, "dev": 60},
            "patch_applied": False,
            "resume_verified": True,
            "fingerprints_stable": True,
            "memory_complete": True,
            "cost_slo_pass": True,
            "artifact_completion_slo_pass": True,
            "hard_failure_slo_pass": True,
            "judge_unavailable_slo_pass": True,
        }
    )
    release = build_formal_release_report(g0, g1, pilot, shadow)
    release_path = tmp_path / "release.json"
    release_path.write_text(__import__("json").dumps(release), encoding="utf-8")

    result = require_formal_training_release(readiness, release_path)

    assert result["training_allowed"] is True


def test_formal_training_entry_rejects_readiness_only(tmp_path: Path) -> None:
    from scripts.train_real_harness import require_formal_training_release

    readiness = tmp_path / "readiness.json"
    readiness.write_text('{"training_allowed": true}', encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="formal release"):
        require_formal_training_release(readiness, tmp_path / "missing-release.json")


def test_readiness_can_reference_and_verify_a_formal_release_report(tmp_path: Path) -> None:
    import json

    from scripts.check_training_readiness import build_training_readiness_from_project
    from videoact.release_gates import build_formal_release_report, seal_report

    release = build_formal_release_report(
        seal_report({"status": "pass", "gate_id": "G0"}),
        seal_report({"status": "pass", "gate_id": "G1"}),
        seal_report(
            {
                "status": "pass",
                "gate_id": "G2",
                "case_count": 20,
                "split_case_counts": {"train": 10, "dev": 10},
                "all_artifacts_complete": True,
                "trusted_observer_complete": True,
                "blind_review_complete": True,
                "disagreement_audit_complete": True,
                "paired_outcome_registered": True,
                "baseline_arm": "rule_template_baseline",
                "candidate_arm": "model_driven_candidate",
                "primary_outcome": "task_score",
                "noninferiority_margin": -1.0,
                "hard_failure_rule": "no_regression",
            }
        ),
        seal_report(
            {
                "status": "pass",
                "gate_id": "G3",
                "case_count": 120,
                "split_case_counts": {"train": 60, "dev": 60},
                "patch_applied": False,
                "resume_verified": True,
                "fingerprints_stable": True,
                "memory_complete": True,
                "cost_slo_pass": True,
                "artifact_completion_slo_pass": True,
                "hard_failure_slo_pass": True,
                "judge_unavailable_slo_pass": True,
            }
        ),
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")

    report = build_training_readiness_from_project(
        project_root=tmp_path,
        formal_release_report=release_path,
    )

    assert report["gates"]["formal_release"]["status"] == "pass"
    assert report["formal_release"]["gate_report_hashes"]["G0"]


def test_readiness_blocks_a_tampered_formal_release_report(tmp_path: Path) -> None:
    import json

    from scripts.check_training_readiness import build_training_readiness_from_project

    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps({"status": "pass", "training_allowed": True, "gate_reports": {}}),
        encoding="utf-8",
    )

    report = build_training_readiness_from_project(
        project_root=tmp_path,
        formal_release_report=release_path,
    )

    assert report["training_allowed"] is False
    assert report["gates"]["formal_release"]["status"] == "blocked"


def test_diagnostic_training_allows_only_human_pending_blockers(tmp_path: Path) -> None:
    from scripts.train_real_harness import require_diagnostic_training_readiness

    report = tmp_path / "readiness.json"
    report.write_text(
        '{'
        '"training_allowed": false,'
        '"numeric_substitutions": [],'
        '"gates": {'
        '"full_test": {"status": "pass"},'
        '"capability": {"status": "pass"},'
        '"dataset": {"status": "pass"},'
        '"frozen_eval": {"status": "pass"},'
        '"real_blender_smoke": {"status": "pass"},'
        '"dynamic_agent_provider": {"status": "pass"},'
        '"golden_review": {"status": "pending"},'
        '"paired_gate": {"status": "pending"}'
        '}'
        '}',
        encoding="utf-8",
    )

    result = require_diagnostic_training_readiness(report)

    assert result["mode"] == "diagnostic_precalibration"
    assert result["formal_training_allowed"] is False
    assert result["visual_scores_permitted"] is False
    assert result["deferred_gates"] == ["golden_review", "paired_gate"]


def test_diagnostic_training_rejects_non_human_readiness_blocker(tmp_path: Path) -> None:
    import pytest

    from scripts.train_real_harness import require_diagnostic_training_readiness

    report = tmp_path / "readiness.json"
    report.write_text(
        '{"training_allowed": false, "numeric_substitutions": [], "gates": {'
        '"full_test": {"status": "fail"}, "golden_review": {"status": "pending"}, '
        '"paired_gate": {"status": "pending"}}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="diagnostic training is blocked"):
        require_diagnostic_training_readiness(report)
