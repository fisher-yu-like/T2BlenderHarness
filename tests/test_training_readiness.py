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
