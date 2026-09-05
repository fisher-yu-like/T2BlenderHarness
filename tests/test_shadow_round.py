from __future__ import annotations

def _cases(train: int = 60, dev: int = 60):
    return [
        {"case_id": f"train-{index:03d}", "split": "train", "prompt": f"private train prompt {index}"}
        for index in range(train)
    ] + [
        {"case_id": f"dev-{index:03d}", "split": "dev", "prompt": f"private dev prompt {index}"}
        for index in range(dev)
    ]


def _success(_context):
    return {
        "status": "pass",
        "artifact_complete": True,
        "artifact_completion_slo_pass": True,
        "memory_recorded": True,
        "cost_slo_pass": True,
        "hard_failure_slo_pass": True,
        "judge_unavailable_slo_pass": True,
        "controller_test_access_blocked": True,
        "fingerprints": {"harness": "h1", "evaluator": "e1"},
        "evidence_refs": ["case-evidence.json"],
    }


def test_shadow_round_is_exact_no_patch_and_does_not_persist_prompt(tmp_path):
    from videoact.release_gates import validate_shadow_report
    from videoact.shadow_round import run_shadow_round

    report = run_shadow_round(
        _cases(),
        runner=_success,
        output_dir=tmp_path,
        experiment_fingerprint="experiment-1",
        component_fingerprints={"harness": "h1", "evaluator": "e1"},
    )

    assert report["status"] == "pass"
    assert report["patch_applied"] is False
    assert report["case_count"] == 120
    assert report["split_case_counts"] == {"train": 60, "dev": 60}
    assert validate_shadow_report(report)["status"] == "pass"
    progress = (tmp_path / "shadow_round_progress.jsonl").read_text(encoding="utf-8")
    assert "private train prompt" not in progress
    assert "private dev prompt" not in progress


def test_shadow_resume_skips_passed_cases_and_retries_failed_case(tmp_path):
    from videoact.shadow_round import run_shadow_round

    calls = []
    fail_once = {"dev-001"}

    def runner(context):
        case_id = context["case_id"]
        calls.append(case_id)
        if case_id in fail_once:
            fail_once.remove(case_id)
            return {"status": "blocked"}
        return _success(context)

    records = _cases()
    first = run_shadow_round(
        records,
        runner=runner,
        output_dir=tmp_path,
        experiment_fingerprint="experiment-1",
        component_fingerprints={"harness": "h1", "evaluator": "e1"},
        expected_train=60,
        expected_dev=60,
    )
    assert first["status"] == "blocked"
    assert calls == [f"train-{index:03d}" for index in range(60)] + [
        f"dev-{index:03d}" for index in range(60)
    ]

    second = run_shadow_round(
        records,
        runner=runner,
        output_dir=tmp_path,
        experiment_fingerprint="experiment-1",
        component_fingerprints={"harness": "h1", "evaluator": "e1"},
        expected_train=60,
        expected_dev=60,
    )
    assert second["status"] == "pass"
    assert second["resumed"] is True
    assert calls[-1] == "dev-001"
    assert calls.count("train-000") == 1
    assert calls.count("dev-000") == 1
    assert calls.count("dev-001") == 2


def test_shadow_round_fails_closed_without_explicit_fingerprint(tmp_path):
    from videoact.shadow_round import run_shadow_round

    report = run_shadow_round(
        _cases(),
        runner=_success,
        output_dir=tmp_path,
    )

    assert report["status"] == "blocked"
    assert "fingerprints_not_stable" in report["failures"]


def test_paired_pilot_runner_builds_g2_evidence():
    from videoact.release_gates import validate_paired_pilot
    from videoact.shadow_round import run_paired_pilot

    def runner(_context):
        return {
            "status": "pass",
            "all_artifacts_complete": True,
            "trusted_observer_complete": True,
            "blind_review_complete": True,
            "disagreement_audit_complete": True,
            "paired_outcome_registered": True,
        }

    report = run_paired_pilot(
        _cases(train=10, dev=10),
        runner=runner,
        primary_outcome="task_score",
        noninferiority_margin=-1.0,
        hard_failure_rule="no_regression",
    )
    assert report["status"] == "pass"
    assert validate_paired_pilot(report)["status"] == "pass"


def test_g4_generalization_report_can_join_formal_release():
    from videoact.release_gates import (
        build_formal_release_report,
        build_generalization_gate_report,
        seal_report,
        validate_formal_release_report,
    )

    frozen = {
        "status": "pass",
        "case_count": 60,
        "fingerprint": "frozen-1",
        "coverage_matrix": {"ood_unseen_dimensions": {"case_count": 60}},
        "leakage_audit": {
            "status": "pass",
            "collisions": [],
            "collision_counts": {
                "case_id": 0,
                "exact_prompt": 0,
                "normalized_prompt": 0,
                "semantic_near_duplicate": 0,
                "source_family": 0,
                "source_identity": 0,
            },
        },
    }
    g4 = build_generalization_gate_report(
        frozen,
        controller_access_report={
            "controller_test_access_blocked": True,
            "baseline_final_only": True,
            "test_metrics_absent_from_controller": True,
        },
        outcome_report={
            "paired_delta_reported": True,
            "ci_reported": True,
            "effect_size_reported": True,
        },
    )
    assert g4["status"] == "pass"
    pilot = seal_report({
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
    })
    shadow = seal_report({
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
    })
    release = build_formal_release_report(
        seal_report({"status": "pass", "gate_id": "G0"}),
        seal_report({"status": "pass", "gate_id": "G1"}),
        pilot,
        shadow,
        g4,
    )
    assert release["gate_reports"]["G4"]["status"] == "pass"
    assert release["status"] == "pass"
    assert validate_formal_release_report(release)["training_allowed"] is True
