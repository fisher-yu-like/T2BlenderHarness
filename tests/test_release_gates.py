from __future__ import annotations


def _pilot_report(**updates):
    from videoact.release_gates import seal_report

    value = {
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
    value.update(updates)
    return seal_report(value)


def _shadow_report(**updates):
    from videoact.release_gates import seal_report

    value = {
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
    value.update(updates)
    return seal_report(value)


def test_paired_pilot_requires_exact_split_counts_and_all_evidence():
    from videoact.release_gates import validate_paired_pilot

    result = validate_paired_pilot(_pilot_report())

    assert result["status"] == "pass"
    assert result["gate_id"] == "G2"
    assert result["report_hash"]
    assert result["failures"] == []


def test_paired_pilot_fails_closed_on_missing_case_evidence():
    from videoact.release_gates import validate_paired_pilot

    result = validate_paired_pilot(_pilot_report(all_artifacts_complete=False))

    assert result["status"] == "blocked"
    assert "all_artifacts_complete" in result["failures"]


def test_shadow_report_requires_no_patch_and_stable_resume_fingerprint():
    from videoact.release_gates import validate_shadow_report

    result = validate_shadow_report(_shadow_report(patch_applied=True))

    assert result["status"] == "blocked"
    assert "patch_applied_must_be_false" in result["failures"]


def test_formal_release_requires_sealed_g0_g1_pilot_and_shadow():
    from videoact.release_gates import build_formal_release_report, seal_report

    g0 = seal_report({"status": "pass", "gate_id": "G0", "report": "all-p0-checks"})
    g1 = seal_report({"status": "pass", "gate_id": "G1", "report": "evaluator-frozen"})
    result = build_formal_release_report(g0, g1, _pilot_report(), _shadow_report())

    assert result["status"] == "pass"
    assert result["training_allowed"] is True
    assert set(result["gate_reports"]) == {"G0", "G1", "G2", "G3"}
    assert result["report_hash"]


def test_formal_release_rejects_tampered_gate_report():
    from videoact.release_gates import build_formal_release_report, seal_report

    g0 = seal_report({"status": "pass", "gate_id": "G0"})
    g1 = seal_report({"status": "pass", "gate_id": "G1"})
    pilot = _pilot_report()
    pilot["primary_outcome"] = "tampered"

    result = build_formal_release_report(g0, g1, pilot, _shadow_report())

    assert result["training_allowed"] is False
    assert result["status"] == "blocked"
    assert "G2" in result["blocking_gates"]


def test_formal_release_rechecks_embedded_gate_evidence_after_outer_reseal():
    from videoact.release_gates import (
        build_formal_release_report,
        seal_report,
        validate_formal_release_report,
    )

    release = build_formal_release_report(
        seal_report({"status": "pass", "gate_id": "G0", "probe": "g0"}),
        seal_report({"status": "pass", "gate_id": "G1", "probe": "g1"}),
        _pilot_report(),
        _shadow_report(),
    )
    release["gate_reports"]["G2"]["source_report"]["primary_outcome"] = "tampered"
    tampered = seal_report(release)

    result = validate_formal_release_report(tampered)

    assert result["training_allowed"] is False
    assert any("G2" in failure for failure in result["failures"])


def test_formal_release_requires_each_embedded_gate_to_remain_sealed():
    from videoact.release_gates import build_formal_release_report, seal_report, validate_formal_release_report

    release = build_formal_release_report(
        seal_report({"status": "pass", "gate_id": "G0"}),
        seal_report({"status": "pass", "gate_id": "G1"}),
        _pilot_report(),
        _shadow_report(),
    )
    release["gate_reports"]["G3"]["source_report"].pop("report_hash")
    tampered = seal_report(release)

    result = validate_formal_release_report(tampered)

    assert result["training_allowed"] is False
    assert any("G3" in failure for failure in result["failures"])


def test_formal_release_verification_does_not_contain_a_stale_outer_hash(tmp_path):
    import json
    import subprocess
    import sys

    from videoact.release_gates import seal_report, verify_sealed_report

    def write(name, payload):
        path = tmp_path / name
        path.write_text(json.dumps(seal_report(payload)), encoding="utf-8")
        return path

    g0 = write("g0.json", {"status": "pass", "gate_id": "G0"})
    g1 = write("g1.json", {"status": "pass", "gate_id": "G1"})
    pilot = write(
        "pilot.json",
        {
            **_pilot_report(),
        },
    )
    shadow = write("shadow.json", {**_shadow_report()})
    # The helper payloads are already sealed; seal_report removes/replaces the
    # nested hash and keeps this test focused on the CLI's outer report.
    output = tmp_path / "release.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/check_formal_release_gates.py",
            "--g0",
            str(g0),
            "--g1",
            str(g1),
            "--pilot",
            str(pilot),
            "--shadow",
            str(shadow),
            "--out",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert verify_sealed_report(report)[0] is True
    assert report["verification"].get("report_hash") is None
