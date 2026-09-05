from __future__ import annotations


def _proposal(owner: str = "director_camera", **updates):
    payload = {
        "proposal_id": "impact-001",
        "owner": owner,
        "root_cause_id": "camera_visibility",
        "target_obligations": ["camera_visibility"],
    }
    payload.update(updates)
    return payload


def test_camera_impact_requires_camera_domain_change_and_obligation_trace():
    from videoact.patch_impact import build_patch_impact_proof, validate_patch_impact

    proof = build_patch_impact_proof(
        _proposal(),
        {"camera_plan": {"shot": "wide"}, "camera_telemetry": {"visible": 0}},
        {"camera_plan": {"shot": "close"}, "camera_telemetry": {"visible": 1}, "evidence_refs": ["obligation_matrix.json:camera_visibility"]},
        changed_files=["src/videoact/director_camera.py"],
        production_call_sites_changed=["compose_camera_plan"],
    )

    assert proof.status == "pass"
    assert proof.camera_plan_changed is True
    assert proof.causal_chain_complete is True
    assert validate_patch_impact(proof).status == "pass"


def test_camera_source_diff_without_downstream_change_is_no_effect():
    from videoact.patch_impact import build_patch_impact_proof

    proof = build_patch_impact_proof(
        _proposal(),
        {"camera_plan": {"shot": "wide"}, "camera_telemetry": {"visible": 0}},
        {"camera_plan": {"shot": "wide"}, "camera_telemetry": {"visible": 0}},
        changed_files=["src/videoact/director_camera.py"],
        production_call_sites_changed=["compose_camera_plan"],
    )

    assert proof.status == "no_effect_patch"
    assert proof.causal_chain_complete is False


def test_director_and_code_owners_have_distinct_downstream_requirements():
    from videoact.patch_impact import build_patch_impact_proof

    director = build_patch_impact_proof(
        _proposal("director_trajectory", root_cause_id="trajectory_execution", target_obligations=["trajectory"]),
        {"plan_hash": "p" * 64, "obligation_hash": "o" * 64},
        {"plan_hash": "q" * 64, "obligation_hash": "r" * 64, "evidence_refs": ["obligation_matrix.json:trajectory"]},
        changed_files=["src/videoact/director_trajectory.py"],
        production_call_sites_changed=["compose_trajectory"],
    )
    code = build_patch_impact_proof(
        _proposal("blender_code_agent", root_cause_id="code_generation"),
        {"source": "def build():\n    return box()\n"},
        {"source": "def build():\n    return cylinder()\n"},
        changed_files=["src/videoact/blender_code_agent.py"],
        production_call_sites_changed=["BlenderCodeAgent.generate"],
    )

    assert director.status == "pass"
    assert director.plan_hash_changed is True
    assert code.status == "pass"
    assert code.code_ast_changed is False
    assert code.code_call_sites_changed is True


def test_cache_reuse_and_untraceable_metric_are_rejected():
    from videoact.patch_impact import build_patch_impact_proof

    reused = build_patch_impact_proof(
        _proposal("blender_code_agent", root_cause_id="code_generation"),
        {"source": "def build():\n    return 1\n"},
        {"source": "def build():\n    return 2\n"},
        changed_files=["src/videoact/blender_code_agent.py"],
        production_call_sites_changed=["BlenderCodeAgent.generate"],
    )
    untraceable = build_patch_impact_proof(
        _proposal(),
        {"camera_plan": {"shot": "wide"}},
        {"camera_plan": {"shot": "close"}, "target_metric_delta": 4.0},
        changed_files=["src/videoact/director_camera.py"],
        production_call_sites_changed=["compose_camera_plan"],
    )

    assert reused.status == "rejected"
    assert reused.cache_reuse_detected is True
    assert untraceable.status == "rejected"
    assert "obligation" in untraceable.reason


def test_missing_production_call_site_is_rejected():
    from videoact.patch_impact import build_patch_impact_proof

    proof = build_patch_impact_proof(
        _proposal(),
        {"camera_plan": {"shot": "wide"}},
        {"camera_plan": {"shot": "close"}},
        changed_files=["src/videoact/director_camera.py"],
    )

    assert proof.status == "rejected"
    assert "production call-site" in proof.reason


def test_executor_can_require_complete_impact_proof(tmp_path):
    from videoact.patch_executor import PatchExecutor
    from videoact.patch_impact import build_patch_impact_proof

    source = tmp_path / "src" / "videoact" / "director_camera.py"
    source.parent.mkdir(parents=True)
    source.write_text("def compose_camera_plan():\n    return 'old'\n", encoding="utf-8")
    proposal = {
        "proposal_id": "impact-executor-001",
        "owner": "director_camera",
        "root_cause_id": "camera_visibility",
        "affected_files": ["src/videoact/director_camera.py"],
        "source_split": "train",
        "target_obligations": ["camera_visibility"],
    }
    proof = build_patch_impact_proof(
        proposal,
        {"camera_plan": {"shot": "wide"}},
        {"camera_plan": {"shot": "close"}, "evidence_refs": ["obligation_matrix.json:camera_visibility"]},
        changed_files=["src/videoact/director_camera.py"],
        production_call_sites_changed=["compose_camera_plan"],
    )
    result = PatchExecutor(
        repo_root=tmp_path,
        owner_challenge_runner=lambda: True,
        unit_test_runner=lambda: True,
        require_impact_proof=True,
    ).execute(
        proposal,
        {"file_contents": {"src/videoact/director_camera.py": "def compose_camera_plan():\n    return 'new'\n"}},
        impact_proof=proof,
    )

    assert result["status"] == "accepted"
    assert result["impact_proof"]["status"] == "pass"


def test_acceptance_gate_requires_impact_proof_when_formal_flag_is_set():
    from videoact.outer_loop import evaluate_candidate

    before = {"train_score": 70.0, "dev_score": 68.0}
    after = {"train_score": 72.0, "dev_score": 68.0}
    train = {"patch_impact_proof_required": True}
    dev = {}
    missing = evaluate_candidate(before, after, train, dev)
    assert missing.accepted is False
    assert "patch_impact_proof" in missing.failed_checks

    passed = evaluate_candidate(
        before,
        after,
        {"patch_impact_proof_required": True},
        {},
        impact_proof={
            "edit_id": "impact-001",
            "owner": "director_camera",
            "source_diff_present": True,
            "production_call_sites_changed": ["compose_camera_plan"],
            "changed_files": ["src/videoact/director_camera.py"],
            "camera_plan_changed": True,
            "target_obligation_ids": ["camera_visibility"],
            "status": "pass",
            "causal_chain_complete": True,
            "reason": "verified",
        },
    )
    assert passed.checks["patch_impact_proof"] is True
