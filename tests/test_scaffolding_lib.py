from __future__ import annotations


def test_scaffolding_contract_lists_required_real_artifacts() -> None:
    from blender.lib.scaffolding import build_runtime_contract, validate_runtime_contract

    contract = build_runtime_contract(
        director_plan_hash="a" * 64,
        required_entities=["actor_a", "red_cup"],
        required_events=["carry_01"],
        required_camera_events=["carry_01"],
    )

    assert "proxy.blend" in contract["required_artifacts"]
    assert "telemetry.json" in contract["required_artifacts"]
    assert "frames/index.json" in contract["required_artifacts"]
    assert validate_runtime_contract(contract) == []


def test_scaffolding_contract_rejects_missing_traceability() -> None:
    from blender.lib.scaffolding import validate_runtime_contract

    failures = validate_runtime_contract({"required_artifacts": ["proxy.blend"]})

    assert "missing_director_plan_hash" in failures
    assert "missing_required_artifact:telemetry.json" in failures
