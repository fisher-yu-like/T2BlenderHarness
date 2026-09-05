from __future__ import annotations

import json
from pathlib import Path

import pytest


def _contract_inputs(**overrides):
    values = {
        "experiment_id": "exp-20260901-a",
        "parent_harness_version": "harness-v5",
        "split_cases": {
            "train": [{"case_id": "train-001", "prompt_hash": "a" * 64, "seed": 11}],
            "dev": [{"case_id": "dev-001", "prompt_hash": "b" * 64, "seed": 22}],
            "test": [{"case_id": "test-001", "prompt_hash": "c" * 64, "seed": 33}],
        },
        "dataset_fingerprint": "dataset-v1",
        "evaluator_fingerprint": {"prompt": "judge prompt v1", "model": "judge-v1"},
        "observer_fingerprint": "observer-source-v1",
        "provider_fingerprint": {"director": "provider-v1", "codegen": "provider-v1"},
        "blender_binary_fingerprint": b"blender-v1",
        "render_settings": {"fps": 24, "resolution": [512, 512]},
        "scoring_policy": {"policy": "scoring-v7", "threshold": 0.8},
        "judge": {"prompt": "judge prompt v1", "model": "judge-v1"},
        "frame_sampler": {"version": "event-aligned-uniform-v1", "max_frames": 8},
        "acceptance_margin": -0.05,
        "test_unlock_milestones": ["G4_final_evaluation"],
    }
    values.update(overrides)
    return values


def test_contract_bundle_is_recomputable_and_tracks_all_frozen_identity_inputs(tmpdir) -> None:
    from videoact.real_artifacts import build_experiment_contract, write_experiment_contract_bundle

    root = Path(str(tmpdir))
    contract = build_experiment_contract(**_contract_inputs())
    paths = write_experiment_contract_bundle(contract, root)

    assert {path.name for path in paths.values()} == {
        "experiment_contract.json",
        "baseline_manifest.json",
        "split_access_policy.json",
        "frozen_component_hashes.json",
    }
    baseline = json.loads((root / "baseline_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "frozen_component_hashes.json").read_text(encoding="utf-8"))
    assert baseline["train"]["case_ids"] == ["train-001"]
    assert baseline["dev"]["case_ids"] == ["dev-001"]
    assert baseline["final_test_evaluation"]["case_ids"] == ["test-001"]
    assert frozen["experiment_fingerprint"] == contract.experiment_fingerprint
    assert frozen["component_hashes"] == contract.frozen_component_hashes

    for changed in (
        {"evaluator_fingerprint": {"prompt": "judge prompt v2", "model": "judge-v1"}},
        {"observer_fingerprint": "observer-source-v2"},
        {"blender_binary_fingerprint": b"blender-v2"},
        {"dataset_fingerprint": "dataset-v2"},
        {"test_unlock_milestones": ["G5_revised_final_evaluation"]},
        {
            "split_cases": {
                "train": [{"case_id": "train-001", "prompt_hash": "a" * 64, "seed": 12}],
                "dev": [{"case_id": "dev-001", "prompt_hash": "b" * 64, "seed": 22}],
                "test": [{"case_id": "test-001", "prompt_hash": "c" * 64, "seed": 33}],
            }
        },
    ):
        assert build_experiment_contract(**_contract_inputs(**changed)).experiment_fingerprint != contract.experiment_fingerprint


def test_contract_rejects_test_identity_in_a_patch_proposal_and_incompatible_comparisons() -> None:
    from videoact.real_artifacts import (
        build_experiment_contract,
        compare_experiment_contracts,
        validate_proposal_split_access,
    )

    contract = build_experiment_contract(**_contract_inputs())

    with pytest.raises(ValueError, match="test"):
        validate_proposal_split_access(
            {"source_split": "train", "source_case_ids": ["train-001", "test-001"]},
            contract,
        )
    with pytest.raises(ValueError, match="test"):
        validate_proposal_split_access(
            {"source_split": "train", "evidence": {"prompt_hash": "c" * 64}},
            contract,
        )
    with pytest.raises(ValueError, match="test"):
        validate_proposal_split_access(
            {
                "source_split": "train",
                "source_case_ids": ["train-001"],
                "controller_context": {"ranked_candidates": [{"case_id": "test-001"}]},
            },
            contract,
        )

    changed = build_experiment_contract(**_contract_inputs(observer_fingerprint="observer-source-v2"))
    comparison = compare_experiment_contracts(contract, changed)
    assert comparison["compatible"] is False
    with pytest.raises(ValueError, match="incompatible"):
        compare_experiment_contracts(contract, changed, require_compatible=True)

    different_experiment = build_experiment_contract(
        **_contract_inputs(experiment_id="exp-20260901-b")
    )
    with pytest.raises(ValueError, match="incompatible"):
        compare_experiment_contracts(contract, different_experiment, require_compatible=True)


def test_contract_rejects_tampered_component_values_with_stale_hashes(tmpdir) -> None:
    from videoact.real_artifacts import build_experiment_contract, load_experiment_contract

    contract = build_experiment_contract(**_contract_inputs())
    payload = contract.model_dump(mode="json")
    payload["acceptance_margin"] = -0.50
    contract_path = Path(str(tmpdir)) / "experiment_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="component hash"):
        load_experiment_contract(contract_path)


def test_contract_locks_scoring_thresholds_until_test_unlock() -> None:
    from videoact.real_artifacts import build_experiment_contract, validate_contract_revision

    contract = build_experiment_contract(**_contract_inputs())
    revised = build_experiment_contract(
        **_contract_inputs(scoring_policy={"policy": "scoring-v7", "threshold": 0.9})
    )

    with pytest.raises(ValueError, match="frozen threshold"):
        validate_contract_revision(contract, revised, test_unlocked=False)
    assert validate_contract_revision(contract, revised, test_unlocked=True)["changed_threshold_fields"] == ["scoring_policy"]


def test_contract_bundle_cannot_be_overwritten_with_a_new_identity(tmpdir) -> None:
    from videoact.real_artifacts import build_experiment_contract, write_experiment_contract_bundle

    root = Path(str(tmpdir))
    write_experiment_contract_bundle(build_experiment_contract(**_contract_inputs()), root)
    changed = build_experiment_contract(**_contract_inputs(experiment_id="exp-20260901-b"))

    with pytest.raises(ValueError, match="immutable"):
        write_experiment_contract_bundle(changed, root)


def test_contract_bundle_rejects_tampered_derived_artifact(tmpdir) -> None:
    from videoact.real_artifacts import build_experiment_contract, load_experiment_contract, write_experiment_contract_bundle

    root = Path(str(tmpdir))
    write_experiment_contract_bundle(build_experiment_contract(**_contract_inputs()), root)
    baseline_path = root / "baseline_manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["train"]["case_ids"] = ["test-001"]
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="bundle artifact mismatch"):
        load_experiment_contract(root)


def test_contract_rejects_test_identity_embedded_in_free_text() -> None:
    from videoact.real_artifacts import build_experiment_contract, validate_proposal_split_access

    contract = build_experiment_contract(**_contract_inputs())
    with pytest.raises(ValueError, match="test"):
        validate_proposal_split_access(
            {
                "source_split": "train",
                "source_case_ids": ["train-001"],
                "notes": "see https://example.invalid/test-001/case for context",
            },
            contract,
        )


def test_contract_runtime_binding_rejects_changed_dataset_or_blender(tmpdir) -> None:
    from videoact.real_artifacts import build_experiment_contract, validate_contract_runtime_inputs

    root = Path(str(tmpdir))
    blender = root / "blender.exe"
    blender.write_bytes(b"blender-v1")
    contract = build_experiment_contract(
        **_contract_inputs(blender_binary_fingerprint=blender)
    )
    with pytest.raises(ValueError, match="dataset"):
        validate_contract_runtime_inputs(contract, dataset_fingerprint="dataset-v2")
    blender.write_bytes(b"blender-v2")
    with pytest.raises(ValueError, match="blender_binary"):
        validate_contract_runtime_inputs(contract, blender_binary=blender)


def test_formal_six_round_runner_refuses_to_create_root_without_contract(tmpdir) -> None:
    from scripts.train_real_harness import run_six_round_protocol

    tmp_path = Path(str(tmpdir))
    root = tmp_path / "formal-rounds"
    with pytest.raises(ValueError, match="experiment contract"):
        run_six_round_protocol(
            root,
            dataset_root=tmp_path / "missing-dataset",
            harness_version="harness-v5",
            evaluator_version="evaluator-v1",
            blender_bin="blender",
            workers=1,
            timeout_s=1,
            vlm_model="judge-v1",
            formal=True,
        )

    assert not root.exists()


def test_readiness_can_carry_the_same_contract_identity() -> None:
    from scripts.check_training_readiness import build_training_readiness
    from videoact.real_artifacts import build_experiment_contract

    contract = build_experiment_contract(**_contract_inputs())
    report = build_training_readiness(
        automated_checks={name: "pass" for name in ("full_test", "capability", "dataset", "frozen_eval")},
        real_blender_smoke={"status": "pass", "generation_mode": "agent", "artifact_status": "complete"},
        golden_review={"status": "pass", "annotators_per_sample": 2},
        dynamic_agent_provider={"status": "pass", "director": "pass", "blender_code": "pass"},
        paired_gate="pass",
        experiment_contract=contract.model_dump(mode="json"),
    )

    assert report["gates"]["experiment_contract"]["status"] == "pass"
    assert report["experiment_contract"]["experiment_id"] == "exp-20260901-a"


def test_readiness_and_formal_runner_reject_stale_or_different_contract_identity(tmpdir) -> None:
    from scripts.check_training_readiness import build_training_readiness
    from scripts.train_real_harness import require_experiment_contract
    from videoact.real_artifacts import build_experiment_contract, write_experiment_contract_bundle

    checks = {name: "pass" for name in ("full_test", "capability", "dataset", "frozen_eval")}
    gate_inputs = {
        "automated_checks": checks,
        "real_blender_smoke": {
            "status": "pass",
            "generation_mode": "agent",
            "artifact_status": "complete",
        },
        "golden_review": {"status": "pass", "annotators_per_sample": 2},
        "dynamic_agent_provider": {"status": "pass", "director": "pass", "blender_code": "pass"},
        "paired_gate": "pass",
    }
    contract = build_experiment_contract(**_contract_inputs())
    stale = contract.model_dump(mode="json")
    stale["acceptance_margin"] = -0.50
    stale_report = build_training_readiness(**gate_inputs, experiment_contract=stale)
    assert stale_report["gates"]["experiment_contract"]["status"] == "blocked"

    contract_root = Path(str(tmpdir)) / "contract"
    write_experiment_contract_bundle(contract, contract_root)
    different = build_experiment_contract(**_contract_inputs(experiment_id="exp-20260901-b"))
    readiness = build_training_readiness(
        **gate_inputs,
        experiment_contract=different.model_dump(mode="json"),
    )
    with pytest.raises(ValueError, match="readiness.*identity"):
        require_experiment_contract(contract_root, readiness_report=readiness)
