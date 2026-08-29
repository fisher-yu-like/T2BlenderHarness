from __future__ import annotations

import json
from pathlib import Path
import pytest


def _payload(**overrides):
    values = {
        "prompt_hash": "p" * 64,
        "dataset_fingerprint": "d" * 64,
        "director_request_hash": "r" * 64,
        "director_response_hash": "R" * 64,
        "codegen_request_hash": "c" * 64,
        "codegen_response_hash": "C" * 64,
        "source_hash": "s" * 64,
        "blend_hash": "b" * 64,
        "observer_source_hash": "o" * 64,
        "telemetry_hash": "t" * 64,
        "mp4_hash": "m" * 64,
        "evaluator_prompt_hash": "e" * 64,
        "evaluator_schema_hash": "E" * 64,
        "evaluator_model_hash": "q" * 64,
        "score_policy_hash": "S" * 64,
        "patch_hash": "x" * 64,
        "harness_version": "harness-v1",
        "blender_binary_hash": "B" * 64,
        "blender_version": "4.3.0",
        "python_lock_hash": "l" * 64,
        "library_hash": "L" * 64,
        "host_hash": "H" * 64,
        "render_settings_hash": "g" * 64,
        "rollout_seed": "none",
        "frame_sampler_hash": "f" * 64,
    }
    values.update(overrides)
    return values


def test_experiment_fingerprint_is_complete_and_changes_when_observer_changes() -> None:
    from videoact.experiment_fingerprint import ExperimentFingerprint

    first = ExperimentFingerprint.model_validate(_payload())
    second = first.model_copy(update={"observer_source_hash": "z" * 64}).with_digest()

    assert first.with_digest().digest != second.digest
    assert first.required_hashes_complete() is True


def test_fingerprint_compatibility_allows_candidate_artifact_changes_but_not_protocol_changes() -> None:
    from videoact.experiment_fingerprint import ExperimentFingerprint, compare_experiment_fingerprints

    before = ExperimentFingerprint.model_validate(_payload()).with_digest()
    after = before.model_copy(
        update={"source_hash": "z" * 64, "blend_hash": "y" * 64, "mp4_hash": "w" * 64, "patch_hash": "v" * 64}
    ).with_digest()

    compatible = compare_experiment_fingerprints(before, after)
    assert compatible["compatible"] is True
    assert compatible["mismatches"] == []

    incompatible = compare_experiment_fingerprints(
        before,
        after.model_copy(update={"dataset_fingerprint": "q" * 64}).with_digest(),
    )
    assert incompatible["compatible"] is False
    assert "dataset_fingerprint" in incompatible["mismatches"]


def test_fingerprint_compatibility_allows_regenerated_candidate_calls_and_harness_version() -> None:
    from videoact.experiment_fingerprint import ExperimentFingerprint, compare_experiment_fingerprints

    before = ExperimentFingerprint.model_validate(_payload()).with_digest()
    after = before.model_copy(
        update={
            "director_request_hash": "d" * 64,
            "director_response_hash": "e" * 64,
            "codegen_request_hash": "f" * 64,
            "codegen_response_hash": "g" * 64,
            "harness_version": "harness-v2",
        }
    ).with_digest()

    result = compare_experiment_fingerprints(before, after)

    assert result["compatible"] is True
    assert result["mismatches"] == []


def test_fingerprint_compatibility_rejects_score_policy_threshold_changes() -> None:
    from videoact.experiment_fingerprint import ExperimentFingerprint, compare_experiment_fingerprints

    before = ExperimentFingerprint.model_validate(_payload(score_policy_hash="a" * 64)).with_digest()
    after = before.model_copy(update={"score_policy_hash": "b" * 64}).with_digest()

    result = compare_experiment_fingerprints(before, after)

    assert result["compatible"] is False
    assert result["mismatches"] == ["score_policy_hash"]


def test_missing_fingerprint_hash_is_fail_closed() -> None:
    from videoact.experiment_fingerprint import ExperimentFingerprint

    payload = _payload()
    payload.pop("telemetry_hash")
    with pytest.raises(ValueError, match="telemetry_hash"):
        ExperimentFingerprint.model_validate(payload)


def test_build_from_run_dir_links_real_stage_and_artifact_hashes(tmp_path: Path) -> None:
    from videoact.experiment_fingerprint import build_from_run_dir

    root = tmp_path / "case-a"
    root.mkdir()
    (root / "blender_job.py").write_text("# generated source\n", encoding="utf-8")
    for name in ("candidate.blend", "telemetry.json", "proxy.mp4"):
        (root / name).write_bytes(name.encode("utf-8"))
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "prompt_hash": "p" * 64,
                "harness_version": "harness-v1",
                "blender_version": "4.3.0",
                "render_settings": {"resolution": [256, 256]},
                "rollout_seed": 7,
            }
        ),
        encoding="utf-8",
    )
    stage = {
        "request_hash": "r" * 64,
        "response_hash": "R" * 64,
    }
    (root / "provider_manifest.json").write_text(
        json.dumps({"stages": {"director": stage, "blender_code": stage}}), encoding="utf-8"
    )
    observer = tmp_path / "trusted_observer.py"
    observer.write_text("# fixed observer\n", encoding="utf-8")
    lock = tmp_path / "uv.lock"
    lock.write_text("lock\n", encoding="utf-8")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"blender")

    fingerprint = build_from_run_dir(
        root,
        dataset_fingerprint="d" * 64,
        blender_binary=blender,
        observer_source_path=observer,
        python_lock_path=lock,
        library_payload={"verified": ["box"]},
        evaluator_prompt_payload={"source": "prompt-hash"},
        evaluator_schema_payload={"type": "object"},
        evaluator_model_id="gpt-5.6-luna",
        score_policy_payload={"version": "scoring-v7"},
        frame_sampler_version="event-aligned-uniform-v1",
    )

    assert fingerprint.digest
    assert fingerprint.rollout_seed == "7"
    assert fingerprint.source_hash != fingerprint.blend_hash
