"""Immutable provenance contract for one prompt-to-video experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


EXPERIMENT_FINGERPRINT_VERSION = "experiment-fingerprint-v1"
REQUIRED_HASH_FIELDS = (
    "prompt_hash",
    "dataset_fingerprint",
    "director_request_hash",
    "director_response_hash",
    "codegen_request_hash",
    "codegen_response_hash",
    "source_hash",
    "blend_hash",
    "observer_source_hash",
    "telemetry_hash",
    "mp4_hash",
    "evaluator_prompt_hash",
    "evaluator_schema_hash",
    "evaluator_model_hash",
    "score_policy_hash",
    "patch_hash",
    "blender_binary_hash",
    "python_lock_hash",
    "library_hash",
    "host_hash",
    "render_settings_hash",
    "frame_sampler_hash",
)
COMPATIBILITY_FIELDS = (
    "prompt_hash",
    "dataset_fingerprint",
    "observer_source_hash",
    "evaluator_prompt_hash",
    "evaluator_schema_hash",
    "evaluator_model_hash",
    "score_policy_hash",
    "blender_binary_hash",
    "blender_version",
    "python_lock_hash",
    "library_hash",
    "host_hash",
    "render_settings_hash",
    "rollout_seed",
    "frame_sampler_hash",
)


class ExperimentFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = EXPERIMENT_FINGERPRINT_VERSION
    prompt_hash: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    director_request_hash: str = Field(min_length=1)
    director_response_hash: str = Field(min_length=1)
    codegen_request_hash: str = Field(min_length=1)
    codegen_response_hash: str = Field(min_length=1)
    source_hash: str = Field(min_length=1)
    blend_hash: str = Field(min_length=1)
    observer_source_hash: str = Field(min_length=1)
    telemetry_hash: str = Field(min_length=1)
    mp4_hash: str = Field(min_length=1)
    evaluator_prompt_hash: str = Field(min_length=1)
    evaluator_schema_hash: str = Field(min_length=1)
    evaluator_model_hash: str = Field(min_length=1)
    score_policy_hash: str = Field(min_length=1)
    patch_hash: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)
    blender_binary_hash: str = Field(min_length=1)
    blender_version: str = Field(min_length=1)
    python_lock_hash: str = Field(min_length=1)
    library_hash: str = Field(min_length=1)
    host_hash: str = Field(min_length=1)
    render_settings_hash: str = Field(min_length=1)
    rollout_seed: str = Field(min_length=1)
    frame_sampler_hash: str = Field(min_length=1)
    digest: str | None = None

    def canonical_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"digest"})

    def with_digest(self) -> "ExperimentFingerprint":
        payload = json.dumps(self.canonical_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return self.model_copy(update={"digest": hashlib.sha256(payload.encode("utf-8")).hexdigest()})

    def required_hashes_complete(self) -> bool:
        return all(bool(str(getattr(self, field, "")).strip()) for field in REQUIRED_HASH_FIELDS)


def compare_experiment_fingerprints(
    first: ExperimentFingerprint | Mapping[str, Any],
    second: ExperimentFingerprint | Mapping[str, Any],
) -> dict[str, Any]:
    """Compare protocol/environment identity, excluding candidate artifacts."""

    left = first if isinstance(first, ExperimentFingerprint) else ExperimentFingerprint.model_validate(first)
    right = second if isinstance(second, ExperimentFingerprint) else ExperimentFingerprint.model_validate(second)
    mismatches = [field for field in COMPATIBILITY_FIELDS if getattr(left, field) != getattr(right, field)]
    return {
        "version": EXPERIMENT_FINGERPRINT_VERSION,
        "compatible": not mismatches,
        "mismatches": mismatches,
        "allowed_candidate_changes": [field for field in REQUIRED_HASH_FIELDS if field not in COMPATIBILITY_FIELDS],
        "first_digest": left.with_digest().digest,
        "second_digest": right.with_digest().digest,
    }


def hash_value(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    source = Path(path)
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_from_run_dir(
    run_dir: str | Path,
    *,
    dataset_fingerprint: str,
    blender_binary: str | Path,
    observer_source_path: str | Path,
    python_lock_path: str | Path,
    library_payload: Any,
    evaluator_prompt_payload: Any,
    evaluator_schema_payload: Any,
    evaluator_model_id: str,
    score_policy_payload: Any,
    frame_sampler_version: str,
    patch_hash: str | None = None,
    host_payload: Mapping[str, Any] | None = None,
) -> ExperimentFingerprint:
    """Build the complete fingerprint only when every runtime hash exists."""

    root = Path(run_dir)
    try:
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
        provider = json.loads((root / "provider_manifest.json").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment fingerprint input manifest unreadable: {type(exc).__name__}") from exc
    if not isinstance(manifest, Mapping) or not isinstance(provider, Mapping):
        raise ValueError("experiment fingerprint manifests must be objects")

    def stage_hashes(name: str) -> tuple[str, str]:
        stage = (provider.get("stages") or {}).get(name)
        if not isinstance(stage, Mapping):
            raise ValueError(f"provider manifest missing {name} stage")
        request_hash = str(stage.get("request_hash") or "").strip()
        response_hash = str(stage.get("response_hash") or "").strip()
        if not request_hash or not response_hash:
            raise ValueError(f"provider manifest missing {name} request/response hashes")
        return request_hash, response_hash

    director_request_hash, director_response_hash = stage_hashes("director")
    codegen_request_hash, codegen_response_hash = stage_hashes("blender_code")
    required_files = {
        "source_hash": root / "blender_job.py",
        "blend_hash": root / "candidate.blend",
        "telemetry_hash": root / "telemetry.json",
        "mp4_hash": root / "proxy.mp4",
    }
    file_hashes: dict[str, str] = {}
    for field, path in required_files.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"experiment fingerprint missing {path.name}")
        file_hashes[field] = hash_file(path)
    blender_path = Path(blender_binary)
    observer_path = Path(observer_source_path)
    lock_path = Path(python_lock_path)
    for label, path in (("blender binary", blender_path), ("observer source", observer_path), ("python lock", lock_path)):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"experiment fingerprint missing {label}")
    declared_observer_hash = str(manifest.get("observer_source_hash") or "").strip()
    actual_observer_hash = hash_file(observer_path)
    if declared_observer_hash and declared_observer_hash != actual_observer_hash:
        raise ValueError("experiment fingerprint observer source hash mismatch")

    host = dict(host_payload or {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    })
    payload = {
        "version": EXPERIMENT_FINGERPRINT_VERSION,
        "prompt_hash": str(manifest.get("prompt_hash") or ""),
        "dataset_fingerprint": str(dataset_fingerprint or ""),
        "director_request_hash": director_request_hash,
        "director_response_hash": director_response_hash,
        "codegen_request_hash": codegen_request_hash,
        "codegen_response_hash": codegen_response_hash,
        **file_hashes,
        "observer_source_hash": actual_observer_hash,
        "evaluator_prompt_hash": hash_value(evaluator_prompt_payload),
        "evaluator_schema_hash": hash_value(evaluator_schema_payload),
        "evaluator_model_hash": hash_value({"model_id": str(evaluator_model_id)}),
        "score_policy_hash": hash_value(score_policy_payload),
        "patch_hash": patch_hash or hash_value({"patch": "none"}),
        "harness_version": str(manifest.get("harness_version") or ""),
        "blender_binary_hash": hash_file(blender_path),
        "blender_version": str(manifest.get("blender_version") or ""),
        "python_lock_hash": hash_file(lock_path),
        "library_hash": hash_value(library_payload),
        "host_hash": hash_value(host),
        "render_settings_hash": hash_value(manifest.get("render_settings") or {}),
        "rollout_seed": str(manifest.get("rollout_seed") if manifest.get("rollout_seed") is not None else "none"),
        "frame_sampler_hash": hash_value(str(frame_sampler_version)),
    }
    fingerprint = ExperimentFingerprint.model_validate(payload)
    if not fingerprint.required_hashes_complete():
        raise ValueError("experiment fingerprint has incomplete required hashes")
    return fingerprint.with_digest()


__all__ = [
    "COMPATIBILITY_FIELDS",
    "EXPERIMENT_FINGERPRINT_VERSION",
    "ExperimentFingerprint",
    "REQUIRED_HASH_FIELDS",
    "build_from_run_dir",
    "compare_experiment_fingerprints",
    "hash_file",
    "hash_value",
]
