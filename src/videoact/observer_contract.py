"""Trust boundary and schema checks for Blender's fixed observer stage."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any


OBSERVER_SCHEMA_VERSION = "trusted-observer-v1"
OBSERVER_MANIFEST_NAME = "telemetry_manifest.json"
OBSERVER_TELEMETRY_NAME = "telemetry.json"
OBSERVER_REQUEST_NAME = "observer_request.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def create_observer_request(
    *,
    candidate_blend_hash: str,
    observer_source_hash: str,
    observer_version: str = OBSERVER_SCHEMA_VERSION,
    mesh_entity_ids: list[str] | None = None,
    obligation_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create a one-use host request consumed by the trusted Blender process."""

    payload = {
        "request_schema_version": OBSERVER_SCHEMA_VERSION,
        "observer_version": observer_version,
        "nonce": secrets.token_hex(24),
        "candidate_blend_hash": str(candidate_blend_hash),
        "observer_source_hash": str(observer_source_hash),
    }
    if mesh_entity_ids:
        payload["mesh_entity_ids"] = list(dict.fromkeys(str(item) for item in mesh_entity_ids if str(item).strip()))
    if obligation_ids:
        payload["obligation_ids"] = list(dict.fromkeys(str(item) for item in obligation_ids if str(item).strip()))
    return payload


def write_observer_request(
    path: str | Path,
    *,
    candidate_blend_hash: str,
    observer_source_hash: str,
    observer_version: str = OBSERVER_SCHEMA_VERSION,
    mesh_entity_ids: list[str] | None = None,
    obligation_ids: list[str] | None = None,
) -> dict[str, Any]:
    payload = create_observer_request(
        candidate_blend_hash=candidate_blend_hash,
        observer_source_hash=observer_source_hash,
        observer_version=observer_version,
        mesh_entity_ids=mesh_entity_ids,
        obligation_ids=obligation_ids,
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _failure(*items: str) -> dict[str, Any]:
    return {"status": "fail", "trusted": False, "failures": list(dict.fromkeys(items)), "telemetry": None}


def _forbidden_semantic_fields(value: Any, forbidden: set[str]) -> set[str]:
    """Find semantic self-attestation keys at every nesting level."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().casefold()
            if normalized in forbidden:
                found.add(str(key))
            found.update(_forbidden_semantic_fields(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_semantic_fields(child, forbidden))
    return found


def read_trusted_observer_output(
    run_dir: str | Path,
    *,
    observer_source_path: str | Path,
    require_request: bool = True,
) -> dict[str, Any]:
    """Validate observer output against the current blend and fixed source.

    The function deliberately returns no telemetry payload on any failure.
    Callers therefore cannot accidentally score a generated job's self-written
    telemetry after a trust-boundary violation.
    """

    root = Path(run_dir)
    telemetry_path = root / OBSERVER_TELEMETRY_NAME
    manifest_path = root / OBSERVER_MANIFEST_NAME
    candidate_path = root / "candidate.blend"
    if not telemetry_path.is_file() or not manifest_path.is_file():
        return _failure("missing_trusted_observer_output")
    if not candidate_path.is_file():
        return _failure("missing_candidate_blend")
    try:
        telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _failure(f"observer_json_unreadable:{type(exc).__name__}")
    if not isinstance(telemetry, dict) or not isinstance(manifest, dict):
        return _failure("observer_output_not_objects")
    failures: list[str] = []
    if manifest.get("trusted") is not True:
        failures.append("observer_not_marked_trusted")
    if manifest.get("schema_version") != OBSERVER_SCHEMA_VERSION:
        failures.append("observer_schema_version_mismatch")
    try:
        expected_blend_hash = sha256_file(candidate_path)
        expected_source_hash = sha256_file(observer_source_path)
        expected_telemetry_hash = sha256_file(telemetry_path)
    except OSError as exc:
        return _failure(f"observer_hash_input_unreadable:{type(exc).__name__}")
    if manifest.get("candidate_blend_hash") != expected_blend_hash:
        failures.append("candidate_blend_hash_mismatch")
    if manifest.get("observer_source_hash") != expected_source_hash:
        failures.append("observer_source_hash_mismatch")
    if manifest.get("telemetry_hash") != expected_telemetry_hash:
        failures.append("telemetry_hash_mismatch")
    if telemetry.get("schema_version") != OBSERVER_SCHEMA_VERSION:
        failures.append("telemetry_schema_version_mismatch")
    requested_obligation_ids: list[str] = []
    required_telemetry = {
        "frame_start",
        "frame_end",
        "fps",
        "entities",
        "observations",
        "camera_observations",
    }
    missing = sorted(required_telemetry - set(telemetry))
    failures.extend(f"telemetry_missing_field:{field}" for field in missing)
    forbidden_semantic_fields = {
        "handoff_success",
        "event_success",
        "current_owner_by_event",
        "semantic_findings",
        "trajectory_success",
    }
    failures.extend(
        f"observer_emitted_forbidden_semantic_field:{field}"
        for field in sorted(_forbidden_semantic_fields(telemetry, forbidden_semantic_fields))
    )
    request_path = root / OBSERVER_REQUEST_NAME
    if require_request:
        if not request_path.is_file():
            failures.append("missing_observer_request")
        else:
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                request = {}
                failures.append(f"observer_request_unreadable:{type(exc).__name__}")
            if isinstance(request, dict):
                requested_obligation_ids = [
                    str(item)
                    for item in request.get("obligation_ids", [])
                    if isinstance(item, str) and item.strip()
                ]
                if manifest.get("request_nonce") != request.get("nonce"):
                    failures.append("observer_request_nonce_mismatch")
                if request.get("candidate_blend_hash") != expected_blend_hash:
                    failures.append("observer_request_blend_hash_mismatch")
                if request.get("observer_source_hash") != expected_source_hash:
                    failures.append("observer_request_source_hash_mismatch")
                if list(telemetry.get("obligation_ids", [])) != list(dict.fromkeys(requested_obligation_ids)):
                    failures.append("observer_obligation_ids_mismatch")
                if list(manifest.get("obligation_ids", [])) != list(dict.fromkeys(requested_obligation_ids)):
                    failures.append("observer_manifest_obligation_ids_mismatch")
    if failures:
        return _failure(*failures)
    return {
        "status": "pass",
        "trusted": True,
        "failures": [],
        "telemetry": telemetry,
        "manifest": manifest,
        "telemetry_hash": expected_telemetry_hash,
        "candidate_blend_hash": expected_blend_hash,
    }


__all__ = [
    "OBSERVER_MANIFEST_NAME",
    "OBSERVER_REQUEST_NAME",
    "OBSERVER_SCHEMA_VERSION",
    "OBSERVER_TELEMETRY_NAME",
    "canonical_json_hash",
    "create_observer_request",
    "read_trusted_observer_output",
    "sha256_file",
    "write_observer_request",
]
