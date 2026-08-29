"""Provider identity and call provenance for real Harness experiments.

The training runner must distinguish a rule/template baseline from a genuine
model-driven call.  This module stores hashes and identity metadata without
persisting prompts, responses, API keys, or other potentially sensitive raw
payloads.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROVIDER_MANIFEST_VERSION = "provider-manifest-v1"


def canonical_hash(value: Any) -> str:
    """Hash JSON-stable data without depending on dictionary insertion order."""

    if isinstance(value, str):
        raw = value.encode("utf-8")
    else:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def provider_owner(provider: Any) -> Any:
    """Return the object carrying identity metadata for a callable provider."""

    return getattr(provider, "__self__", provider)


def provider_identity(provider: Any) -> dict[str, Any]:
    owner = provider_owner(provider)
    return {
        "provider_kind": getattr(owner, "provider_kind", "injected_provider"),
        "model_id": getattr(owner, "model_id", getattr(owner, "name", None)),
        "model_version": getattr(owner, "model_version", "unknown"),
        "template_backed": getattr(owner, "template_backed", None),
        "llm_generated": getattr(owner, "llm_generated", None),
    }


def make_call_record(
    *,
    stage: str,
    provider_kind: str,
    model_id: str | None,
    model_version: str | None,
    call_id: str | None,
    request_schema: Any = None,
    response_schema: Any = None,
    prompt: str | None = None,
    request: Any = None,
    response: Any = None,
    template_backed: bool | None,
    llm_generated: bool | None,
    started_at: str | None = None,
    ended_at: str | None = None,
    retry_count: int = 0,
    error: str | None = None,
) -> dict[str, Any]:
    request_payload = request if request is not None else {"prompt": prompt or "", "schema": request_schema}
    response_payload = _as_json(response) if response is not None else None
    return {
        "stage": str(stage),
        "provider_kind": str(provider_kind),
        "model_id": model_id,
        "model_version": model_version,
        "call_id": call_id,
        "request_schema_hash": canonical_hash(request_schema) if request_schema is not None else None,
        "response_schema_hash": canonical_hash(response_schema) if response_schema is not None else None,
        "prompt_hash": canonical_hash(prompt or "") if prompt is not None else None,
        "request_hash": canonical_hash(request_payload),
        "response_hash": canonical_hash(response_payload) if response_payload is not None else None,
        "started_at": started_at or now_utc(),
        "ended_at": ended_at or now_utc(),
        "retry_count": int(retry_count),
        "error": error,
        "template_backed": template_backed,
        "llm_generated": llm_generated,
    }


class ProviderManifest:
    """Appendable manifest for the Director and Blender codegen stages."""

    def __init__(self, *, case_id: str, prompt: str, provider_mode: str, harness_version: str) -> None:
        self.case_id = str(case_id)
        self.prompt_hash = canonical_hash(prompt)
        self.provider_mode = str(provider_mode)
        self.harness_version = str(harness_version)
        self.stages: dict[str, dict[str, Any]] = {}

    def record(self, **kwargs: Any) -> dict[str, Any]:
        stage = str(kwargs.get("stage") or "unknown")
        call = make_call_record(**kwargs)
        return self.add_record(call)

    def add_record(self, call: Mapping[str, Any]) -> dict[str, Any]:
        """Append an already materialized call record.

        Providers own the exact request/response boundary and therefore may
        materialize a record at call time.  The job preparer then copies that
        record into the per-case manifest without recomputing hashes from a
        potentially transformed response.
        """

        normalized = dict(call)
        stage = str(normalized.get("stage") or "unknown")
        current = self.stages.get(stage)
        if current is None:
            current = {"calls": []}
            self.stages[stage] = current
        current.setdefault("calls", []).append(normalized)
        # Keep a convenient summary for gate code while retaining retries.
        current.update(normalized)
        current["retry_count"] = max(int(current.get("retry_count") or 0), len(current["calls"]) - 1)
        return normalized

    def as_dict(self) -> dict[str, Any]:
        stages = {stage: dict(value) for stage, value in self.stages.items()}
        stage_values = list(stages.values())
        template_values = [value.get("template_backed") for value in stage_values]
        generated_values = [value.get("llm_generated") for value in stage_values]
        template_backed: bool | None
        llm_generated: bool | None
        if not stage_values or any(value is None for value in template_values):
            template_backed = None
        else:
            template_backed = any(bool(value) for value in template_values)
        if not stage_values or any(value is None for value in generated_values):
            llm_generated = None
        else:
            llm_generated = all(bool(value) for value in generated_values)
        errors = [
            f"{stage}:{value.get('error')}"
            for stage, value in stages.items()
            if value.get("error")
        ]
        return {
            "manifest_version": PROVIDER_MANIFEST_VERSION,
            "case_id": self.case_id,
            "prompt_hash": self.prompt_hash,
            "provider_mode": self.provider_mode,
            "harness_version": self.harness_version,
            "template_backed": template_backed,
            "llm_generated": llm_generated,
            "status": "error" if errors else "complete" if stages else "incomplete",
            "errors": errors,
            "stages": stages,
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


__all__ = [
    "PROVIDER_MANIFEST_VERSION",
    "ProviderManifest",
    "canonical_hash",
    "make_call_record",
    "now_utc",
    "provider_identity",
    "provider_owner",
]
