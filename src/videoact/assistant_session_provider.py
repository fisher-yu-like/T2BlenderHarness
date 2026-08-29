"""Assistant-session structured provider: the driving agent session IS the LLM.

This arm replaces the HTTP transport of the external structured providers with
an auditable request/response file exchange against the driving coding-agent
session (model id ``glm-5.3-flash``).  For every call the provider materializes
a request file under the session root and blocks until the driving agent writes
the matching response file.  Nothing else changes: identity, provenance hashes,
and fail-closed behavior match the external providers, ``template_backed``
stays False, and a missing or timed-out response is a hard provider error --
never a template fallback.  Pre-authored responses may be supplied per scene
before a run; they are only consumed after an exact scene/prompt match.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from .external_structured_provider import (
    OpenAICompatibleStructuredProvider,
    _normalize_strict_schema,
    codegen_response_boundary,
    director_interpretation_boundary,
)
from .provider_provenance import now_utc

PROVIDER_KIND = "agent_session_structured"
DEFAULT_MODEL_ID = "glm-5.3-flash"
DEFAULT_MODEL_VERSION = "zcode-agent-session-glm-5.3-flash-v1"
DEFAULT_WAIT_TIMEOUT_S = 7200.0
DEFAULT_POLL_INTERVAL_S = 2.0


def _payload_to_jsonable(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"repr": repr(payload)}


def _scene_id_of(payload: Any) -> str | None:
    if hasattr(payload, "scene_id"):
        return str(payload.scene_id)
    if isinstance(payload, Mapping):
        plan = payload.get("director_plan")
        if isinstance(plan, Mapping):
            request = plan.get("request")
            if isinstance(request, Mapping) and request.get("scene_id"):
                return str(request["scene_id"])
        request = payload.get("request")
        if isinstance(request, Mapping) and request.get("scene_id"):
            return str(request["scene_id"])
    return None


def _raw_prompt_of(payload: Any) -> str | None:
    if hasattr(payload, "prompt"):
        return str(payload.prompt)
    if isinstance(payload, Mapping):
        plan = payload.get("director_plan")
        if isinstance(plan, Mapping):
            request = plan.get("request")
            if isinstance(request, Mapping) and request.get("prompt"):
                return str(request["prompt"])
    return None


class AssistantSessionProvider(OpenAICompatibleStructuredProvider):
    """Structured provider whose transport is the driving agent session."""

    def __init__(
        self,
        *,
        session_root: str | Path,
        response_schema: Mapping[str, Any],
        prompt_builder: Any,
        stage: str,
        model: str = DEFAULT_MODEL_ID,
        model_version: str = DEFAULT_MODEL_VERSION,
        wait_timeout_s: float = DEFAULT_WAIT_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("provider_kind", PROVIDER_KIND)
        super().__init__(
            response_schema=response_schema,
            prompt_builder=prompt_builder,
            stage=stage,
            model=model,
            model_version=model_version,
            **kwargs,
        )
        self.session_root = Path(session_root)
        self.wait_timeout_s = float(os.getenv("ASSISTANT_WAIT_TIMEOUT_S") or wait_timeout_s)
        self.poll_interval_s = max(0.05, float(poll_interval_s))

    # -- transport ---------------------------------------------------------

    def __call__(self, payload: Any) -> dict[str, Any]:
        prompt = self.prompt_builder(payload)
        return self.exchange(payload=payload, prompt=prompt, schema=self.response_schema)

    def exchange(self, *, payload: Any, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        started_at = now_utc()
        call_id = f"assistant:{self.stage}:{uuid.uuid4().hex}"
        normalized_schema = _normalize_strict_schema(dict(schema))
        request_schema = {
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "response_schema": normalized_schema,
            },
            "required": ["prompt", "response_schema"],
            "additionalProperties": False,
        }
        requests_dir = self.session_root / "requests" / self.stage
        responses_dir = self.session_root / "responses" / self.stage
        preauth_dir = self.session_root / "preauth" / self.stage
        requests_dir.mkdir(parents=True, exist_ok=True)
        responses_dir.mkdir(parents=True, exist_ok=True)

        scene_id = _scene_id_of(payload)
        raw_prompt = _raw_prompt_of(payload)
        # Windows filenames cannot contain ':'; keep the call_id verbatim in
        # records and use a sanitized token for request/response file names.
        file_token = call_id.replace(":", "-")
        request_path = requests_dir / f"{file_token}.json"
        response_path = responses_dir / f"{file_token}.json"
        request_record = {
            "stage": self.stage,
            "call_id": call_id,
            "provider_kind": self.provider_kind,
            "model_id": self.model_id,
            "scene_id": scene_id,
            "requested_at": started_at,
            "respond_to": str(response_path.resolve()),
            "payload": _payload_to_jsonable(payload),
            "prompt": prompt,
            "response_schema": normalized_schema,
        }
        request_path.write_text(
            json.dumps(request_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        def fail(reason: str) -> RuntimeError:
            self._record_call(
                call_id=call_id,
                prompt=prompt,
                request_schema=request_schema,
                response_schema=normalized_schema,
                started_at=started_at,
                error=reason,
            )
            return RuntimeError(reason)

        result = self._consume_preauth(
            preauth_dir=preauth_dir,
            scene_id=scene_id,
            raw_prompt=raw_prompt,
            response_path=response_path,
        )
        if result is None:
            deadline = time.monotonic() + self.wait_timeout_s
            while not response_path.is_file():
                if time.monotonic() >= deadline:
                    raise fail(
                        f"assistant_response_timeout:{self.stage}:{call_id}:"
                        f"no response at {response_path}"
                    )
                time.sleep(self.poll_interval_s)
            try:
                raw = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise fail(
                    f"assistant_response_unreadable:{self.stage}:{type(exc).__name__}:{exc}"
                ) from exc
            if not isinstance(raw, dict):
                raise fail(f"assistant_response_not_object:{self.stage}:{call_id}")
            result = dict(raw)

        # The model must not be able to invent provenance: bind the typed
        # response to the transport call that produced it, exactly like the
        # external provider boundary does.  Only the CodegenResponse contract
        # carries the binding fields; the Director interpretation contract is
        # extra-forbidden, so its transport provenance lives solely in the
        # recorded call record.
        if self.stage == "blender_code":
            result["llm_call_id"] = call_id
            result["generation_provenance"] = {
                **(result.get("generation_provenance") or {}),
                "transport": "assistant_session_file_exchange",
                "session_root": str(self.session_root.resolve()),
                "request_file": str(request_path.resolve()),
                "response_file": str(response_path.resolve()),
            }
        self._record_call(
            call_id=call_id,
            prompt=prompt,
            request_schema=request_schema,
            response_schema=normalized_schema,
            response=result,
            started_at=started_at,
        )
        return result

    def _consume_preauth(
        self,
        *,
        preauth_dir: Path,
        scene_id: str | None,
        raw_prompt: str | None,
        response_path: Path,
    ) -> dict[str, Any] | None:
        """Use a pre-authored response only on an exact scene/prompt match."""

        if not scene_id or not raw_prompt:
            return None
        candidate = preauth_dir / f"{scene_id}.json"
        if not candidate.is_file():
            return None
        try:
            record = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(record, dict):
            return None
        if str(record.get("scene_id") or "") != scene_id:
            return None
        if str(record.get("prompt") or "") != raw_prompt:
            return None
        response = record.get("response")
        if not isinstance(response, dict):
            return None
        # Materialize the consumed preauth as the response file so provenance
        # shows exactly which authored artifact satisfied the request.
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(
            json.dumps(
                {"preauth_source": str(candidate.resolve()), **response},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return dict(response)

    # -- factory -----------------------------------------------------------

    @classmethod
    def for_director(cls, session_root: str | Path, **kwargs: Any) -> "AssistantSessionProvider":
        schema, builder = director_interpretation_boundary()
        return cls(
            session_root=session_root,
            response_schema=schema,
            prompt_builder=builder,
            stage="director",
            **kwargs,
        )

    @classmethod
    def for_codegen(cls, session_root: str | Path, **kwargs: Any) -> "AssistantSessionProvider":
        schema, builder = codegen_response_boundary()
        return cls(
            session_root=session_root,
            response_schema=schema,
            prompt_builder=builder,
            stage="blender_code",
            **kwargs,
        )
