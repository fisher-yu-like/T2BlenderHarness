"""Codex-host provider bridge for structured Director and Blender outputs.

The bridge is intentionally a transport boundary, not a planner or a code
template.  Codex is asked for one schema-constrained JSON response in a
read-only subprocess.  Any process, schema, or JSON error is raised to the
caller so the surrounding Harness can fail closed.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .provider_provenance import make_call_record, now_utc


def _parse_json_message(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").removeprefix("json").strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("Codex response must be a JSON object")
    return value


def _normalize_strict_schema(value: Any) -> Any:
    """Make object schemas compatible with strict structured-output APIs.

    Pydantic correctly omits fields with defaults from ``required``.  Codex's
    structured-output endpoint uses the stricter OpenAI contract instead:
    every declared property must be required, with optionality represented by
    a nullable type.  Normalize recursively so nested Director fields such as
    ``actor_id`` and ``receiver_id`` do not make the provider request invalid.
    """

    if isinstance(value, dict):
        normalized = {key: _normalize_strict_schema(item) for key, item in value.items()}
        properties = normalized.get("properties")
        if normalized.get("type") == "object" or isinstance(properties, dict) or "additionalProperties" in normalized:
            if not isinstance(properties, dict):
                properties = {}
                normalized["properties"] = properties
            normalized["required"] = list(properties.keys())
            normalized["additionalProperties"] = False
        prefix_items = normalized.pop("prefixItems", None)
        if isinstance(prefix_items, list) and prefix_items and "items" not in normalized:
            # OpenAI-compatible structured output accepts homogeneous
            # ``items`` arrays, while Pydantic emits fixed tuples as
            # ``prefixItems``.  The active tuple is two integers, so preserve
            # its length bounds and use the common item schema.
            if all(item == prefix_items[0] for item in prefix_items):
                normalized["items"] = prefix_items[0]
            else:
                normalized["items"] = {"anyOf": prefix_items}
        if normalized.get("type") == "array" and "items" not in normalized:
            normalized["items"] = {}
        return normalized
    if isinstance(value, list):
        return [_normalize_strict_schema(item) for item in value]
    return value


class CodexExecProvider:
    """Call the local Codex CLI for one structured response."""

    def __init__(
        self,
        *,
        command: str = "codex",
        timeout_s: int = 1800,
        response_schema: Mapping[str, Any] | None = None,
        prompt_builder: Callable[[Any], str] | None = None,
        stage: str = "unknown",
        provider_kind: str = "codex_exec_local",
        model_id: str = "codex-cli",
        model_version: str = "codex-exec-v1",
        template_backed: bool = False,
        llm_generated: bool = True,
    ) -> None:
        self.command = command
        self.timeout_s = int(timeout_s)
        self.response_schema = dict(response_schema or {})
        self.prompt_builder = prompt_builder or (lambda payload: json.dumps(payload, ensure_ascii=False, sort_keys=True))
        self.stage = str(stage)
        self.provider_kind = str(provider_kind)
        self.model_id = str(model_id)
        self.model_version = str(model_version)
        self.template_backed = bool(template_backed)
        self.llm_generated = bool(llm_generated)
        self.call_records: list[dict[str, Any]] = []
        self.last_call_manifest: dict[str, Any] | None = None

    def __call__(self, payload: Any) -> dict[str, Any]:
        if not self.response_schema:
            raise RuntimeError("CodexExecProvider requires a response schema")
        return self.call(prompt=self.prompt_builder(payload), schema=self.response_schema)

    def call(self, *, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        started_at = now_utc()
        call_id = f"codex-exec:{self.stage}:{uuid.uuid4().hex}"
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
        with tempfile.TemporaryDirectory(prefix="t2blender-codex-") as directory:
            root = Path(directory)
            schema_path = root / "response-schema.json"
            output_path = root / "last-message.json"
            schema_path.write_text(json.dumps(normalized_schema, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            command = [
                self.command,
                "exec",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                "-",
            ]
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_s,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self._record_call(
                    call_id=call_id,
                    prompt=prompt,
                    request_schema=request_schema,
                    response_schema=normalized_schema,
                    started_at=started_at,
                    error=f"{type(exc).__name__}:{exc}",
                )
                raise RuntimeError(f"Codex exec unavailable: {exc}") from exc
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "non-zero exit").strip()
                self._record_call(
                    call_id=call_id,
                    prompt=prompt,
                    request_schema=request_schema,
                    response_schema=normalized_schema,
                    started_at=started_at,
                    error=f"returncode={completed.returncode}:{detail}",
                )
                raise RuntimeError(f"Codex exec failed: {detail}")
            message = output_path.read_text(encoding="utf-8") if output_path.is_file() else completed.stdout
            try:
                result = _parse_json_message(message)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                self._record_call(
                    call_id=call_id,
                    prompt=prompt,
                    request_schema=request_schema,
                    response_schema=normalized_schema,
                    started_at=started_at,
                    error=f"invalid_json:{type(exc).__name__}:{exc}",
                )
                raise RuntimeError(f"Codex exec returned invalid JSON: {exc}") from exc
            self._record_call(
                call_id=call_id,
                prompt=prompt,
                request_schema=request_schema,
                response_schema=normalized_schema,
                response=result,
                started_at=started_at,
            )
            return result

    def _record_call(
        self,
        *,
        call_id: str,
        prompt: str,
        request_schema: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        started_at: str,
        response: Any = None,
        error: str | None = None,
    ) -> None:
        record = make_call_record(
            stage=self.stage,
            provider_kind=self.provider_kind,
            model_id=self.model_id,
            model_version=self.model_version,
            call_id=call_id,
            request_schema=request_schema,
            response_schema=response_schema,
            prompt=prompt,
            response=response,
            template_backed=self.template_backed,
            llm_generated=self.llm_generated,
            started_at=started_at,
            ended_at=now_utc(),
            error=error,
        )
        self.call_records.append(record)
        self.last_call_manifest = record

    def last_call(self, stage: str | None = None) -> dict[str, Any] | None:
        """Return the latest exact provider-boundary record for a stage."""

        if stage is None:
            return self.call_records[-1] if self.call_records else None
        for record in reversed(self.call_records):
            if record.get("stage") == stage:
                return record
        return None

    @classmethod
    def for_director(cls, **kwargs: Any) -> "CodexExecProvider":
        from .director_prompt import PromptInterpretation

        schema = PromptInterpretation.model_json_schema()
        schema["required"] = [name for name in schema.get("required", []) if name != "request"]
        schema.get("properties", {}).pop("request", None)

        def build_prompt(request: Any) -> str:
            return (
                "Interpret the exact text prompt into the supplied Director interpretation JSON schema. "
                "Return only JSON; preserve exact evidence spans and do not invent unsupported entities, "
                "actions, or camera facts. Unresolved material uncertainty must be hard.\n"
                + json.dumps(
                    {
                        "prompt": request.prompt,
                        "scene_id": request.scene_id,
                        "duration_s": request.duration_s,
                        "fps": request.fps,
                        "obligations": request.obligations,
                        "schema": schema,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )

        return cls(stage="director", response_schema=schema, prompt_builder=build_prompt, **kwargs)

    @classmethod
    def for_codegen(cls, **kwargs: Any) -> "CodexExecProvider":
        from .codegen_contracts import CodegenResponse

        schema = CodegenResponse.model_json_schema()

        def build_prompt(payload: Any) -> str:
            return (
                "Generate one case-specific Blender job source from this DirectorPlan. Return only the "
                "schema-constrained JSON object. Use only listed blender.lib primitives, import the "
                "runtime scaffolding, bind every required ID to executable logic, save candidate.blend "
                "for the trusted observer, and never "
                "reference legacy template compilers.\n"
                + json.dumps(payload, ensure_ascii=False, sort_keys=True)
            )

        return cls(stage="blender_code", response_schema=schema, prompt_builder=build_prompt, **kwargs)


__all__ = ["CodexExecProvider"]
