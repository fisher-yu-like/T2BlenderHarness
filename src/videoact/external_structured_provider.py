"""External OpenAI-compatible structured providers for Harness generation.

The Director and Blender-code stages are deliberately separate provider
boundaries.  This adapter sends one schema-constrained chat-completions
request to the configured external model and records both successful and
failed calls.  It never compiles Blender source and never falls back to a
template or to deterministic parsing.

``GLMStructuredProvider`` is the explicit Zhipu GLM arm.  It uses the
official ``/api/paas/v4`` root and JSON-object mode because the Zhipu API
supports ``json_object`` rather than the JSON-schema response format used by
some other OpenAI-compatible services.  The returned object is still checked
against the local Pydantic contract by the caller.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from .codex_exec_provider import _normalize_strict_schema
from .provider_provenance import make_call_record, now_utc


GLM_JSON_TOKEN = "__JTK__"


def _restore_glm_json_token(result: dict[str, Any]) -> dict[str, Any]:
    """Restore a transport-safe token only inside model-generated source.

    ``glm-5.3-flash`` can drop the consecutive ``json`` token from code
    strings returned inside a JSON-object response. The codegen prompt uses a
    transport-safe token for both the primary and fallback providers, then
    restores it before source validation. This is a lossless transport
    normalization, not a source/template fallback; all static and real-
    Blender gates still run.
    """

    source = result.get("generated_code")
    if not isinstance(source, str):
        return result
    restored = source
    repairs: list[str] = []
    token_count = restored.count(GLM_JSON_TOKEN)
    if token_count:
        restored = restored.replace(GLM_JSON_TOKEN, "json")
        repairs.append(f"placeholder:{token_count}")

    # The current GLM endpoint can also delete the literal token instead of
    # honoring the placeholder.  These substitutions are deliberately narrow:
    # they only restore an otherwise syntactically identifiable import/name or
    # the two required JSON artifact suffixes.  No scene or plan content is
    # invented here, and the caller still runs the full static gate.
    restored, count = re.subn(
        r"from\s+import\s+dumps\s+as\s+serialize_",
        "from json import dumps as serialize_json",
        restored,
        count=1,
    )
    if count:
        repairs.append(f"truncated_json_import:{count}")
    restored, count = re.subn(
        r"(?<![A-Za-z0-9_])serialize_(?![A-Za-z0-9_])",
        "serialize_json",
        restored,
    )
    if count:
        repairs.append(f"truncated_json_serializer:{count}")
    restored, count = re.subn(r"(?<![A-Za-z0-9_])\.dumps\(", "json.dumps(", restored)
    if count:
        repairs.append(f"truncated_json_dumps:{count}")
    restored, count = re.subn(r"(?<![A-Za-z0-9_])\.loads\(", "json.loads(", restored)
    if count:
        repairs.append(f"truncated_json_loads:{count}")
    restored, count = re.subn(
        r"([\"'])(telemetry|index)\.([\"'])",
        r"\1\2.json\3",
        restored,
    )
    if count:
        repairs.append(f"truncated_json_filename:{count}")

    lines = restored.splitlines(keepends=True)
    if lines and lines[0].strip() == "import":
        # This is the endpoint's characteristic deletion of the module name
        # from an ``import json`` line.  Drop only the invalid line; add a
        # complete import below only when the repaired source demonstrably
        # needs it.
        restored = "".join(lines[1:])
        repairs.append("empty_import_line_removed:1")
    needs_json_module = "json." in restored and not re.search(
        r"(?:^|\n)\s*(?:import\s+json\b|from\s+json\s+import\b)", restored
    )
    needs_json_serializer = "serialize_json" in restored and "from json import" not in restored
    prefix: list[str] = []
    if needs_json_module:
        prefix.append("import json\n")
        repairs.append("missing_json_import_restored:1")
    if needs_json_serializer:
        prefix.append("from json import dumps as serialize_json\n")
        repairs.append("missing_json_serializer_import_restored:1")
    if prefix:
        restored = "".join(prefix) + restored
    if not repairs:
        return result
    provenance = dict(result.get("generation_provenance") or {})
    provenance["transport_source_normalization"] = repairs
    if token_count:
        provenance["transport_token_restored"] = GLM_JSON_TOKEN
        provenance["transport_token_replacement_count"] = token_count
    updated = dict(result)
    updated["generated_code"] = restored
    updated["generation_provenance"] = provenance
    return updated


def _chat_completions_endpoint(base_url: str, *, append_v1: bool = True) -> str:
    normalized = str(base_url).rstrip("/")
    if normalized.casefold().endswith("/chat/completions"):
        return normalized
    if append_v1 and not normalized.casefold().endswith("/v1"):
        normalized += "/v1"
    return normalized + "/chat/completions"


def _parse_chat_json(raw: Mapping[str, Any]) -> dict[str, Any]:
    choices = raw.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        raise ValueError("structured chat response has no choices")
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, list):
        content = "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, Mapping)
        )
    if not content:
        raise ValueError("structured chat response has no message content")
    text = str(content).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("structured chat response must be a JSON object")
    return value


def _strip_director_request_echo(result: dict[str, Any]) -> dict[str, Any]:
    """Drop known request-context echoes before local contract validation.

    Some OpenAI-compatible models repeat the prompt envelope (``prompt``,
    ``scene_id``, ``obligations`` and similar fields) beside the requested
    interpretation, even when the response schema excludes it.  Those fields
    are transport echoes, not planner claims; discard only this closed list and
    leave every interpretation field untouched so Pydantic remains the
    semantic gate.
    """

    context_fields = {
        "prompt",
        "prompt_length_chars",
        "span_convention",
        "scene_id",
        "duration_s",
        "fps",
        "obligations",
        "schema",
        "provider",
        "policy",
    }
    return {
        key: value
        for key, value in result.items()
        if key not in context_fields
    }


def director_interpretation_boundary() -> tuple[dict[str, Any], Callable[[Any], str]]:
    """Shared Director interpretation schema and prompt builder."""

    from .director_prompt import PromptInterpretation

    schema = PromptInterpretation.model_json_schema()
    schema["required"] = [name for name in schema.get("required", []) if name != "request"]
    schema.get("properties", {}).pop("request", None)

    def build_prompt(request: Any) -> str:
        prompt_text = str(request.prompt)
        return (
            "Interpret the exact text prompt into the supplied Director interpretation JSON schema. "
            "Return only JSON; preserve exact evidence spans and do not invent unsupported entities, "
            "actions, or camera facts. Evidence offsets are Python-style zero-based, half-open "
            "character ranges [start, end); valid end is at most prompt_length_chars. If you cannot "
            "calculate an exact span by checking prompt[prompt_span[0]:prompt_span[1]] against "
            "quoted_text before returning; do not count punctuation or whitespace by eye. If you "
            "cannot calculate an exact span, set prompt_span and quoted_text to null instead of guessing. "
            "Do not use inclusive end offsets. Unresolved material uncertainty must be hard. "
            "For a concise benchmark prompt containing an explicit subject followed by a camera cue, "
            "the subject is a valid visible environment/prop target and the camera cue is an executable "
            "observe event; do not mark the target unresolved merely because its styling or contents are "
            "unspecified. Keep only optional appearance detail as soft uncertainty. Keep camera cues in "
            "camera_cues; their observe directive must target the visible subject entity, never use camera "
            "as a target entity, and never invent a camera entity solely to satisfy a camera cue.\n"
            + json.dumps(
                {
                    "prompt": prompt_text,
                    "prompt_length_chars": len(prompt_text),
                    "span_convention": "0-based half-open Python slice [start,end)",
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

    return schema, build_prompt


def build_codegen_prompt(payload: Any, normalized_schema: Mapping[str, Any]) -> str:
    """Build the Blender codegen prompt from one case payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("Blender codegen payload must be a mapping")
    plan_hash = str(payload.get("director_plan_hash") or "").strip()
    if len(plan_hash) != 64:
        raise ValueError("codegen payload requires a 64-character director_plan_hash")
    signature_rows: list[str] = []
    for category, entries in sorted((payload.get("library_signatures") or {}).items()):
        for entry in entries or []:
            if not isinstance(entry, Mapping):
                continue
            name = str(entry.get("name") or "").strip()
            module = str(entry.get("module") or "").strip()
            return_contract = str(entry.get("return_contract") or "").strip()
            if name and module:
                signature_rows.append(
                    f"- {name}: from {module} import {name}; return_contract={return_contract or 'verified_value'}"
                )
    verified_import_table = "\n".join(signature_rows) or "(no verified primitive was supplied)"
    repair_context = ""
    if payload.get("validation_feedback"):
        repair_context = (
            "THIS IS A REPAIR ATTEMPT. The previous model source failed the local static gate. "
            "Regenerate the complete source from scratch; do not copy malformed lines from the previous "
            "candidate. Correct every listed violation before returning. Validation feedback: "
            + json.dumps(list(payload.get("validation_feedback") or []), ensure_ascii=False)
            + "\nPrevious candidate source (reference only; never repeat its invalid syntax):\n"
            + str(payload.get("previous_generated_code") or "")
            + "\nEND PREVIOUS CANDIDATE.\n"
        )
    opening_protocol = (
        "The generated_code opening protocol is exact and must be copied as complete physical lines "
        "before any scene code: line 1 `import bpy`; line 2 `from pathlib import Path`; line 3 "
        "`from json import dumps as serialize_json`. If a line is not needed, it must still not be "
        "replaced with a bare `import`; keep the complete line.\n"
    )
    return (
        "Generate one complete case-specific Blender source file (Python) from this exact payload. "
        "NON-NEGOTIABLE FIRST-LINE RULE: every import statement must contain a module/name on the "
        "same physical line; never emit a blank import line such as 'import' or 'import '. The first "
        "non-whitespace line must be a complete import such as 'from pathlib import Path' or 'import bpy'. "
        "Use complete statements only; inside generated_code the safe protocol import is `from __JTK__ import dumps as serialize___JTK__`; "
        "`from pathlib import Path`, "
        "and `import bpy` (plus complete exact verified-library imports). __JTK__ is a transport placeholder "
        "that the Host restores to the consecutive letters j-s-o-n before validation. Never place a bare "
        "import token on its own line. "
        "Return exactly one JSON object matching CodegenResponse. The generated_code string must be "
        "executable in the isolated Blender job directory and must bind DIRECTOR_PLAN, use only the "
        "listed verified blender.lib calls, preserve every entity/event/trajectory/camera obligation, "
        "report library_calls using the exact short function names from the signature table (for "
        "example box, not blender.lib.geometry.box), although the source may import the verified "
        "module path. Treat the signature table's exact module and return_contract fields as "
        "authoritative: import each primitive from its exact module; never infer a module from its "
        "category or function name. "
        "save candidate.blend, and render the animation. Do not call compile_real_proxy_job, "
        "direct_prompt_code, real_proxy_job, or any generic fallback. Do not copy a context example; "
        "adapt the current plan into a new source. If the requirements cannot be satisfied, return "
        "status=hard_uncertainty without source. Before returning, mentally run "
        "compile(generated_code, '<blender_job.py>', 'exec') and verify that every import, attribute, "
        "function name, JSON/file call, and string delimiter is complete. Import the required runtime "
        "helpers from blender.lib.scaffolding (not only from blender.lib), and include the exact "
        "payload director_plan_hash in the source so the case binding can be audited. The source must "
        "contain a direct assignment named DIRECTOR_PLAN and a literal plan hash equal to "
        + plan_hash
        + " (exact literal). Never emit a "
        "blank import, "
        "a truncated identifier, or an import of os, subprocess, or sys; eval and exec calls are "
        "forbidden. The verified geometry functions return a (vertices, faces) tuple, not a Blender "
        "object: create a mesh with bpy.data.meshes.new, call mesh.from_pydata(vertices, [], faces), "
        "then create/link the bpy object before assigning name, materials, animation, or parenting. "
        "For every visual entity in DIRECTOR_PLAN, set the exact observer metadata on its primary mesh "
        "object before saving: obj['entity_id'] must equal the plan entity ID, obj['entity_kind'] must "
        "equal the plan entity kind, and obj['geometry_style'] should describe the generated geometry. "
        "Do not rely on the Blender object name alone for entity discovery; the trusted observer reads these "
        "custom properties from the saved candidate.blend. "
        "Typed return contracts are literal: geometry returns (vertices, faces); camera returns a list "
        "of CameraKeyframe dataclasses accessed as .frame, .location, and .target (never tuple indexing "
        "or .look_at); constraints return ConstraintSpec dataclasses; scaffolding returns mappings or "
        "failure lists. "
        "Bind the case directory with OUTPUT_DIR = Path(__file__).resolve().parent (or an equivalent "
        "bpy.path.abspath('//') expression); never hard-code /tmp, a developer home directory, or any "
        "other absolute output path. Write telemetry.json and frames/index.json under OUTPUT_DIR when "
        "the host contract requests them. The required filenames are exactly telemetry.json and "
        "frames/index.json; do not truncate either filename or an attribute such as json.dump. Prefer "
        "Path.write_text(serialize___JTK__(payload), encoding='utf-8') for these two artifacts, which "
        "avoids an unnecessary open/dump sequence. "
        "The runtime is Blender 5.1: mathutils.Vector has no length_xy() method. For a horizontal "
        "vector length, compute math.sqrt(vector.x * vector.x + vector.y * vector.y) (or use a "
        "verified equivalent) and never call Vector.length_xy(). CameraKeyframe.location and "
        "CameraKeyframe.target are tuples; never subtract them directly. Convert each to a "
        "mathutils.Vector first or subtract their numeric components explicitly. For every camera "
        "keyframe, orient the Blender camera toward the resolved target with "
        "(Vector(target) - Vector(location)).to_track_quat('-Z', 'Y').to_euler() (or an equivalent "
        "verified look-at construction); do not hand-derive Euler signs or leave the camera pointing "
        "along an arbitrary axis. Keep the resolved subject inside the camera frame and configure "
        "If the source calls Vector directly, include the complete line `from mathutils import Vector`; "
        "if it calls mathutils.Vector, include `import mathutils`; never use either name without its "
        "matching import. "
        "sensible camera.data.clip_start and camera.data.clip_end values before rendering; never "
        "write scene.render.clip_start or scene.render.clip_end because those attributes do not exist. "
        "The target Blender executable is version 5.1.2: its Eevee render-engine enum is exactly "
        "'BLENDER_EEVEE'; never emit 'BLENDER_EEVEE_NEXT', which is not a valid enum in this runtime. "
        "If using easing or trigonometry, import math explicitly and qualify calls as math.cos, "
        "math.sin, and math.pi; never emit bare cos, sin, or pi names. "
        "Blender 5.1 Action data has no direct action.fcurves collection; keyframe animated objects "
        "with object.keyframe_insert or another Blender 5.1-supported API, and never iterate "
        "action.fcurves. All geometry radius, depth, height, width, and scale values passed to "
        "verified primitives must be strictly positive; never pass a literal 0 or 0.0 for any "
        "radius, depth, height, width, or scale argument. For a mesh object's material slots use "
        "obj.data.materials; Blender Object itself has no materials collection. "
        "For the artifact contract, set scene.render.image_settings.file_format='PNG', set "
        "scene.render.filepath to the absolute job path ending in frames/animation/frame_, and call "
        "bpy.ops.render.render(animation=True) so the fixed observer can collect numbered animation "
        "frames. Do not write an MP4 from the generated job. "
        "For every executable entity and event in DIRECTOR_PLAN, materialize the matching Blender object "
        "or animation/camera binding; do not omit an environment subject merely because it is static. "
        "Return no markdown fences.\n"
        + opening_protocol
        + "Authoritative verified import table (copy the module literally; category is not a module):\n"
        + verified_import_table
        + "\n"
        + repair_context
        + "\nRuntime-only boilerplate shape (adapt paths to OUTPUT_DIR; do not copy a scene): "
        "from __JTK__ import dumps as serialize___JTK__; from pathlib import Path; "
        "OUTPUT_DIR = Path(__file__).resolve().parent; "
        "FRAME_DIR = OUTPUT_DIR / 'frames' / 'animation'; "
        "scene.render.filepath = str(FRAME_DIR / 'frame_'); "
        "write telemetry.__JTK__ and frames/index.__JTK__ with serialize___JTK__ before rendering; "
        "the Host restores __JTK__ to json before validation.\n"
        + json.dumps(
            {
                "codegen_response_schema": normalized_schema,
                "payload": dict(payload),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def codegen_response_boundary() -> tuple[dict[str, Any], Callable[[Any], str]]:
    """Shared Blender codegen schema and prompt builder."""

    from .codegen_contracts import CodegenResponse

    schema = CodegenResponse.model_json_schema()
    normalized_schema = _normalize_strict_schema(dict(schema))

    def build_prompt(payload: Any) -> str:
        return build_codegen_prompt(payload, normalized_schema)

    return schema, build_prompt


class OpenAICompatibleStructuredProvider:
    """Call an external structured-output model for one Harness stage.

    The defaults preserve the existing generic OpenAI-compatible Director
    adapter.  Provider-specific endpoints may opt out of the historical
    automatic ``/v1`` suffix, as the official Zhipu root ends in ``/v4``.
    """

    def __init__(
        self,
        *,
        response_schema: Mapping[str, Any],
        prompt_builder: Callable[[Any], str],
        stage: str = "director",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        opener: Callable[..., Any] | None = None,
        provider_kind: str = "external_openai_compatible",
        model_version: str = "chat-completions-structured-v1",
        max_tokens: int = 1600,
        response_format_type: str = "json_schema",
        append_v1: bool = True,
        reasoning_effort: str | None = None,
        do_sample: bool | None = None,
        user_agent: str = "T2BlenderHarness/structured-provider",
    ) -> None:
        self.response_schema = _normalize_strict_schema(dict(response_schema))
        self.prompt_builder = prompt_builder
        self.stage = str(stage)
        self.provider_kind = str(provider_kind)
        self.model_id = str(model or os.getenv("OPENAI_DIRECTOR_MODEL") or "gpt-5.6-luna")
        self.model_version = str(model_version)
        self.base_url = str(
            base_url or os.getenv("OPENAI_BASE_URL") or "https://ai-pixel.online/v1"
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout_s = float(timeout_s)
        self.opener = opener or urllib.request.urlopen
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if response_format_type not in {"json_schema", "json_object"}:
            raise ValueError("response_format_type must be json_schema or json_object")
        self.response_format_type = str(response_format_type)
        self.append_v1 = bool(append_v1)
        self.reasoning_effort = reasoning_effort
        self.do_sample = do_sample
        self.user_agent = str(user_agent)
        self.template_backed = False
        self.llm_generated = True
        self.call_records: list[dict[str, Any]] = []

    def __call__(self, payload: Any) -> dict[str, Any]:
        return self.call(prompt=self.prompt_builder(payload), schema=self.response_schema)

    def call(self, *, prompt: str, schema: Mapping[str, Any]) -> dict[str, Any]:
        started_at = now_utc()
        call_id = f"external:{self.stage}:{uuid.uuid4().hex}"
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
        response_format = (
            {"type": "json_object"}
            if self.response_format_type == "json_object"
            else {
                "type": "json_schema",
                "json_schema": {
                    "name": "director_interpretation",
                    "strict": True,
                    "schema": normalized_schema,
                },
            }
        )
        system_content = (
            "You are the external BlenderCodeAgent. Generate a complete, case-specific Blender Python "
            "source file from the supplied DirectorPlan and verified library signatures. Return exactly one "
            "JSON object matching the supplied CodegenResponse contract; put source only in generated_code. "
            "Never emit markdown, a generic scene template, or a fallback implementation."
            if self.stage == "blender_code"
            else "You are the external DirectorAgent planner. Interpret the exact prompt into the provided "
            "schema. Preserve exact evidence spans, event order, entity identities, trajectory obligations, "
            "and camera cues. Do not generate Blender Python. Return only JSON."
        )
        payload = {
            "model": self.model_id,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_content,
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": response_format,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.do_sample is not None:
            payload["do_sample"] = bool(self.do_sample)

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

        if not self.api_key:
            raise fail("api_key_not_configured")

        request = urllib.request.Request(
            _chat_completions_endpoint(self.base_url, append_v1=self.append_v1),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
            if not isinstance(raw, Mapping):
                raise ValueError("structured chat response must be an object")
            result = _parse_chat_json(raw)
            if self.stage == "director":
                result = _strip_director_request_echo(result)
            if self.stage == "blender_code":
                result = _restore_glm_json_token(result)
                # The model must not be able to invent provenance.  Bind the
                # typed response to the transport call that produced it.
                result["llm_call_id"] = call_id
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except OSError:
                detail = ""
            raise fail(f"HTTPError:{exc.code}:{self._redact_error_detail(detail)}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError, ValueError) as exc:
            raise fail(f"{type(exc).__name__}:{self._redact_error_detail(str(exc))}") from exc

        self._record_call(
            call_id=call_id,
            prompt=prompt,
            request_schema=request_schema,
            response_schema=normalized_schema,
            response=result,
            started_at=started_at,
        )
        return result

    def _redact_error_detail(self, detail: str) -> str:
        """Keep provider diagnostics useful without persisting credentials."""

        value = str(detail)
        if self.api_key:
            value = value.replace(self.api_key, "<redacted-api-key>")
        return value.replace("Bearer <redacted-api-key>", "Bearer <redacted-api-key>")

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
        self.call_records.append(
            make_call_record(
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
        )

    def last_call(self, stage: str | None = None) -> dict[str, Any] | None:
        if stage is None:
            return self.call_records[-1] if self.call_records else None
        for record in reversed(self.call_records):
            if record.get("stage") == stage:
                return record
        return None

    @classmethod
    def for_director(cls, **kwargs: Any) -> "OpenAICompatibleStructuredProvider":
        schema, build_prompt = director_interpretation_boundary()
        return cls(
            response_schema=schema,
            prompt_builder=build_prompt,
            stage="director",
            **kwargs,
        )

    @classmethod
    def for_codegen(cls, **kwargs: Any) -> "OpenAICompatibleStructuredProvider":
        """Build the structured provider boundary for Blender source code."""

        schema, build_prompt = codegen_response_boundary()
        return cls(
            response_schema=schema,
            prompt_builder=build_prompt,
            stage="blender_code",
            **kwargs,
        )


class FallbackStructuredProvider:
    """Try a primary structured provider and fail over to a secondary one.

    The wrapper only changes transport availability. Both providers retain
    their own schema, prompt, identity, and call provenance; no generated
    source or semantic response is fabricated by the fallback layer.
    """

    def __init__(self, *, primary: Any, fallback: Any) -> None:
        self.primary = primary
        self.fallback = fallback
        self.stage = str(getattr(primary, "stage", getattr(fallback, "stage", "unknown")))
        self.provider_kind = str(getattr(primary, "provider_kind", "structured_primary"))
        self.model_id = str(getattr(primary, "model_id", "primary"))
        self.model_version = str(getattr(primary, "model_version", "unknown"))
        self.template_backed = False
        self.llm_generated = True
        self.fallback_used = False
        self.fallback_errors: list[str] = []
        self.call_records: list[dict[str, Any]] = []
        self._seen_counts = {id(primary): 0, id(fallback): 0}

    def _sync_records(self, provider: Any) -> None:
        records = getattr(provider, "call_records", None)
        if not isinstance(records, list):
            return
        identity = id(provider)
        before = self._seen_counts.get(identity, 0)
        fresh = records[before:]
        self._seen_counts[identity] = len(records)
        for record in fresh:
            if isinstance(record, dict):
                self.call_records.append(dict(record))

    def __call__(self, payload: Any) -> dict[str, Any]:
        try:
            result = self.primary(payload)
        except Exception as primary_error:
            self._sync_records(self.primary)
            self.fallback_used = True
            self.fallback_errors.append(f"{type(primary_error).__name__}: {primary_error}")
            try:
                result = self.fallback(payload)
            except Exception as fallback_error:
                self._sync_records(self.fallback)
                raise RuntimeError(
                    "structured primary provider failed and fallback provider failed: "
                    f"primary={type(primary_error).__name__}:{primary_error}; "
                    f"fallback={type(fallback_error).__name__}:{fallback_error}"
                ) from fallback_error
            self._sync_records(self.fallback)
            return result
        self._sync_records(self.primary)
        return result

    def last_call(self, stage: str | None = None) -> dict[str, Any] | None:
        for record in reversed(self.call_records):
            if stage is None or record.get("stage") == stage:
                return record
        return None


class GLMStructuredProvider(OpenAICompatibleStructuredProvider):
    """Official Zhipu GLM-5.3-Flash provider for both generation stages."""

    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    DEFAULT_MODEL = "glm-5.3-flash"
    PROVIDER_KIND = "zhipu_glm_openai_compatible"

    def __init__(
        self,
        *,
        response_schema: Mapping[str, Any],
        prompt_builder: Callable[[Any], str],
        stage: str = "director",
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 120.0,
        opener: Callable[..., Any] | None = None,
        max_tokens: int = 20000,
        reasoning_effort: str | None = "high",
        do_sample: bool | None = False,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported GLM provider options: {unknown}")
        super().__init__(
            response_schema=response_schema,
            prompt_builder=prompt_builder,
            stage=stage,
            model=model or os.getenv("GLM_MODEL") or self.DEFAULT_MODEL,
            base_url=base_url or os.getenv("GLM_BASE_URL") or self.DEFAULT_BASE_URL,
            api_key=api_key if api_key is not None else os.getenv("GLM_API_KEY"),
            timeout_s=timeout_s,
            opener=opener,
            provider_kind=self.PROVIDER_KIND,
            model_version="glm-chat-completions-json-v1",
            max_tokens=max_tokens,
            response_format_type="json_object",
            append_v1=False,
            reasoning_effort=reasoning_effort,
            do_sample=do_sample,
            user_agent="T2BlenderHarness/glm-structured-provider",
        )

    @classmethod
    def for_director(cls, **kwargs: Any) -> "GLMStructuredProvider":
        # Complex_Plot prompts can require many prompt-grounded events and
        # evidence entries. The larger response budget and the default high
        # reasoning effort keep the semantic interpretation complete.
        kwargs.setdefault("max_tokens", 12000)
        kwargs.setdefault("reasoning_effort", "high")
        return super().for_director(**kwargs)  # type: ignore[return-value]

    @classmethod
    def for_codegen(cls, **kwargs: Any) -> "GLMStructuredProvider":
        kwargs.setdefault("max_tokens", 20000)
        kwargs.setdefault("reasoning_effort", "low")
        return super().for_codegen(**kwargs)  # type: ignore[return-value]


__all__ = [
    "FallbackStructuredProvider",
    "GLMStructuredProvider",
    "OpenAICompatibleStructuredProvider",
    "_chat_completions_endpoint",
]
