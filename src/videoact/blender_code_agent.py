"""Fail-closed L3 Blender code generation by verified library composition."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable
from typing import Any

from .codegen_contracts import CodegenRequest, CodegenResponse


FORBIDDEN_IMPORTS = {"os", "subprocess", "sys"}
FORBIDDEN_CALLS = {"eval", "exec"}
LEGACY_TEMPLATE_REFERENCES = {"compile_real_proxy_job", "direct_prompt_code", "real_proxy_job"}
RUNTIME_MARKERS = {
    "bpy_import",
    "director_plan_binding",
    "candidate_blend_save",
    "sample_frame_index",
    "blend_save",
    "animation_render",
}


def _uncertainty(identifier: str, description: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "description": description,
        "severity": "hard",
        "resolved": False,
    }


def validate_generated_source(
    source: str,
    *,
    allowed_library_calls: set[str],
    verified_library_modules: dict[str, str] | None = None,
    require_runtime: bool = True,
) -> list[str]:
    """Return static violations for generated Python source."""

    if not source.strip():
        return ["generated_code_empty"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc.msg}:{exc.lineno}"]

    violations: list[str] = []
    has_library_import = False
    has_scaffolding_import = False
    has_bpy_import = False
    has_director_plan_binding = False
    has_candidate_blend_path = False
    has_animation_frame_path = False
    has_job_dir_binding = False
    has_telemetry_artifact = False
    has_frame_index_artifact = False
    imported_library_functions: dict[str, tuple[str, str]] = {}
    imported_library_modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORTS:
                    violations.append(f"forbidden_import:{root}")
                if alias.name == "bpy":
                    has_bpy_import = True
                if alias.name.startswith("blender.lib"):
                    has_library_import = True
                    has_scaffolding_import |= alias.name.startswith("blender.lib.scaffolding")
                    if alias.name.startswith("blender.lib."):
                        imported_library_modules[alias.asname or alias.name.rsplit(".", 1)[-1]] = alias.name
                if any(part in LEGACY_TEMPLATE_REFERENCES for part in alias.name.split(".")):
                    violations.append(f"legacy_template_reference:{alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            if root in FORBIDDEN_IMPORTS:
                violations.append(f"forbidden_import:{root}")
            if root == "bpy":
                has_bpy_import = True
            if module.startswith("blender.lib"):
                has_library_import = True
                has_scaffolding_import |= module.startswith("blender.lib.scaffolding")
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    imported_library_functions[alias.asname or alias.name] = (module, alias.name)
            if any(part in LEGACY_TEMPLATE_REFERENCES for part in module.split(".")):
                violations.append(f"legacy_template_reference:{module}")
        elif isinstance(node, ast.Name) and node.id in LEGACY_TEMPLATE_REFERENCES:
            violations.append(f"legacy_template_reference:{node.id}")
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "DIRECTOR_PLAN" for target in node.targets):
                has_director_plan_binding = True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            normalized_value = node.value.replace("\\", "/").casefold()
            if normalized_value.endswith("candidate.blend"):
                has_candidate_blend_path = True
            if normalized_value.endswith("frame_"):
                has_animation_frame_path = True
            if normalized_value.endswith("telemetry.json"):
                has_telemetry_artifact = True
            if normalized_value.endswith("index.json"):
                has_frame_index_artifact = True
            if normalized_value.startswith("/tmp/") or "/tmp/" in normalized_value:
                violations.append("runtime_forbidden_hardcoded_output_path")
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in allowed_library_calls:
                if function.id not in imported_library_functions:
                    violations.append(f"library_call_not_imported:{function.id}")
                elif verified_library_modules:
                    expected = verified_library_modules.get(function.id)
                    actual, _original = imported_library_functions[function.id]
                    if expected and actual != expected:
                        violations.append(
                            f"library_import_module_mismatch:{function.id}:{actual}!={expected}"
                        )
            elif isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
                alias = function.value.id
                primitive_name = function.attr
                if primitive_name in allowed_library_calls and alias in imported_library_modules:
                    actual = imported_library_modules[alias]
                    if verified_library_modules:
                        expected = verified_library_modules.get(primitive_name)
                        if expected and actual != expected:
                            violations.append(
                                f"library_import_module_mismatch:{primitive_name}:{actual}!={expected}"
                            )
            if isinstance(function, ast.Name) and function.id in FORBIDDEN_CALLS:
                violations.append(f"forbidden_call:{function.id}")

    if not has_library_import:
        violations.append("missing_verified_library_import")
    if not allowed_library_calls:
        violations.append("no_library_capability_available")
    if require_runtime:
        source_lower = source.lower()
        has_job_dir_binding = (
            "__file__" in source_lower
            and ".parent" in source_lower
            and ("resolve()" in source_lower or ".absolute()" in source_lower)
        ) or ("bpy.path.abspath" in source_lower and "//" in source_lower)
        if not has_bpy_import:
            violations.append("runtime_missing:bpy_import")
        if not has_scaffolding_import:
            violations.append("runtime_missing:scaffolding_import")
        if not has_director_plan_binding:
            violations.append("runtime_missing:director_plan_binding")
        if not has_candidate_blend_path:
            violations.append("runtime_missing:candidate_blend")
        if "bpy.ops.wm.save_as_mainfile" not in source_lower:
            violations.append("runtime_missing:blend_save")
        if "bpy.ops.render.render" not in source_lower or "animation" not in source_lower:
            violations.append("runtime_missing:animation_render")
        if (
            "scene.render.filepath" not in source_lower
            or "image_settings.file_format" not in source_lower
            or "frame_" not in source_lower
            or not has_animation_frame_path
        ):
            violations.append("runtime_missing:animation_frame_path")
        if not has_job_dir_binding:
            violations.append("runtime_missing:job_dir_binding")
        if not has_telemetry_artifact:
            violations.append("runtime_missing:telemetry_artifact")
        if not has_frame_index_artifact:
            violations.append("runtime_missing:frame_index_artifact")
    return list(dict.fromkeys(violations))


class BlenderCodeAgent:
    """Generate one constrained Blender source at compile time.

    ``provider`` is an injected structured-output callable.  A missing or
    failing provider is an explicit hard uncertainty; this class never invokes
    the legacy template compiler as a fallback.
    """

    def __init__(
        self,
        *,
        provider: Callable[[dict[str, Any]], Any] | None = None,
        library_signatures: dict[str, list[dict[str, Any]]] | None = None,
        model: str = "codex-local",
        max_codegen_attempts: int = 1,
    ) -> None:
        if not 1 <= int(max_codegen_attempts) <= 3:
            raise ValueError("max_codegen_attempts must be between 1 and 3")
        self.provider = provider
        self.library_signatures = library_signatures or {}
        self.model = model
        self.max_codegen_attempts = int(max_codegen_attempts)

    def build_payload(
        self,
        request: CodegenRequest,
        *,
        validation_feedback: list[str] | None = None,
        previous_generated_code: str | None = None,
    ) -> dict[str, Any]:
        """Build a deterministic provider payload from the codegen contract."""

        director_plan_hash = hashlib.sha256(
            json.dumps(
                request.director_plan,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        payload: dict[str, Any] = {
            "model": self.model,
            "director_plan": request.director_plan,
            "director_plan_hash": director_plan_hash,
            "library_signatures": {
                category: [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in signatures
                ]
                for category, signatures in request.library_signatures.items()
            },
            "context_examples": [item.model_dump(mode="json") for item in request.context_examples],
            "harness_version": request.harness_version,
            "constraints": request.constraints,
            "instructions": [
                "Implement only the supplied DirectorPlan.",
            "Compose verified blender.lib primitives; do not freehand unknown geometry.",
            "Geometry primitives return (vertices, faces) mesh data; adapt it with bpy.data.meshes.new and mesh.from_pydata before assigning Blender object attributes.",
            "Preserve every required entity, ordered event, trajectory, and camera cue; save candidate.blend for the trusted observer. Generated telemetry and sample frames are optional and untrusted.",
                "Return structured CodegenResponse and declare every library call.",
            ],
        }
        if validation_feedback:
            payload["validation_feedback"] = list(validation_feedback)
        if previous_generated_code is not None:
            payload["previous_generated_code"] = previous_generated_code
        return payload

    @staticmethod
    def _hard(
        reason: str,
        *,
        call_id: str = "unavailable",
        generation_provenance: dict[str, Any] | None = None,
    ) -> CodegenResponse:
        return CodegenResponse(
            status="hard_uncertainty",
            uncertainties=[_uncertainty("codegen_hard_uncertainty", reason)],
            llm_call_id=call_id,
            generation_provenance=generation_provenance or {},
        )

    def generate(self, request: CodegenRequest) -> CodegenResponse:
        """Generate and statically validate source, failing closed on any error."""

        if self.provider is None:
            return self._hard("no structured Blender codegen provider is configured")
        feedback: list[str] = []
        previous_source: str | None = None
        last_call_id = "unavailable"
        for attempt in range(1, self.max_codegen_attempts + 1):
            try:
                repair_source = previous_source
                if repair_source is not None and len(repair_source) > 2400:
                    repair_source = (
                        repair_source[:1200]
                        + "\n...<source excerpt omitted for repair prompt>...\n"
                        + repair_source[-1200:]
                    )
                raw = self.provider(
                    self.build_payload(
                        request,
                        validation_feedback=feedback or None,
                        previous_generated_code=repair_source,
                    )
                )
                response = raw if isinstance(raw, CodegenResponse) else CodegenResponse.model_validate(raw)
                response = request.validate_response(response)
                last_call_id = response.llm_call_id
            except Exception as exc:
                return self._hard(f"provider_or_schema_error:{type(exc).__name__}:{exc}", call_id=last_call_id)

            if response.status != "success":
                return response
            previous_source = response.generated_code
            current_feedback = validate_generated_source(
                response.generated_code,
                allowed_library_calls=request.available_library_calls,
                verified_library_modules={
                    signature.name: signature.module
                    for signatures in request.library_signatures.values()
                    for signature in signatures
                    if signature.module
                },
            )
            if not current_feedback:
                if attempt > 1:
                    response = response.model_copy(
                        update={
                            "generation_provenance": {
                                **response.generation_provenance,
                                "bounded_repair_attempt": attempt,
                                "repaired_static_violations": feedback,
                            }
                        }
                    )
                return response
            feedback = current_feedback
            if attempt == self.max_codegen_attempts:
                return self._hard(
                    "static_source_gate:" + ",".join(feedback),
                    call_id=response.llm_call_id,
                    generation_provenance={
                        "failed_codegen_attempts": attempt,
                        "invalid_source_sha256": hashlib.sha256(
                            previous_source.encode("utf-8")
                        ).hexdigest(),
                        "invalid_source_excerpt": (
                            previous_source[:2000]
                            + ("\n...<omitted>...\n" if len(previous_source) > 4000 else "")
                            + (previous_source[-2000:] if len(previous_source) > 4000 else "")
                        ),
                        "static_violations": list(feedback),
                    },
                )
        return self._hard("codegen_attempts_exhausted", call_id=last_call_id)


__all__ = ["BlenderCodeAgent", "validate_generated_source"]
