"""Fail-closed L3 Blender code generation by verified library composition."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from .codegen_contracts import CodegenRequest, CodegenResponse


FORBIDDEN_IMPORTS = {"os", "subprocess", "sys"}
FORBIDDEN_CALLS = {"eval", "exec"}
LEGACY_TEMPLATE_REFERENCES = {"compile_real_proxy_job", "direct_prompt_code", "real_proxy_job"}
RUNTIME_MARKERS = {
    "bpy_import",
    "director_plan_binding",
    "telemetry_output",
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
    string_constants: set[str] = set()
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
            if any(part in LEGACY_TEMPLATE_REFERENCES for part in module.split(".")):
                violations.append(f"legacy_template_reference:{module}")
        elif isinstance(node, ast.Name) and node.id in LEGACY_TEMPLATE_REFERENCES:
            violations.append(f"legacy_template_reference:{node.id}")
        elif isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "DIRECTOR_PLAN" for target in node.targets):
                has_director_plan_binding = True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_constants.add(node.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            violations.append(f"forbidden_call:{node.func.id}")

    if not has_library_import:
        violations.append("missing_verified_library_import")
    if not allowed_library_calls:
        violations.append("no_library_capability_available")
    if require_runtime:
        source_lower = source.lower()
        if not has_bpy_import:
            violations.append("runtime_missing:bpy_import")
        if not has_scaffolding_import:
            violations.append("runtime_missing:scaffolding_import")
        if not has_director_plan_binding:
            violations.append("runtime_missing:director_plan_binding")
        if "telemetry.json" not in string_constants:
            violations.append("runtime_missing:telemetry_output")
        if "index.json" not in string_constants or "sample_frames" not in source:
            violations.append("runtime_missing:sample_frame_index")
        if "bpy.ops.wm.save_as_mainfile" not in source_lower:
            violations.append("runtime_missing:blend_save")
        if "bpy.ops.render.render" not in source_lower or "animation" not in source_lower:
            violations.append("runtime_missing:animation_render")
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
    ) -> None:
        self.provider = provider
        self.library_signatures = library_signatures or {}
        self.model = model

    def build_payload(self, request: CodegenRequest) -> dict[str, Any]:
        """Build a deterministic provider payload from the codegen contract."""

        return {
            "model": self.model,
            "director_plan": request.director_plan,
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
                "Preserve every required entity, ordered event, trajectory, camera cue, and telemetry field.",
                "Return structured CodegenResponse and declare every library call.",
            ],
        }

    @staticmethod
    def _hard(reason: str, *, call_id: str = "unavailable") -> CodegenResponse:
        return CodegenResponse(
            status="hard_uncertainty",
            uncertainties=[_uncertainty("codegen_hard_uncertainty", reason)],
            llm_call_id=call_id,
        )

    def generate(self, request: CodegenRequest) -> CodegenResponse:
        """Generate and statically validate source, failing closed on any error."""

        if self.provider is None:
            return self._hard("no structured Blender codegen provider is configured")
        try:
            raw = self.provider(self.build_payload(request))
            response = raw if isinstance(raw, CodegenResponse) else CodegenResponse.model_validate(raw)
            request.validate_response(response)
        except Exception as exc:
            return self._hard(f"provider_or_schema_error:{type(exc).__name__}:{exc}")

        if response.status != "success":
            return response
        violations = validate_generated_source(
            response.generated_code,
            allowed_library_calls=request.available_library_calls,
        )
        if violations:
            return self._hard("static_source_gate:" + ",".join(violations), call_id=response.llm_call_id)
        return response


__all__ = ["BlenderCodeAgent", "validate_generated_source"]
