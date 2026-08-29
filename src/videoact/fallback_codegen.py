"""Explicit L4 code generation boundary.

L4 is a recovery path for a declared library limitation only.  It is never a
template compiler and it is never entered for a provider or schema failure.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .blender_code_agent import BlenderCodeAgent, validate_generated_source
from .codegen_contracts import CodegenRequest, CodegenResponse


def _hard(reason: str) -> CodegenResponse:
    return CodegenResponse(
        status="hard_uncertainty",
        uncertainties=[
            {
                "id": "fallback_codegen_hard_uncertainty",
                "description": reason,
                "severity": "hard",
                "resolved": False,
            }
        ],
        llm_call_id="fallback-unavailable",
    )


class FallbackCodegen:
    """Generate explicit new primitive code after ``library_insufficient``."""

    def __init__(self, *, provider: Callable[[dict[str, Any]], Any] | None) -> None:
        self.provider = provider

    def generate(self, request: CodegenRequest, initial: CodegenResponse) -> CodegenResponse:
        if initial.status != "library_insufficient":
            return _hard("L4 fallback is not allowed unless L3 declared library_insufficient")
        if self.provider is None:
            return _hard("explicit L4 fallback provider is unavailable")
        try:
            payload = BlenderCodeAgent(provider=self.provider).build_payload(request)
            payload["mode"] = "explicit_l4_new_primitive"
            payload["fallback_reason"] = initial.fallback_reason
            raw = self.provider(payload)
            response = raw if isinstance(raw, CodegenResponse) else CodegenResponse.model_validate(raw)
            request.validate_response(response)
        except Exception as exc:
            return _hard(f"L4 provider_or_schema_error:{type(exc).__name__}:{exc}")
        if response.status != "success":
            return _hard(f"L4 did not produce executable code: {response.status}")
        violations = validate_generated_source(
            response.generated_code,
            allowed_library_calls=request.available_library_calls,
        )
        if violations:
            return _hard("L4 static_source_gate:" + ",".join(violations))
        return response


__all__ = ["FallbackCodegen"]
