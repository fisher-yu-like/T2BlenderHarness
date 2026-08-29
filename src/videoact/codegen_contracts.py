"""Typed boundaries for per-case Blender code generation."""

from __future__ import annotations

from typing import Any
from typing_extensions import Literal

from pydantic import Field, model_validator

from .director_contracts import ContractModel


class FunctionSignature(ContractModel):
    """Public metadata for one verified Blender library primitive."""

    name: str = Field(min_length=1)
    category: str = Field(min_length=1)
    # These two fields are optional for backwards-compatible hand-authored
    # test/context records, but production exports always populate them.  The
    # codegen prompt must use them instead of guessing a module from category.
    module: str = ""
    return_contract: str = ""
    signature: str = Field(min_length=1)
    docstring: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    cost_estimate: str = Field(min_length=1)
    example_usage: str = Field(min_length=1)
    usage_count: int = Field(default=0, ge=0)


class CodegenExample(ContractModel):
    """A reviewed plan/source pair that may be supplied as few-shot context."""

    case_id: str = Field(min_length=1)
    director_plan: dict[str, Any]
    library_calls: list[str] = Field(default_factory=list)
    generated_code: str = Field(min_length=1)
    artifact_path: str | None = None
    plan_hash: str | None = Field(default=None, min_length=64, max_length=64)
    code_hash: str | None = Field(default=None, min_length=64, max_length=64)
    review_source: str | None = Field(default=None, min_length=1)


class CodegenRequest(ContractModel):
    """Input sent to a BlenderCodeAgent at compile time."""

    director_plan: dict[str, Any]
    library_signatures: dict[str, list[FunctionSignature]] = Field(default_factory=dict)
    context_examples: list[CodegenExample] = Field(default_factory=list)
    harness_version: str = Field(min_length=1)
    constraints: list[str] = Field(default_factory=list)

    @property
    def available_library_calls(self) -> set[str]:
        return {
            signature.name
            for signatures in self.library_signatures.values()
            for signature in signatures
        }

    def validate_response(self, response: "CodegenResponse") -> "CodegenResponse":
        """Validate that a response only calls primitives visible in this request."""

        available = self.available_library_calls
        canonical_calls: list[str] = []
        unknown: list[str] = []
        for call in response.library_calls:
            name = str(call).rsplit(".", 1)[-1]
            if call in available:
                canonical_calls.append(call)
            elif str(call).startswith("blender.lib.") and name in available:
                # The provider may report the import-qualified function name;
                # keep the manifest canonical while still requiring that the
                # module path belongs to the verified library namespace.
                canonical_calls.append(name)
            else:
                unknown.append(str(call))
        if unknown:
            raise ValueError(f"unknown library calls: {unknown}")
        return response.model_copy(update={"library_calls": list(dict.fromkeys(canonical_calls))})


class CodegenResponse(ContractModel):
    """Strict result returned by the L3 agent or L4 fallback."""

    status: Literal["success", "library_insufficient", "hard_uncertainty"]
    generated_code: str = ""
    library_calls: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None
    uncertainties: list[dict[str, Any]] = Field(default_factory=list)
    llm_call_id: str = Field(default="local-unset", min_length=1)
    generation_provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_state(self) -> "CodegenResponse":
        if self.status == "success" and not self.generated_code.strip():
            raise ValueError("success response requires non-empty generated_code")
        if self.status == "library_insufficient" and not (self.fallback_reason or "").strip():
            raise ValueError("library_insufficient response requires fallback_reason")
        if self.status == "hard_uncertainty" and self.generated_code.strip():
            raise ValueError("hard_uncertainty response cannot contain generated_code")
        if len(self.library_calls) != len(set(self.library_calls)):
            raise ValueError("library_calls must be unique")
        return self
