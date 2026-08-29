from __future__ import annotations

import pytest
from pydantic import ValidationError

from videoact.codegen_contracts import (
    CodegenExample,
    CodegenRequest,
    CodegenResponse,
    FunctionSignature,
)


def _signature(name: str = "box") -> FunctionSignature:
    return FunctionSignature(
        name=name,
        category="geometry",
        signature=f"{name}(center: Vec3, size: Vec3)",
        docstring="Create a verified geometry primitive.",
        tags=["geometry"],
        cost_estimate="low",
        example_usage=f"{name}((0, 0, 0), (1, 1, 1))",
    )


def _request() -> CodegenRequest:
    return CodegenRequest(
        director_plan={"id": "plan-1", "entities": ["actor_a"], "events": ["carry_01"]},
        library_signatures={"geometry": [_signature()]},
        context_examples=[
            CodegenExample(
                case_id="example-1",
                director_plan={"id": "example-plan"},
                library_calls=["box"],
                generated_code="from blender.lib.geometry import box",
            )
        ],
        harness_version="agent-codegen-v1",
        constraints=["no_penetration", "visibility_checked"],
    )


def test_codegen_request_and_success_response_round_trip() -> None:
    request = _request()
    response = CodegenResponse(
        status="success",
        generated_code="from blender.lib.geometry import box\nscene = box((0, 0, 0), (1, 1, 1))",
        library_calls=["box"],
        llm_call_id="call-1",
    )

    assert request.harness_version == "agent-codegen-v1"
    assert response.library_calls == ["box"]
    assert "CodegenRequest" in str(CodegenRequest.model_json_schema())


def test_codegen_response_rejects_unknown_library_call() -> None:
    response = CodegenResponse(
        status="success",
        generated_code="print('not a library composition')",
        library_calls=["not_in_signatures"],
    )
    with pytest.raises(ValueError, match="unknown library calls"):
        _request().validate_response(response)


def test_library_insufficient_requires_reason_and_hard_uncertainty_has_no_code() -> None:
    with pytest.raises(ValidationError, match="fallback_reason"):
        CodegenResponse(status="library_insufficient")

    response = CodegenResponse(
        status="hard_uncertainty",
        uncertainties=[{"id": "unc-1", "description": "missing geometry", "severity": "hard", "resolved": False}],
    )
    assert response.generated_code == ""


def test_contracts_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError):
        FunctionSignature(
            name="box",
            category="geometry",
            signature="box()",
            docstring="Create a box.",
            tags=[],
            cost_estimate="low",
            example_usage="box()",
            unexpected=True,
        )
