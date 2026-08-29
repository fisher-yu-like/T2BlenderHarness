from __future__ import annotations

from videoact.blender_code_agent import BlenderCodeAgent
from videoact.codegen_contracts import CodegenRequest, CodegenResponse, FunctionSignature


def _request() -> CodegenRequest:
    return CodegenRequest(
        director_plan={"entities": [{"id": "actor_a"}], "events": [{"id": "walk_01"}]},
        library_signatures={
            "geometry": [
                FunctionSignature(
                    name="box",
                    category="geometry",
                    signature="box(center, size)",
                    docstring="Create a box.",
                    tags=["geometry"],
                    cost_estimate="low",
                    example_usage="box((0, 0, 0), (1, 1, 1))",
                )
            ]
        },
        harness_version="h1",
    )


def test_fallback_is_only_entered_from_explicit_library_insufficient() -> None:
    from videoact.fallback_codegen import FallbackCodegen

    response = FallbackCodegen(provider=None).generate(
        _request(),
        CodegenResponse(status="success", generated_code="source", library_calls=[]),
    )

    assert response.status == "hard_uncertainty"
    assert "not allowed" in response.uncertainties[0]["description"]


def test_fallback_provider_failure_never_returns_template_code() -> None:
    from videoact.fallback_codegen import FallbackCodegen

    def provider(_payload):
        raise RuntimeError("new primitive provider unavailable")

    response = FallbackCodegen(provider=provider).generate(
        _request(),
        CodegenResponse(status="library_insufficient", fallback_reason="spiral mesh is absent"),
    )

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""


def test_fallback_reuses_the_same_static_source_gate() -> None:
    from videoact.fallback_codegen import FallbackCodegen

    def provider(_payload):
        return {
            "status": "success",
            "generated_code": "from blender.lib.geometry import box\n# insufficient runtime",
            "library_calls": ["box"],
        }

    response = FallbackCodegen(provider=provider).generate(
        _request(),
        CodegenResponse(status="library_insufficient", fallback_reason="new primitive required"),
    )

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""
