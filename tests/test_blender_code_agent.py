from __future__ import annotations

from videoact.blender_code_agent import BlenderCodeAgent
from videoact.codegen_contracts import CodegenRequest, FunctionSignature


def _request() -> CodegenRequest:
    return CodegenRequest(
        director_plan={
            "request": {"prompt": "Alice carries the red cup."},
            "entities": [{"id": "actor_a"}, {"id": "red_cup"}],
            "events": [{"id": "carry_01"}],
        },
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
        harness_version="agent-codegen-v1",
        constraints=["no_penetration", "coverage_required"],
    )


def test_agent_accepts_valid_library_composition() -> None:
    seen: dict[str, object] = {}

    def provider(payload):
        seen.update(payload)
        return {
            "status": "success",
            "generated_code": (
                "import bpy\n"
                "from blender.lib.geometry import box\n"
                "from blender.lib.scaffolding import build_runtime_contract\n"
                "DIRECTOR_PLAN = {'actor_a': 'actor_a', 'red_cup': 'red_cup', 'carry_01': 'carry_01'}\n"
                "telemetry_path = 'telemetry.json'\n"
                "sample_frames = ['index.json']\n"
                "runtime_contract = build_runtime_contract('" + "a" * 64 + "', ['actor_a'], ['carry_01'], ['carry_01'])\n"
                "mesh = box((0, 0, 0), (1, 1, 1))\n"
                "bpy.ops.wm.save_as_mainfile(filepath='proxy.blend')\n"
                "bpy.ops.render.render(animation=True)\n"
            ),
            "library_calls": ["box"],
            "llm_call_id": "call-1",
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "success"
    assert response.library_calls == ["box"]
    assert "director_plan" in seen
    assert "library_signatures" in seen


def test_provider_failure_returns_hard_uncertainty_without_code() -> None:
    def provider(_payload):
        raise RuntimeError("endpoint unavailable")

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""
    assert response.uncertainties


def test_unknown_library_call_is_fail_closed() -> None:
    def provider(_payload):
        return {
            "status": "success",
            "generated_code": "from blender.lib.geometry import box",
            "library_calls": ["not_in_signatures"],
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""


def test_forbidden_python_operations_are_fail_closed() -> None:
    def provider(_payload):
        return {
            "status": "success",
            "generated_code": "import os\nfrom blender.lib.geometry import box\nbox((0,0,0),(1,1,1))",
            "library_calls": ["box"],
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""


def test_agent_requires_real_blender_runtime_contract() -> None:
    def provider(_payload):
        return {
            "status": "success",
            "generated_code": "from blender.lib.geometry import box\nmesh = box((0,0,0),(1,1,1))",
            "library_calls": ["box"],
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert "runtime" in response.uncertainties[0]["description"]


def test_agent_rejects_legacy_template_references() -> None:
    def provider(_payload):
        return {
            "status": "success",
            "generated_code": (
                "import bpy\n"
                "from blender.lib.geometry import box\n"
                "from blender.lib.scaffolding import build_runtime_contract\n"
                "DIRECTOR_PLAN = {'actor_a': 'actor_a'}\n"
                "telemetry_path = 'telemetry.json'\n"
                "sample_frames = ['index.json']\n"
                "runtime_contract = build_runtime_contract('" + "a" * 64 + "', ['actor_a'], ['carry_01'], ['carry_01'])\n"
                "legacy = compile_real_proxy_job\n"
                "bpy.ops.wm.save_as_mainfile(filepath='proxy.blend')\n"
                "bpy.ops.render.render(animation=True)\n"
                "mesh = box((0,0,0),(1,1,1))\n"
            ),
            "library_calls": ["box"],
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert "template" in response.uncertainties[0]["description"]


def test_without_provider_never_falls_back_to_template() -> None:
    response = BlenderCodeAgent().generate(_request())

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""
