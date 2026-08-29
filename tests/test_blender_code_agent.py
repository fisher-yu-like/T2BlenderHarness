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
                "from pathlib import Path\n"
                "from blender.lib.geometry import box\n"
                "from blender.lib.scaffolding import build_runtime_contract\n"
                "DIRECTOR_PLAN = {'actor_a': 'actor_a', 'red_cup': 'red_cup', 'carry_01': 'carry_01'}\n"
                "OUTPUT_DIR = Path(__file__).resolve().parent\n"
                "telemetry_path = 'telemetry.json'\n"
                "sample_frames = ['index.json']\n"
                "runtime_contract = build_runtime_contract('" + "a" * 64 + "', ['actor_a'], ['carry_01'], ['carry_01'])\n"
                "mesh = box((0, 0, 0), (1, 1, 1))\n"
                "bpy.context.scene.render.image_settings.file_format = 'PNG'\n"
                "bpy.context.scene.render.filepath = 'frames/animation/frame_'\n"
                "bpy.ops.wm.save_as_mainfile(filepath='candidate.blend')\n"
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


def test_agent_runtime_contract_requires_candidate_blend_not_generated_telemetry() -> None:
    def provider(_payload):
        return {
            "status": "success",
            "generated_code": (
                "import bpy\n"
                "from pathlib import Path\n"
                "from blender.lib.geometry import box\n"
                "from blender.lib.scaffolding import build_runtime_contract\n"
                "DIRECTOR_PLAN = {'actor_a': 'actor_a'}\n"
                "OUTPUT_DIR = Path(__file__).resolve().parent\n"
                "runtime_contract = build_runtime_contract('" + "a" * 64 + "', ['actor_a'], ['carry_01'], ['carry_01'])\n"
                "mesh = box((0, 0, 0), (1, 1, 1))\n"
                "bpy.context.scene.render.image_settings.file_format = 'PNG'\n"
                "bpy.context.scene.render.filepath = 'frames/animation/frame_'\n"
                "bpy.ops.wm.save_as_mainfile(filepath='candidate.blend')\n"
            ),
            "library_calls": ["box"],
            "llm_call_id": "candidate-only-1",
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert "animation_render" in response.uncertainties[0]["description"]


def test_provider_failure_returns_hard_uncertainty_without_code() -> None:
    def provider(_payload):
        raise RuntimeError("endpoint unavailable")

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""
    assert response.uncertainties


def test_static_codegen_failure_can_request_one_bounded_model_repair() -> None:
    calls: list[dict[str, object]] = []
    valid_source = (
        "import bpy\n"
        "from pathlib import Path\n"
        "from blender.lib.geometry import box\n"
        "from blender.lib.scaffolding import build_runtime_contract\n"
        "DIRECTOR_PLAN = {'actor_a': 'actor_a'}\n"
        "OUTPUT_DIR = Path(__file__).resolve().parent\n"
        "telemetry_path = 'telemetry.json'\n"
        "index_path = 'index.json'\n"
        "bpy.context.scene.render.image_settings.file_format = 'PNG'\n"
        "bpy.context.scene.render.filepath = str(OUTPUT_DIR / 'frames' / 'animation' / 'frame_')\n"
        "bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / 'candidate.blend'))\n"
        "bpy.ops.render.render(animation=True)\n"
    )

    def provider(payload):
        calls.append(payload)
        if len(calls) == 1:
            return {
                "status": "success",
                "generated_code": "import ",
                "library_calls": ["box"],
                "llm_call_id": "first",
            }
        return {
            "status": "success",
            "generated_code": valid_source,
            "library_calls": ["box"],
            "llm_call_id": "second",
        }

    response = BlenderCodeAgent(provider=provider, max_codegen_attempts=2).generate(_request())

    assert response.status == "success"
    assert response.generated_code == valid_source
    assert len(calls) == 2
    assert "validation_feedback" in calls[1]
    assert "syntax_error" in str(calls[1]["validation_feedback"])
    assert calls[1]["previous_generated_code"] == "import "


def test_codegen_repair_prompt_is_a_model_rewrite_not_a_template_fallback() -> None:
    from videoact.external_structured_provider import GLMStructuredProvider

    provider = GLMStructuredProvider.for_codegen(
        api_key="glm-test-secret",
        base_url="https://provider.example/api/paas/v4",
        opener=lambda request, timeout: _Response({}),
    )
    prompt = provider.prompt_builder(
        {
            "director_plan": {},
            "director_plan_hash": "a" * 64,
            "validation_feedback": ["syntax_error:invalid syntax:1"],
            "previous_generated_code": "import ",
        }
    )

    assert "THIS IS A REPAIR ATTEMPT" in prompt
    assert "Regenerate the complete source from scratch" in prompt
    assert "syntax_error:invalid syntax:1" in prompt
    assert "import " in prompt


def test_final_static_failure_preserves_bounded_diagnostic_without_returning_source() -> None:
    def provider(_payload):
        return {
            "status": "success",
            "generated_code": "import ",
            "library_calls": ["box"],
            "llm_call_id": "bad-source",
        }

    response = BlenderCodeAgent(provider=provider).generate(_request())

    assert response.status == "hard_uncertainty"
    assert response.generated_code == ""
    assert response.generation_provenance["invalid_source_sha256"]
    assert response.generation_provenance["invalid_source_excerpt"] == "import "
    assert response.generation_provenance["static_violations"] == ["syntax_error:invalid syntax:1"]


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


def test_library_call_must_be_imported_from_the_verified_module() -> None:
    from videoact.blender_code_agent import validate_generated_source

    source = (
        "from blender.lib.layout import track_to_constraint\n"
        "track_to_constraint('camera', 'target')\n"
    )

    violations = validate_generated_source(
        source,
        allowed_library_calls={"track_to_constraint"},
        verified_library_modules={"track_to_constraint": "blender.lib.constraints"},
        require_runtime=False,
    )

    assert any(item.startswith("library_import_module_mismatch:track_to_constraint") for item in violations)


def test_verified_library_call_cannot_be_used_without_an_import() -> None:
    from videoact.blender_code_agent import validate_generated_source

    violations = validate_generated_source(
        "from blender.lib.geometry import ellipsoid\nbox((0, 0, 0), (1, 1, 1))\n",
        allowed_library_calls={"box", "ellipsoid"},
        verified_library_modules={"box": "blender.lib.geometry", "ellipsoid": "blender.lib.geometry"},
        require_runtime=False,
    )

    assert "library_call_not_imported:box" in violations


def test_runtime_contract_requires_host_collectable_telemetry_and_frame_index() -> None:
    from videoact.blender_code_agent import validate_generated_source

    source = (
        "import bpy\n"
        "from pathlib import Path\n"
        "from blender.lib.geometry import box\n"
        "from blender.lib.scaffolding import build_runtime_contract\n"
        "DIRECTOR_PLAN = {}\n"
        "OUTPUT_DIR = Path(__file__).resolve().parent\n"
        "bpy.context.scene.render.image_settings.file_format = 'PNG'\n"
        "bpy.context.scene.render.filepath = str(OUTPUT_DIR / 'frames' / 'animation' / 'frame_')\n"
        "bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / 'candidate.blend'))\n"
        "bpy.ops.render.render(animation=True)\n"
    )

    violations = validate_generated_source(
        source,
        allowed_library_calls={"box", "build_runtime_contract"},
        verified_library_modules={
            "box": "blender.lib.geometry",
            "build_runtime_contract": "blender.lib.scaffolding",
        },
    )

    assert "runtime_missing:telemetry_artifact" in violations
    assert "runtime_missing:frame_index_artifact" in violations


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
                "from pathlib import Path\n"
                "from blender.lib.geometry import box\n"
                "from blender.lib.scaffolding import build_runtime_contract\n"
                "DIRECTOR_PLAN = {'actor_a': 'actor_a'}\n"
                "OUTPUT_DIR = Path(__file__).resolve().parent\n"
                "telemetry_path = 'telemetry.json'\n"
                "sample_frames = ['index.json']\n"
                "runtime_contract = build_runtime_contract('" + "a" * 64 + "', ['actor_a'], ['carry_01'], ['carry_01'])\n"
                "bpy.context.scene.render.image_settings.file_format = 'PNG'\n"
                "bpy.context.scene.render.filepath = 'frames/animation/frame_'\n"
                "legacy = compile_real_proxy_job\n"
                "bpy.ops.wm.save_as_mainfile(filepath='candidate.blend')\n"
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
