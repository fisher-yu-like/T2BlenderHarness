from __future__ import annotations

import json
import hashlib

import pytest


class _Response:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_glm_provider_uses_official_v4_endpoint_and_json_object(monkeypatch):
    from videoact.external_structured_provider import GLMStructuredProvider

    monkeypatch.delenv("GLM_BASE_URL", raising=False)
    monkeypatch.delenv("GLM_MODEL", raising=False)
    monkeypatch.setenv("GLM_API_KEY", "glm-test-secret")
    captured: dict[str, object] = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return _Response({"choices": [{"message": {"content": '{"ok": true}'}}]})

    provider = GLMStructuredProvider(
        response_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        prompt_builder=lambda payload: json.dumps(payload),
        opener=opener,
        timeout_s=23,
    )

    assert provider({"case_id": "case-a"}) == {"ok": True}
    assert provider.provider_kind == "zhipu_glm_openai_compatible"
    assert provider.model_id == "glm-5.3-flash"
    assert provider.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert captured["url"] == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    assert captured["authorization"] == "Bearer glm-test-secret"
    assert captured["timeout"] == 23
    assert captured["payload"]["model"] == "glm-5.3-flash"
    assert captured["payload"]["response_format"] == {"type": "json_object"}
    assert "glm-test-secret" not in json.dumps(provider.last_call(), sort_keys=True)


def test_glm_provider_restores_transport_token_only_in_generated_source():
    from videoact.external_structured_provider import GLMStructuredProvider

    provider = GLMStructuredProvider.for_codegen(
        api_key="glm-test-secret",
        base_url="https://provider.example/api/paas/v4",
        opener=lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "success",
                                    "generated_code": (
                                        "from __JTK__ import dumps as serialize___JTK__\n"
                                        "path = 'telemetry.__JTK__'\n"
                                    ),
                                    "library_calls": [],
                                }
                            )
                        }
                    }
                ]
            }
        ),
    )

    result = provider({"director_plan": {}, "director_plan_hash": "a" * 64})

    assert result["generated_code"] == "from json import dumps as serialize_json\npath = 'telemetry.json'\n"
    assert result["generation_provenance"]["transport_token_restored"] == "__JTK__"
    assert result["generation_provenance"]["transport_token_replacement_count"] == 3


def test_glm_provider_recovers_only_identifiable_endpoint_token_truncation():
    from videoact.external_structured_provider import GLMStructuredProvider

    provider = GLMStructuredProvider.for_codegen(
        api_key="glm-test-secret",
        base_url="https://provider.example/api/paas/v4",
        opener=lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "success",
                                    "generated_code": (
                                        "import bpy\n"
                                        "from  import dumps as serialize_\n"
                                        "value = serialize_({})\n"
                                        "path = 'telemetry.'\n"
                                        "index = 'index.'\n"
                                    ),
                                    "library_calls": [],
                                }
                            )
                        }
                    }
                ]
            }
        ),
    )

    result = provider({"director_plan": {}, "director_plan_hash": "a" * 64})

    assert "from json import dumps as serialize_json" in result["generated_code"]
    assert "serialize_json({})" in result["generated_code"]
    assert "telemetry.json" in result["generated_code"]
    assert "index.json" in result["generated_code"]
    assert "truncated_json_import:1" in result["generation_provenance"]["transport_source_normalization"]


def test_glm_codegen_factory_returns_a_structured_codegen_response():
    from videoact.external_structured_provider import GLMStructuredProvider

    provider = GLMStructuredProvider.for_codegen(
        api_key="glm-test-secret",
        base_url="https://provider.example/api/paas/v4",
        model="glm-5.3-flash",
        opener=lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "success",
                                    "generated_code": "import bpy\n",
                                    "library_calls": ["box"],
                                    "llm_call_id": "glm-code-call",
                                }
                            )
                        }
                    }
                ]
            }
        ),
    )

    result = provider({"director_plan": {"id": "plan-a"}, "director_plan_hash": "a" * 64})

    assert result["status"] == "success"
    assert result["llm_call_id"].startswith("external:blender_code:")
    assert provider.stage == "blender_code"
    assert provider.provider_kind == "zhipu_glm_openai_compatible"
    code_prompt = provider.prompt_builder({"director_plan": {}, "director_plan_hash": "a" * 64})
    assert "complete case-specific Blender source" in code_prompt
    assert "compile" in code_prompt.lower()
    assert "os" in code_prompt
    assert "subprocess" in code_prompt
    assert "sys" in code_prompt
    assert "blender.lib.scaffolding" in code_prompt
    assert "director_plan_hash" in code_prompt
    assert "exact literal" in code_prompt
    assert "a" * 64 in provider.prompt_builder({"director_plan_hash": "a" * 64, "director_plan": {}})
    assert "vertices" in code_prompt
    assert "from_pydata" in code_prompt
    assert "exact module" in code_prompt.lower()
    assert "frames/animation/frame_" in code_prompt
    assert "Path(__file__).resolve().parent" in code_prompt
    assert "telemetry.json" in code_prompt
    assert "CameraKeyframe" in code_prompt
    assert "target" in code_prompt
    assert "blank import" in code_prompt
    assert "json.dump" in code_prompt
    assert "__JTK__" in code_prompt


def test_codegen_provider_assigns_transport_call_id_when_model_omits_it():
    from videoact.external_structured_provider import GLMStructuredProvider

    provider = GLMStructuredProvider.for_codegen(
        api_key="glm-test-secret",
        base_url="https://provider.example/api/paas/v4",
        opener=lambda request, timeout: _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "status": "success",
                                    "generated_code": "import bpy\n",
                                    "library_calls": [],
                                }
                            )
                        }
                    }
                ]
            }
        ),
    )

    result = provider({"director_plan": {}, "director_plan_hash": "a" * 64})

    assert result["llm_call_id"].startswith("external:blender_code:")


def test_blender_codegen_payload_contains_stable_plan_hash_for_source_binding():
    from videoact.blender_code_agent import BlenderCodeAgent
    from videoact.codegen_contracts import CodegenRequest

    request = CodegenRequest(
        director_plan={"id": "plan-a", "events": ["observe"]},
        harness_version="h1",
    )
    payload = BlenderCodeAgent(model="glm-5.3-flash").build_payload(request)

    expected = hashlib.sha256(
        json.dumps(
            request.director_plan,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert payload["director_plan_hash"] == expected


def test_codegen_contract_normalizes_verified_qualified_library_calls_only():
    from videoact.codegen_contracts import CodegenRequest, CodegenResponse, FunctionSignature

    request = CodegenRequest(
        director_plan={"id": "plan-a"},
        library_signatures={
            "geometry": [
                FunctionSignature(
                    name="box",
                    category="geometry",
                    signature="box(center, size)",
                    docstring="Create a box.",
                    cost_estimate="low",
                    example_usage="box((0, 0, 0), (1, 1, 1))",
                )
            ]
        },
        harness_version="h1",
    )
    response = CodegenResponse(
        status="success",
        generated_code="source",
        library_calls=["blender.lib.geometry.box"],
    )

    validated = request.validate_response(response)

    assert validated.library_calls == ["box"]
    with pytest.raises(ValueError, match="unknown library calls"):
        request.validate_response(
            response.model_copy(update={"library_calls": ["blender.lib.geometry.not_verified"]})
        )


def test_formal_provider_mode_accepts_glm_and_builds_two_glm_boundaries():
    from scripts.train_real_harness import build_dynamic_codex_agents, require_model_provider_mode

    assert require_model_provider_mode("glm") == "glm"
    director, code_agent = build_dynamic_codex_agents(provider_mode="glm", timeout_s=31)

    assert director.mode == "dynamic"
    assert director.provider == "external-glm"
    assert director.interpreter.provider.provider_kind == "zhipu_glm_openai_compatible"
    assert director.interpreter.provider.stage == "director"
    assert code_agent.model == "glm-5.3-flash"
    assert code_agent.provider.provider_kind == "zhipu_glm_openai_compatible"
    assert code_agent.provider.stage == "blender_code"
    assert code_agent.provider.template_backed is False
    assert code_agent.provider.llm_generated is True
    assert code_agent.max_codegen_attempts == 2


def test_glm_code_generation_uses_low_reasoning_budget_by_default_for_latency():
    from videoact.external_structured_provider import GLMStructuredProvider

    director = GLMStructuredProvider.for_director(api_key="glm-test-secret")
    codegen = GLMStructuredProvider.for_codegen(api_key="glm-test-secret")

    assert director.reasoning_effort == "high"
    assert codegen.reasoning_effort == "low"


def test_glm_agent_mode_rejects_fallback_codegen_and_reuses_no_cross_attempt_cache(tmp_path):
    from scripts.prepare_real_jobs import prepare_jobs
    from videoact.fallback_codegen import FallbackCodegen

    with pytest.raises(ValueError, match="fallback_codegen"):
        prepare_jobs(
            "train",
            tmp_path / "runs",
            dataset_root=tmp_path / "dataset",
            provider_mode="glm",
            fallback_codegen=FallbackCodegen(provider=None),
        )


def test_glm_dynamic_audit_rejects_a_non_glm_stage(tmp_path):
    from scripts.train_real_harness import audit_dynamic_agent_index

    case_dir = tmp_path / "case-a"
    case_dir.mkdir()
    plan_hash = "a" * 64
    source = case_dir / "blender_job.py"
    source.write_text(
        "DIRECTOR_PLAN = {'plan_hash': '" + plan_hash + "'}\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "manifest_version": "provider-manifest-v1",
        "case_id": "case-a",
        "provider_mode": "glm",
        "template_backed": False,
        "llm_generated": True,
        "status": "complete",
        "stages": {
            "director": {
                "provider_kind": "zhipu_glm_openai_compatible",
                "model_id": "glm-5.3-flash",
                "model_version": "glm-chat-completions-json-v1",
                "call_id": "director-call",
                "request_schema_hash": "b" * 64,
                "response_schema_hash": "c" * 64,
                "prompt_hash": "d" * 64,
                "request_hash": "e" * 64,
                "response_hash": "f" * 64,
                "started_at": "2026-08-29T00:00:00+00:00",
                "ended_at": "2026-08-29T00:00:01+00:00",
                "template_backed": False,
                "llm_generated": True,
            },
            "blender_code": {
                "provider_kind": "codex_exec_local",
                "model_id": "codex-cli",
                "model_version": "codex-exec-v1",
                "call_id": "code-call",
                "request_schema_hash": "b" * 64,
                "response_schema_hash": "c" * 64,
                "prompt_hash": "d" * 64,
                "request_hash": "e" * 64,
                "response_hash": "f" * 64,
                "started_at": "2026-08-29T00:00:00+00:00",
                "ended_at": "2026-08-29T00:00:01+00:00",
                "template_backed": False,
                "llm_generated": True,
            },
        },
    }
    manifest_path = case_dir / "provider_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_dynamic_agent_index(
        {
            "generation_mode": "agent",
            "provider_mode": "glm",
            "jobs": [
                {
                    "case_id": "case-a",
                    "status": "prepared",
                    "codegen_call_id": "code-call",
                    "director_plan_hash": plan_hash,
                    "job_path": str(source),
                    "code_hash": digest,
                    "provider_manifest_path": str(manifest_path),
                }
            ],
        },
        run_root=tmp_path,
        expected_case_ids=["case-a"],
    )

    assert report["status"] == "fail"
    assert "case-a:blender_code_provider_is_not_glm" in report["failures"]
