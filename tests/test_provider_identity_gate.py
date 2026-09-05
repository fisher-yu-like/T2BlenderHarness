from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


def test_rule_template_baseline_is_explicitly_not_model_driven() -> None:
    from scripts.train_real_harness import build_dynamic_codex_agents

    director, code_agent = build_dynamic_codex_agents(provider_mode="rule_template_baseline")

    assert director.mode == "dynamic"
    assert director.provider == "rule_template_baseline"
    assert code_agent.model == "rule_template_baseline"
    provider = code_agent.provider.__self__
    assert provider.provider_kind == "rule_template_baseline"
    assert provider.template_backed is True
    assert provider.llm_generated is False


def test_model_mode_uses_the_codex_exec_provider() -> None:
    from scripts.train_real_harness import build_dynamic_codex_agents
    from videoact.codex_exec_provider import CodexExecProvider
    from videoact.external_structured_provider import OpenAICompatibleStructuredProvider

    director, code_agent = build_dynamic_codex_agents(
        codex_command="codex-test",
        timeout_s=31,
        provider_mode="model",
    )

    assert director.mode == "dynamic"
    assert director.provider == "external-director"
    assert isinstance(director.interpreter.provider, OpenAICompatibleStructuredProvider)
    assert director.interpreter.provider.stage == "director"
    assert director.interpreter.provider.provider_kind == "external_openai_compatible"
    assert code_agent.model == "local-codex-exec"
    assert isinstance(code_agent.provider, CodexExecProvider)
    assert code_agent.provider.stage == "blender_code"
    assert code_agent.provider.provider_kind == "codex_exec_local"
    assert code_agent.provider.template_backed is False
    assert code_agent.provider.llm_generated is True
    assert code_agent.require_visible_lighting is True


def test_model_mode_allows_one_bounded_codegen_repair() -> None:
    from scripts.train_real_harness import build_dynamic_codex_agents

    _director, code_agent = build_dynamic_codex_agents(
        codex_command="codex-test",
        timeout_s=31,
        provider_mode="model",
    )

    assert code_agent.max_codegen_attempts == 2


def test_external_director_provider_records_structured_response_and_endpoint() -> None:
    from videoact.external_structured_provider import OpenAICompatibleStructuredProvider

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": '{"entities": []}'}}]}
            ).encode("utf-8")

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    provider = OpenAICompatibleStructuredProvider(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="planner-test",
        response_schema={"type": "object", "properties": {"entities": {"type": "array"}}},
        prompt_builder=lambda payload: json.dumps(payload, sort_keys=True),
        stage="director",
        opener=opener,
        timeout_s=17,
    )

    result = provider({"prompt": "A person walks."})

    assert result == {"entities": []}
    assert captured["url"] == "https://provider.example/v1/chat/completions"
    assert captured["timeout"] == 17
    assert captured["payload"]["model"] == "planner-test"
    assert captured["payload"]["response_format"]["type"] == "json_schema"
    assert provider.last_call("director")["error"] is None
    assert provider.last_call("director")["llm_generated"] is True


def test_external_director_prompt_freezes_half_open_evidence_offsets() -> None:
    from videoact.director_contracts import DirectorRequest
    from videoact.external_structured_provider import OpenAICompatibleStructuredProvider

    provider = OpenAICompatibleStructuredProvider.for_director(
        api_key="test-key",
        base_url="https://provider.example/v1",
    )
    request = DirectorRequest(
        prompt="One person adjusts the tie of another person.",
        scene_id="case-a",
        duration_s=10.0,
        fps=12,
        provider="external-director",
        policy="director-v5-external-structured",
    )
    prompt = provider.prompt_builder(request)

    assert "0-based half-open Python slice [start,end)" in prompt
    assert '"prompt_length_chars": 45' in prompt


def test_external_director_provider_reads_standard_openai_environment(monkeypatch) -> None:
    from videoact.external_structured_provider import OpenAICompatibleStructuredProvider

    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.example")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"entities\\":[]}"}}]}'

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return Response()

    provider = OpenAICompatibleStructuredProvider(
        response_schema={"type": "object", "properties": {"entities": {"type": "array"}}},
        prompt_builder=lambda payload: json.dumps(payload, sort_keys=True),
        opener=opener,
        timeout_s=19,
    )

    assert provider({"prompt": "A person walks."}) == {"entities": []}
    assert captured == {
        "url": "https://environment.example/v1/chat/completions",
        "authorization": "Bearer environment-key",
        "timeout": 19,
    }


def test_formal_provider_mode_rejects_the_baseline() -> None:
    from scripts.train_real_harness import require_model_provider_mode

    with pytest.raises(ValueError, match="--provider-mode model"):
        require_model_provider_mode("rule_template_baseline")

    assert require_model_provider_mode("model") == "model"


def test_provider_manifest_records_hashes_and_identity() -> None:
    from videoact.provider_provenance import ProviderManifest

    manifest = ProviderManifest(
        case_id="case-a",
        prompt="A red ball rolls.",
        provider_mode="model",
        harness_version="harness-v1",
    )
    manifest.record(
        stage="director",
        provider_kind="codex_exec_local",
        model_id="codex-cli",
        model_version="test-cli",
        call_id="director-call-1",
        request_schema={"type": "object"},
        response_schema={"type": "object", "properties": {"entities": {}}},
        prompt="Interpret this exact prompt",
        response={"entities": []},
        template_backed=False,
        llm_generated=True,
    )
    payload = manifest.as_dict()

    assert payload["provider_mode"] == "model"
    assert payload["template_backed"] is False
    assert payload["llm_generated"] is True
    assert payload["stages"]["director"]["call_id"] == "director-call-1"
    assert len(payload["stages"]["director"]["request_schema_hash"]) == 64
    assert len(payload["stages"]["director"]["response_hash"]) == 64
    assert payload["stages"]["director"]["prompt_hash"] == hashlib.sha256(
        "Interpret this exact prompt".encode("utf-8")
    ).hexdigest()


def test_model_agent_index_requires_provider_manifest(tmp_path: Path) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    source = tmp_path / "case-a" / "blender_job.py"
    source.parent.mkdir()
    source.write_text(
        "CASE_SCENE_PROFILE = {'profile_version': 'codex-local-case-profile-v2', 'case_signature': 'a'}\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = audit_dynamic_agent_index(
        {
            "generation_mode": "agent",
            "provider_mode": "model",
            "jobs": [
                {
                    "case_id": "case-a",
                    "status": "prepared",
                    "codegen_call_id": "call-codegen-1",
                    "director_plan_hash": "a" * 64,
                    "job_path": str(source),
                    "code_hash": digest,
                }
            ],
        },
        run_root=tmp_path,
        expected_case_ids=["case-a"],
    )

    assert report["status"] == "fail"
    assert "case-a:missing_provider_manifest" in report["failures"]


def test_model_readiness_requires_audited_manifest_and_formal_judge_config(tmp_path: Path) -> None:
    from scripts.check_training_readiness import _provider_evidence

    case_dir = tmp_path / "case-a"
    case_dir.mkdir()
    plan_hash = "a" * 64
    source = case_dir / "blender_job.py"
    source.write_text(
        "DIRECTOR_PLAN = {'plan_hash': '" + plan_hash + "'}\n"
        "CASE_SCENE_PROFILE = {'profile_version': 'codex-local-case-profile-v2', 'case_signature': '" + plan_hash[:16] + "'}\n",
        encoding="utf-8",
    )
    manifest = {
        "manifest_version": "provider-manifest-v1",
        "case_id": "case-a",
        "provider_mode": "model",
        "template_backed": False,
        "llm_generated": True,
        "status": "complete",
        "stages": {
            stage: {
                "provider_kind": "external_openai_compatible" if stage == "director" else "codex_exec_local",
                "model_id": "gpt-5.6-luna" if stage == "director" else "codex-cli",
                "model_version": "codex-exec-v1",
                "call_id": f"{stage}-call-1",
                "request_schema_hash": "b" * 64,
                "response_schema_hash": "c" * 64,
                "prompt_hash": "d" * 64,
                "request_hash": "e" * 64,
                "response_hash": "f" * 64,
                "started_at": "2026-08-29T00:00:00+00:00",
                "ended_at": "2026-08-29T00:00:01+00:00",
                "template_backed": False,
                "llm_generated": True,
            }
            for stage in ("director", "blender_code")
        },
    }
    (case_dir / "provider_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    job_index = {
        "generation_mode": "agent",
        "provider_mode": "model",
        "jobs": [
            {
                "case_id": "case-a",
                "status": "prepared",
                "codegen_call_id": "blender_code-call-1",
                "director_plan_hash": plan_hash,
                "job_path": str(source),
                "code_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
                "provider_manifest_path": str(case_dir / "provider_manifest.json"),
            }
        ],
    }
    (tmp_path / "job_index.json").write_text(json.dumps(job_index), encoding="utf-8")

    report = _provider_evidence(tmp_path)

    assert report["status"] == "fail"
    assert report["reason"] == "formal_evaluator_config_missing"


def test_baseline_manifest_never_claims_llm_generation(tmp_path: Path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs
    from videoact.codex_self_provider import CodexLocalProvider
    from videoact.director import DirectorAgent
    from videoact.blender_code_agent import BlenderCodeAgent

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    record = {
        "case_id": "case-a",
        "prompt": "A person carries the red cup.",
        "duration_s": 4.0,
        "fps": 12,
        "required_events": [],
        "oracle_expectations": {"required_entity_ids": ["actor_a", "prop_01_cup"]},
        "proxy_scene": {"entities": []},
    }
    (dataset / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (dataset / "splits.json").write_text(json.dumps({"train": ["case-a"]}), encoding="utf-8")

    provider = CodexLocalProvider()
    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        harness_version="baseline-v1",
        director_agent=DirectorAgent.from_provider(
            provider.director,
            provider_name="rule_template_baseline",
            policy="rule-template-baseline-v1",
        ),
        code_agent=BlenderCodeAgent(provider=provider.codegen, model="rule_template_baseline"),
        provider_mode="rule_template_baseline",
    )

    assert index["jobs"][0]["status"] == "prepared"
    manifest = json.loads(
        (tmp_path / "runs" / "case-a" / "provider_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["provider_mode"] == "rule_template_baseline"
    assert manifest["template_backed"] is True
    assert manifest["llm_generated"] is False
    assert all(stage["template_backed"] is True for stage in manifest["stages"].values())
