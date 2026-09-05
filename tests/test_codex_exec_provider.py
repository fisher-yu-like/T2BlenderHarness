from __future__ import annotations

import json


def test_codex_exec_provider_returns_json_from_the_last_message(tmp_path, monkeypatch):
    from videoact.codex_exec_provider import CodexExecProvider

    calls = {}

    def fake_run(command, *, input, capture_output, text, encoding, errors, timeout, check):
        calls.update({"command": command, "input": input, "encoding": encoding, "errors": errors, "timeout": timeout})
        output_path = command[command.index("-o") + 1]
        (tmp_path / "result.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
        # The provider receives a path in its temporary directory, not this
        # test directory; copy the same JSON to that path.
        from pathlib import Path

        Path(output_path).write_text(json.dumps({"status": "success"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)

    result = CodexExecProvider(command="codex", timeout_s=17).call(
        prompt="return a JSON object",
        schema={"type": "object"},
    )

    assert result == {"status": "success"}
    assert calls["command"][0] == "codex"
    assert "exec" in calls["command"]
    assert "--output-schema" in calls["command"]
    assert calls["timeout"] == 17
    assert calls["encoding"] == "utf-8"
    assert calls["errors"] == "replace"
    assert "return a JSON object" in calls["input"]


def test_codex_exec_provider_fails_closed_on_nonzero_exit(monkeypatch):
    import pytest

    from videoact.codex_exec_provider import CodexExecProvider

    def fake_run(*_args, **_kwargs):
        return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "codex unavailable"})()

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="codex unavailable"):
        CodexExecProvider().call(prompt="return JSON", schema={"type": "object"})


def test_codex_exec_provider_normalizes_nested_optional_properties_for_strict_schema(tmp_path, monkeypatch):
    from pathlib import Path

    from videoact.codex_exec_provider import CodexExecProvider

    captured = {}

    def fake_run(command, *, input, capture_output, text, encoding, errors, timeout, check):
        schema_path = Path(command[command.index("--output-schema") + 1])
        captured["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)

    CodexExecProvider(command="codex", response_schema={"type": "object"}).call(
        prompt="return JSON",
        schema={
            "type": "object",
            "properties": {
                "required_value": {"type": "string"},
                "optional_value": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "nested": {
                    "type": "object",
                    "properties": {"optional_nested": {"type": "string"}},
                    "required": [],
                },
                "span": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"type": "integer"}],
                    "minItems": 2,
                    "maxItems": 2,
                },
                "freeform": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["required_value"],
        },
    )

    assert captured["schema"]["required"] == ["required_value", "optional_value", "nested", "span", "freeform"]
    assert captured["schema"]["properties"]["nested"]["required"] == ["optional_nested"]
    assert captured["schema"]["properties"]["span"]["items"] == {"type": "integer"}
    assert "prefixItems" not in captured["schema"]["properties"]["span"]
    assert captured["schema"]["properties"]["freeform"]["additionalProperties"] == {"type": "string"}
    assert "properties" not in captured["schema"]["properties"]["freeform"]
    assert "required" not in captured["schema"]["properties"]["freeform"]


def test_codegen_factory_emits_closed_open_map_fields_for_codex_strict_schema(tmp_path, monkeypatch):
    from pathlib import Path

    from videoact.codex_exec_provider import CodexExecProvider

    captured = {}

    def fake_run(command, *, input, capture_output, text, encoding, errors, timeout, check):
        del input, capture_output, text, encoding, errors, timeout, check
        schema_path = Path(command[command.index("--output-schema") + 1])
        captured["schema"] = json.loads(schema_path.read_text(encoding="utf-8"))
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps({"status": "hard_uncertainty"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)

    provider = CodexExecProvider.for_codegen(command="codex-test")
    provider.call(prompt="return JSON", schema=provider.response_schema)

    provenance = captured["schema"]["properties"]["generation_provenance"]
    uncertainty = captured["schema"]["properties"]["uncertainties"]["items"]
    for field in (provenance, uncertainty):
        assert field["properties"] == {}
        assert field["required"] == []
        assert field["additionalProperties"] is False


def test_codegen_factory_uses_low_reasoning_effort_by_default(tmp_path, monkeypatch):
    from pathlib import Path

    from videoact.codex_exec_provider import CodexExecProvider

    captured = {}

    def fake_run(command, *, input, capture_output, text, encoding, errors, timeout, check):
        del input, capture_output, text, encoding, errors, timeout, check
        captured["command"] = command
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps({"status": "hard_uncertainty"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)

    provider = CodexExecProvider.for_codegen(command="codex-test")
    provider.call(prompt="return JSON", schema=provider.response_schema)

    config_index = captured["command"].index("--config")
    assert captured["command"][config_index + 1] == 'model_reasoning_effort="low"'


def test_codegen_factory_prompt_states_static_runtime_bindings():
    from videoact.codex_exec_provider import CodexExecProvider

    provider = CodexExecProvider.for_codegen(command="codex-test")
    prompt = provider.prompt_builder(
        {
            "director_plan": {},
            "director_plan_hash": "a" * 64,
        }
    )

    assert "DIRECTOR_PLAN =" in prompt
    assert "OUTPUT_DIR = Path(__file__).resolve().parent" in prompt
    assert "bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / 'candidate.blend'))" in prompt
    assert "are not substitutes for those required bindings" in prompt
    assert "CameraKeyframe" in prompt
    assert ".frame, .location, and .target" in prompt
    assert "never use `.rotation`" in prompt
    assert "obj['entity_id']" in prompt
    assert "obj['entity_kind']" in prompt
    assert "obj['geometry_style']" in prompt
    assert "visible lighting" in prompt
    assert "bpy.ops.object.light_add" in prompt


def test_codex_exec_provider_attaches_explicit_image_inputs(tmp_path, monkeypatch):
    from pathlib import Path

    from videoact.codex_exec_provider import CodexExecProvider

    image_a = tmp_path / "frame_0001.png"
    image_b = tmp_path / "frame_0061.png"
    image_a.write_bytes(b"png-a")
    image_b.write_bytes(b"png-b")
    captured = {}

    def fake_run(command, *, input, capture_output, text, encoding, errors, timeout, check):
        del input, capture_output, text, encoding, errors, timeout, check
        captured["command"] = command
        output_path = Path(command[command.index("-o") + 1])
        output_path.write_text(json.dumps({"status": "success"}), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr("videoact.codex_exec_provider.subprocess.run", fake_run)

    result = CodexExecProvider(command="codex", response_schema={"type": "object"}).call(
        prompt="inspect these frames",
        schema={"type": "object"},
        image_paths=[image_a, image_b],
    )

    assert result == {"status": "success"}
    command = captured["command"]
    assert command[command.index("--image") + 1] == str(image_a.resolve())
    second_image = command.index("--image", command.index("--image") + 1)
    assert command[second_image + 1] == str(image_b.resolve())
