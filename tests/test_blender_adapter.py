import json


def test_cli_backend_builds_batch_command():
    from videoact.blender_adapter import BlenderCliBackend

    backend = BlenderCliBackend(blender_bin="blender-custom")

    assert backend.build_command("scene_script.py") == [
        "blender-custom",
        "-b",
        "--python",
        "scene_script.py",
    ]


def test_mcp_backend_logs_request_and_returns_success(tmp_path):
    from videoact.blender_adapter import BlenderMcpBackend

    backend = BlenderMcpBackend(
        transport=lambda request: {"status": "success", "artifact_paths": {"blend": "scene.blend"}}
    )
    result = backend.run("scene_script.py", tmp_path)

    assert result.status == "success"
    log = tmp_path / "mcp_calls.jsonl"
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["method"] == "execute_script"
    assert record["script_path"] == "scene_script.py"


def test_adapter_falls_back_to_cli_when_mcp_fails(tmp_path):
    from videoact.blender_adapter import BlenderAdapter, BlenderMcpBackend
    from videoact.contracts import ExecutionResult

    class FakeCli:
        def run(self, script_path, run_dir, timeout_s=300):
            return ExecutionResult(status="success", backend="fake", artifact_paths={"proxy": "proxy.mp4"})

    class FakeMcp:
        def run(self, script_path, run_dir, timeout_s=300):
            return ExecutionResult(status="failed", backend="mcp", error="transport unavailable")

    adapter = BlenderAdapter(cli=FakeCli(), mcp=FakeMcp())
    result = adapter.run("scene_script.py", tmp_path, prefer="mcp")

    assert result.status == "success"
    assert result.fallback_used is True
    assert result.artifact_paths["proxy"] == "proxy.mp4"


def test_cli_backend_converts_timeout_to_execution_result(tmp_path):
    from videoact.blender_adapter import BlenderCliBackend

    def timeout_runner(*args, **kwargs):
        raise TimeoutError("timed out")

    backend = BlenderCliBackend(runner=timeout_runner)
    result = backend.run("scene_script.py", tmp_path, timeout_s=1)

    assert result.status == "timeout"
    assert result.backend == "cli"
