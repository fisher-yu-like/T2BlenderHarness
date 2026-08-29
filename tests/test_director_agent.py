from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_director_agent_preserves_prompt_fingerprints_and_projects_compatibility_outputs():
    from videoact.director import DirectorAgent

    prompt = "Alice carries the red cube while Bob carries the blue cube, then Alice hands the red cube to Bob."
    result = DirectorAgent().plan(prompt, scene_id="director-agent-test", duration_s=12.0, fps=24)

    assert result.director_plan.request.prompt == prompt
    assert result.director_plan.provider_fingerprint.startswith("provider:")
    assert result.director_plan.policy_fingerprint.startswith("policy:")
    assert result.scene_contract.scene_id == "director-agent-test"
    assert result.trajectory_plan.timebase.fps == 24
    assert result.camera_plan == result.trajectory_plan.camera
    assert len(result.director_plan.content_hash()) == 64
    assert result.director_plan.camera_plan is not None
    assert result.director_plan.trajectory_summary["entities"]
    assert result.director_plan.coverage_obligations


def test_director_agent_rejects_unresolved_hard_uncertainty_before_projection():
    from videoact.director import DirectorAgent
    from videoact.director_contracts import DirectorUncertainty

    class HardUncertaintyInterpreter:
        def interpret(self, request):
            from videoact.director_prompt import DeterministicPromptInterpreter

            interpretation = DeterministicPromptInterpreter().interpret(request)
            interpretation.uncertainties.append(
                DirectorUncertainty(
                    id="unc_missing_actor",
                    description="No actor can be resolved.",
                    severity="hard",
                    resolved=False,
                )
            )
            return interpretation

    with pytest.raises(ValueError, match="unresolved hard uncertainty"):
        DirectorAgent(interpreter=HardUncertaintyInterpreter()).plan(
            "Someone moves an object.",
            scene_id="hard-uncertainty",
        )


def test_inner_loop_persists_director_plan_json(tmp_path: Path):
    from videoact.blender_adapter import BlenderAdapter
    from videoact.inner_loop import run_inner_loop

    class SuccessAdapter(BlenderAdapter):
        def run(self, script_path, output_dir, *, prefer="mcp"):
            from videoact.contracts import ExecutionResult

            return ExecutionResult(status="success", backend="fake", artifact_paths={"script": str(script_path)})

    result = run_inner_loop(
        {
            "case_id": "director-inner",
            "prompt": "Alice carries the red cube while Bob carries the blue cube.",
            "duration_s": 4.0,
            "fps": 12,
        },
        {"version": "harness-test"},
        tmp_path,
        adapter=SuccessAdapter(),
        max_attempts=1,
    )

    assert result.status == "success"
    director_plan_path = tmp_path / "attempts" / "01" / "director_plan.json"
    payload = json.loads(director_plan_path.read_text(encoding="utf-8"))
    assert payload["request"]["prompt"].startswith("Alice carries")


def test_prepare_real_jobs_persists_director_plan_json(tmp_path: Path):
    from scripts.prepare_real_jobs import prepare_jobs

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    case = {
        "case_id": "multi-director-01",
        "prompt": "Alice carries the red cube while Bob carries the blue cube.",
        "duration_s": 4.0,
        "fps": 12,
        "proxy_scene": {},
    }
    (dataset_root / "manifest.jsonl").write_text(json.dumps(case) + "\n", encoding="utf-8")
    (dataset_root / "splits.json").write_text(json.dumps({"train": [case["case_id"]]}), encoding="utf-8")

    prepare_jobs("train", tmp_path / "out", dataset_root=dataset_root, generation_mode="template_baseline")

    director_plan = tmp_path / "out" / case["case_id"] / "director_plan.json"
    assert director_plan.exists()
    assert json.loads(director_plan.read_text(encoding="utf-8"))["request"]["scene_id"] == "multi-director-01"
