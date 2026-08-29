from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = ROOT / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_prompt_adapter_has_no_director_dependency():
    source = (SCRIPT_ROOT / "prepare_direct_prompt_jobs.py").read_text(encoding="utf-8")
    compiler_source = (ROOT / "blender" / "direct_prompt_code.py").read_text(encoding="utf-8")

    for text in (source, compiler_source):
        assert "DirectorAgent" not in text
        assert "DirectorPlan" not in text
        assert "SceneContract" not in text


def test_direct_prompt_job_contains_raw_prompt_and_compilable_blender_source(tmp_path):
    module = _load_module("prepare_direct_prompt_jobs", SCRIPT_ROOT / "prepare_direct_prompt_jobs.py")
    record = {
        "case_id": "direct-ablation-01",
        "prompt": "Alice carries the red cube to Bob, then hands the red cube to Bob.",
        "duration_s": 6.0,
        "fps": 12,
    }

    result = module.prepare_direct_job(
        record,
        tmp_path / record["case_id"],
        harness_version="direct-prompt-code-v1",
        evaluator_version="three-arm-v1",
        render_settings={"engine": "BLENDER_EEVEE_NEXT", "resolution": [128, 128], "samples": 1},
    )

    run_dir = Path(result["run_dir"])
    assert (run_dir / "blender_job.py").is_file()
    assert (run_dir / "direct_code_manifest.json").is_file()
    assert (run_dir / "run_manifest.json").is_file()
    assert (run_dir / "scene_contract.json").is_file()
    assert (run_dir / "trajectory.json").is_file()
    assert (run_dir / "camera_plan.json").is_file()
    assert not (run_dir / "director_plan.json").exists()
    assert record["prompt"] in (run_dir / "direct_code_manifest.json").read_text(encoding="utf-8")
    compile((run_dir / "blender_job.py").read_text(encoding="utf-8"), "direct_blender_job.py", "exec")
