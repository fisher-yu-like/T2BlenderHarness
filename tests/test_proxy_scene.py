from videoact.contracts import RunManifest
from videoact.scene_contract import SceneContractBuilder
from videoact.trajectory import TrajectoryPlanner


def test_proxy_script_is_bound_to_run_and_plan_metadata(tmp_path):
    from blender.proxy_scene import compile_proxy_script

    contract = SceneContractBuilder().build("Walk to a table and pick up a cup.")
    plan = TrajectoryPlanner().plan(contract)
    manifest = RunManifest(
        run_id="run-001",
        scene_id=contract.scene_id,
        prompt_hash="prompt-hash",
        harness_version="h1",
        evaluator_version="e1",
        plan_hash="plan-hash",
        backend="fake",
        frame_start=plan.timebase.frame_start,
        frame_end=plan.timebase.frame_end,
    )

    output = tmp_path / "proxy_scene.py"
    script = compile_proxy_script(plan, manifest, output)

    assert "run-001" in script
    assert "harness_version" in script
    assert "plan-hash" in script
    assert "frame_start" in script
    assert output.read_text(encoding="utf-8") == script


def test_proxy_script_does_not_require_blender_at_compile_time():
    from blender.proxy_scene import compile_proxy_script

    contract = SceneContractBuilder().build("Observe a table.")
    plan = TrajectoryPlanner().plan(contract)
    manifest = RunManifest(
        run_id="run-002",
        scene_id=contract.scene_id,
        prompt_hash="prompt-hash",
        harness_version="h1",
        evaluator_version="e1",
        plan_hash="plan-hash",
        backend="fake",
        frame_start=1,
        frame_end=240,
    )

    script = compile_proxy_script(plan, manifest)

    assert "import bpy" in script
