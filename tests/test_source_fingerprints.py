from __future__ import annotations


def test_source_fingerprint_detects_constant_only_template_reuse() -> None:
    from videoact.source_fingerprints import compare_source_fingerprints

    first = """
import bpy
from blender.lib.geometry import box
def build():
    actor = box((0, 0, 0), (1, 2, 1))
    actor.name = 'actor_a'
    bpy.context.scene.frame_start = 1
"""
    second = """
import bpy
from blender.lib.geometry import box
def build():
    target = box((3, 4, 0), (2, 1, 1))
    target.name = 'actor_b'
    bpy.context.scene.frame_start = 5
"""

    result = compare_source_fingerprints(first, second)

    assert result["probable_template_reuse"] is True
    assert result["normalized_ast_equal"] is True
    assert result["library_call_sequence_equal"] is True


def test_source_fingerprint_preserves_different_control_and_animation_structure() -> None:
    from videoact.source_fingerprints import compare_source_fingerprints

    static = """
import bpy
from blender.lib.geometry import box
box((0, 0, 0), (1, 1, 1))
bpy.ops.wm.save_as_mainfile(filepath='candidate.blend')
"""
    animated = """
import bpy
from blender.lib.geometry import box
for frame in range(1, 25):
    bpy.context.scene.frame_set(frame)
    box((0, frame * 0.1, 0), (1, 1, 1)).keyframe_insert(data_path='location', frame=frame)
bpy.ops.wm.save_as_mainfile(filepath='candidate.blend')
"""

    result = compare_source_fingerprints(static, animated)

    assert result["probable_template_reuse"] is False
    assert result["control_flow_equal"] is False
    assert result["animation_signature_equal"] is False


def test_dynamic_agent_audit_rejects_parameter_only_sources(tmp_path) -> None:
    from scripts.train_real_harness import audit_dynamic_agent_index

    plan_a = "a" * 64
    plan_b = "b" * 64
    source_a = tmp_path / "case-a" / "blender_job.py"
    source_b = tmp_path / "case-b" / "blender_job.py"
    source_a.parent.mkdir()
    source_b.parent.mkdir()
    template = """
import bpy
from blender.lib.geometry import box
DIRECTOR_PLAN = {"plan_hash": PLAN_HASH}
def build():
    NAME = ACTOR_NAME
    actor = box(LOCATION, (1, 2, 1))
    actor.name = NAME
    bpy.ops.wm.save_as_mainfile(filepath="candidate.blend")
"""
    source_a.write_text(
        template.replace("PLAN_HASH", repr(plan_a)).replace("ACTOR_NAME", repr("actor_a")).replace("LOCATION", repr((0, 0, 0))),
        encoding="utf-8",
    )
    source_b.write_text(
        template.replace("PLAN_HASH", repr(plan_b)).replace("ACTOR_NAME", repr("actor_b")).replace("LOCATION", repr((4, 2, 0))),
        encoding="utf-8",
    )

    jobs = []
    for case_id, plan_hash, source in (("case-a", plan_a, source_a), ("case-b", plan_b, source_b)):
        manifest_path = source.parent / "provider_manifest.json"
        stages = {
            stage: {
                "provider_kind": "codex_exec_local",
                "model_id": "codex-cli",
                "model_version": "test",
                "call_id": f"{case_id}-{stage}",
                "request_schema_hash": "1" * 64,
                "response_schema_hash": "2" * 64,
                "prompt_hash": "3" * 64,
                "request_hash": "4" * 64,
                "response_hash": "5" * 64,
                "started_at": "2026-08-29T00:00:00+00:00",
                "ended_at": "2026-08-29T00:00:01+00:00",
                "template_backed": False,
                "llm_generated": True,
            }
            for stage in ("director", "blender_code")
        }
        manifest_path.write_text(
            __import__("json").dumps(
                {
                    "manifest_version": "provider-manifest-v1",
                    "case_id": case_id,
                    "provider_mode": "model",
                    "template_backed": False,
                    "llm_generated": True,
                    "status": "complete",
                    "stages": stages,
                }
            ),
            encoding="utf-8",
        )
        import hashlib

        jobs.append(
            {
                "case_id": case_id,
                "status": "prepared",
                "codegen_call_id": f"{case_id}-blender_code",
                "director_plan_hash": plan_hash,
                "job_path": str(source),
                "code_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
                "provider_manifest_path": str(manifest_path),
            }
        )

    report = audit_dynamic_agent_index(
        {"generation_mode": "agent", "provider_mode": "model", "jobs": jobs},
        run_root=tmp_path,
        expected_case_ids=["case-a", "case-b"],
    )

    assert report["status"] == "fail"
    assert "probable_template_reuse" in report["failures"]


def test_labeled_fingerprint_audit_reports_metrics_for_at_least_100_pairs() -> None:
    from videoact.source_fingerprints import evaluate_fingerprint_pairs

    pairs = []
    for index in range(50):
        pairs.append(
            {
                "pair_id": f"reuse-{index:03d}",
                "first": "import bpy\nvalue = 1\nbpy.ops.mesh.primitive_cube_add()\n",
                "second": f"import bpy\nvalue = {index + 2}\nbpy.ops.mesh.primitive_cube_add()\n",
                "expected_template_reuse": True,
            }
        )
    for index in range(50):
        pairs.append(
            {
                "pair_id": f"different-{index:03d}",
                "first": "import bpy\nfor _ in range(2):\n    bpy.ops.mesh.primitive_cube_add()\n",
                "second": "import bpy\nif True:\n    bpy.ops.mesh.primitive_cube_add()\n",
                "expected_template_reuse": False,
            }
        )

    report = evaluate_fingerprint_pairs(pairs, minimum_pairs=100)

    assert report["status"] == "pass"
    assert report["pair_count"] == 100
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["confusion_matrix"] == {
        "true_positive": 50,
        "false_positive": 0,
        "true_negative": 50,
        "false_negative": 0,
    }
