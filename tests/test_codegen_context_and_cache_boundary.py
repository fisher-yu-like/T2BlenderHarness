from __future__ import annotations

import json
from pathlib import Path


def test_invalid_codegen_context_is_not_loaded_into_provider_request(tmp_path: Path) -> None:
    from videoact.codegen_context import load_validated_context_examples

    root = tmp_path / "examples"
    root.mkdir()
    (root / "manifest.jsonl").write_text(
        json.dumps({"case_id": "invalid-example", "generation_mode": "template_baseline"}) + "\n",
        encoding="utf-8",
    )

    examples, status = load_validated_context_examples(root)

    assert examples == []
    assert status == "invalid"


def test_valid_context_provenance_is_present_in_codegen_payload() -> None:
    from videoact.blender_code_agent import BlenderCodeAgent
    from videoact.codegen_contracts import CodegenExample, CodegenRequest, FunctionSignature

    example = CodegenExample(
        case_id="example-01",
        director_plan={"id": "plan-example"},
        library_calls=["box"],
        generated_code="from blender.lib.geometry import box",
        plan_hash="p" * 64,
        code_hash="c" * 64,
        review_source="human_review",
    )
    request = CodegenRequest(
        director_plan={"id": "plan-current"},
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
        context_examples=[example],
        harness_version="h1",
    )

    payload = BlenderCodeAgent(model="codex-local").build_payload(request)

    assert payload["context_examples"][0]["case_id"] == "example-01"
    assert payload["context_examples"][0]["plan_hash"] == "p" * 64
    assert payload["context_examples"][0]["review_source"] == "human_review"


def test_codegen_payload_explains_geometry_mesh_data_adapter_contract() -> None:
    from videoact.blender_code_agent import BlenderCodeAgent
    from videoact.codegen_contracts import CodegenRequest

    payload = BlenderCodeAgent(model="glm-5.3-flash").build_payload(
        CodegenRequest(director_plan={"id": "plan-a"}, harness_version="h1")
    )

    instructions = " ".join(payload["instructions"])
    assert "vertices" in instructions
    assert "faces" in instructions
    assert "from_pydata" in instructions


def test_code_cache_manifest_preserves_context_provenance(tmp_path: Path) -> None:
    from videoact.code_cache import CodeCache

    cache = CodeCache(tmp_path / "cache")
    cache.store(
        "plan-1",
        "h1",
        "from blender.lib.geometry import box\n",
        llm_call_id="call-1",
        metadata={"context_status": "pass", "context_example_ids": ["example-01"]},
    )

    entry = json.loads(cache.manifest_path.read_text(encoding="utf-8").splitlines()[0])

    assert entry["context_status"] == "pass"
    assert entry["context_example_ids"] == ["example-01"]
