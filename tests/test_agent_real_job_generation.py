from __future__ import annotations

import hashlib
import json

from videoact.blender_code_agent import BlenderCodeAgent
from videoact.director import DirectorAgent


def _dataset(root) -> None:
    root.mkdir()
    record = {
        "case_id": "agent-case-01",
        "prompt": "Alice carries the red cup. Alice places the red cup.",
        "duration_s": 4.0,
        "fps": 24,
        "required_events": ["carry_actor_a_red_cup", "place_actor_a_red_cup"],
        "oracle_expectations": {
            "required_entity_ids": ["actor_a", "red_cup"],
            "required_camera_events": ["carry_actor_a_red_cup", "place_actor_a_red_cup"],
        },
        "proxy_scene": {
            "camera": {"must_show_events": ["carry_actor_a_red_cup", "place_actor_a_red_cup"]},
            "entities": [
                {"id": "actor_a", "kind": "character", "role": "participant"},
                {"id": "red_cup", "kind": "prop", "role": "target_object"},
            ],
        },
    }
    (root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "splits.json").write_text(json.dumps({"train": [record["case_id"]]}), encoding="utf-8")


def test_agent_mode_without_provider_fails_closed(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs

    dataset = tmp_path / "dataset"
    _dataset(dataset)
    index = prepare_jobs("train", tmp_path / "runs", dataset_root=dataset, harness_version="agent-codegen-v1")

    assert index["jobs"][0]["status"] == "director_failed"
    assert not (tmp_path / "runs" / "agent-case-01" / "blender_job.py").exists()
    assert (tmp_path / "runs" / "agent-case-01" / "director_failure.json").exists()


def test_agent_mode_freezes_valid_case_specific_source(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs
    from videoact.director_prompt import DeterministicPromptInterpreter

    dataset = tmp_path / "dataset"
    _dataset(dataset)

    def provider(payload):
        plan = payload["director_plan"]
        tokens = [
            *(entity["id"] for entity in plan["entities"]),
            *(event["id"] for event in plan["events"]),
        ]
        return {
            "status": "success",
            "generated_code": (
                "import bpy\n"
                "from pathlib import Path\n"
                "from blender.lib.geometry import box\n"
                "from blender.lib.scaffolding import build_runtime_contract\n"
                "DIRECTOR_PLAN = {" + ", ".join(repr(token) + ": " + repr(token) for token in tokens) + "}\n"
                "OUTPUT_DIR = Path(__file__).resolve().parent\n"
                "runtime_contract = build_runtime_contract('" + "a" * 64 + "', " + repr([token for token in tokens if token.startswith('actor_')]) + ", " + repr([token for token in tokens if token.startswith('carry_') or token.startswith('place_')]) + ", " + repr([token for token in tokens if token.startswith('carry_') or token.startswith('place_')]) + ")\n"
                "mesh = box((0, 0, 0), (1, 1, 1))\n"
                "bpy.ops.wm.save_as_mainfile(filepath='candidate.blend')\n"
            ),
            "library_calls": ["box"],
            "llm_call_id": "call-agent-case-01",
        }

    def director_provider(request):
        interpretation = DeterministicPromptInterpreter().interpret(request)
        return {
            "entities": [item.model_dump(mode="json") for item in interpretation.entities],
            "directives": [item.model_dump(mode="json") for item in interpretation.directives],
            "evidence": [item.model_dump(mode="json") for item in interpretation.evidence],
            "assumptions": [],
            "uncertainties": [
                item.model_dump(mode="json")
                for item in interpretation.uncertainties
                if item.severity == "soft"
            ],
        }

    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        harness_version="agent-codegen-v1",
        director_agent=DirectorAgent.from_provider(director_provider, provider_name="codex-local"),
        code_agent=BlenderCodeAgent(provider=provider),
    )

    case_dir = tmp_path / "runs" / "agent-case-01"
    assert index["jobs"][0]["status"] == "prepared"
    assert (case_dir / "blender_job.py").read_text(encoding="utf-8").startswith("import bpy")
    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["code_hash"] == hashlib.sha256((case_dir / "blender_job.py").read_bytes()).hexdigest()
    entries = [json.loads(line) for line in (tmp_path / "runs" / "code_cache" / "code_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert entries[0]["llm_call_id"] == "call-agent-case-01"
    assert (case_dir / "coverage_report.json").is_file()


def test_agent_mode_rejects_the_deterministic_baseline(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs

    dataset = tmp_path / "dataset"
    _dataset(dataset)
    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        harness_version="agent-codegen-v1",
        director_agent=DirectorAgent(),
        code_agent=BlenderCodeAgent(provider=lambda _payload: {}),
    )

    assert index["jobs"][0]["status"] == "director_failed"
    failure = json.loads((tmp_path / "runs" / "agent-case-01" / "director_failure.json").read_text(encoding="utf-8"))
    assert "dynamic" in failure["reason"]


def test_invalid_codegen_context_fails_before_blender_code_provider(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs
    from videoact.director_prompt import DeterministicPromptInterpreter

    dataset = tmp_path / "dataset"
    _dataset(dataset)
    context_root = tmp_path / "invalid-context"
    context_root.mkdir()
    (context_root / "manifest.jsonl").write_text(
        json.dumps({"case_id": "unreviewed", "generation_mode": "template_baseline"}) + "\n",
        encoding="utf-8",
    )

    def director_provider(request):
        interpretation = DeterministicPromptInterpreter().interpret(request)
        return {
            "entities": [item.model_dump(mode="json") for item in interpretation.entities],
            "directives": [item.model_dump(mode="json") for item in interpretation.directives],
            "evidence": [item.model_dump(mode="json") for item in interpretation.evidence],
            "assumptions": [],
            "uncertainties": [
                item.model_dump(mode="json")
                for item in interpretation.uncertainties
                if item.severity == "soft"
            ],
        }

    codegen_calls = []

    def code_provider(_payload):
        codegen_calls.append(True)
        raise AssertionError("invalid context must not reach BlenderCodeAgent provider")

    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        harness_version="agent-codegen-v1",
        director_agent=DirectorAgent.from_provider(director_provider, provider_name="codex-local"),
        code_agent=BlenderCodeAgent(provider=code_provider),
        codegen_examples_root=context_root,
    )

    assert codegen_calls == []
    assert index["jobs"][0]["status"] == "codegen_failed"
    failure = json.loads((tmp_path / "runs" / "agent-case-01" / "codegen_failure.json").read_text(encoding="utf-8"))
    assert failure["reason"] == "invalid_codegen_context"


def test_template_mode_is_explicit(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs

    dataset = tmp_path / "dataset"
    _dataset(dataset)
    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        harness_version="template-baseline-v1",
        generation_mode="template_baseline",
    )

    assert index["jobs"][0]["status"] == "prepared"
    assert index["jobs"][0]["generation_mode"] == "template_baseline"
    assert (tmp_path / "runs" / "agent-case-01" / "blender_job.py").exists()


def test_director_exception_is_recorded_without_aborting_batch(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs

    dataset = tmp_path / "dataset"
    _dataset(dataset)

    class FailingDirector:
        def plan(self, *_args, **_kwargs):
            raise RuntimeError("structured provider unavailable")

    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        director_agent=FailingDirector(),
        code_agent=BlenderCodeAgent(provider=lambda _payload: {}),
    )

    assert index["jobs"][0]["status"] == "director_failed"
    failure = json.loads((tmp_path / "runs" / "agent-case-01" / "director_failure.json").read_text(encoding="utf-8"))
    assert "structured provider unavailable" in failure["reason"]


def test_dynamic_director_provider_failure_preserves_exact_call_provenance(tmp_path) -> None:
    from scripts.prepare_real_jobs import prepare_jobs
    from videoact.provider_provenance import make_call_record

    dataset = tmp_path / "dataset"
    _dataset(dataset)

    class FailingProvider:
        provider_kind = "codex_exec_local"
        model_id = "codex-cli"
        model_version = "codex-exec-v1"
        template_backed = False
        llm_generated = True

        def __init__(self):
            self.call_records = []

        def __call__(self, request):
            self.call_records.append(
                make_call_record(
                    stage="director",
                    provider_kind=self.provider_kind,
                    model_id=self.model_id,
                    model_version=self.model_version,
                    call_id="codex-exec:director:failed-call",
                    prompt=request.prompt,
                    request_schema={"type": "object"},
                    response_schema={"type": "object"},
                    template_backed=False,
                    llm_generated=True,
                    error="TimeoutExpired: structured call exceeded timeout",
                )
            )
            raise RuntimeError("structured provider unavailable")

        def last_call(self, stage=None):
            if stage is None:
                return self.call_records[-1] if self.call_records else None
            for record in reversed(self.call_records):
                if record.get("stage") == stage:
                    return record
            return None

    provider = FailingProvider()
    from videoact.director import DirectorAgent

    index = prepare_jobs(
        "train",
        tmp_path / "runs",
        dataset_root=dataset,
        harness_version="agent-codegen-v1",
        director_agent=DirectorAgent.from_provider(provider, provider_name="codex-exec-local"),
        code_agent=BlenderCodeAgent(provider=lambda _payload: {}),
    )

    assert index["jobs"][0]["status"] == "director_failed"
    manifest = json.loads(
        (tmp_path / "runs" / "agent-case-01" / "provider_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "error"
    assert manifest["stages"]["director"]["call_id"] == "codex-exec:director:failed-call"
    assert manifest["stages"]["director"]["error"].startswith("TimeoutExpired:")
    assert "preparation_gate" in manifest["stages"]
    assert manifest["llm_generated"] is None
