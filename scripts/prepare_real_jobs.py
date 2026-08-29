"""Prepare immutable Blender MCP jobs for real proxy rendering."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blender.real_proxy_job import compile_real_proxy_job
from blender.lib.__meta__ import collect_library_signatures
from videoact.blender_code_agent import BlenderCodeAgent, validate_generated_source
from videoact.case_coverage import validate_case_coverage
from videoact.code_cache import CodeCache
from videoact.codegen_context import load_validated_context_examples
from videoact.codegen_contracts import CodegenRequest, CodegenResponse, FunctionSignature
from videoact.director import DirectorAgent
from videoact.fallback_codegen import FallbackCodegen
from videoact.real_artifacts import RealRunManifest, fingerprint_real_run
from videoact.run_manifest import hash_payload, hash_prompt, write_manifest


RENDER_SETTINGS = {"engine": "BLENDER_EEVEE_NEXT", "resolution": [256, 256], "samples": 16}


def _load_records(dataset_root: str | Path, split: str, case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["case_id"]: record for record in records}
    if split == "calibration":
        labels = [json.loads(line) for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [label["case_id"] for label in labels]
    else:
        ids = json.loads((root / "splits.json").read_text(encoding="utf-8"))[split]
    if case_ids is not None:
        unknown = set(case_ids) - set(ids)
        if unknown:
            raise ValueError(f"case IDs are not all in {split} split: {sorted(unknown)}")
        ids = case_ids
    return [by_id[case_id] for case_id in ids]


def _case_obligations(record: dict[str, Any]) -> dict[str, list[str]]:
    """Expose stable case IDs to the planner without replacing prompt semantics."""

    oracle = record.get("oracle_expectations") or {}
    proxy_scene = record.get("proxy_scene") or {}
    entities = record.get("entities") or (proxy_scene.get("entities") or [])
    entity_ids = list(oracle.get("required_entity_ids") or [item.get("id") for item in entities if item.get("id")])
    event_graph = record.get("event_graph") or []
    event_ids = list(record.get("required_events") or oracle.get("event_order") or [item.get("id") for item in event_graph if item.get("id")])
    camera = proxy_scene.get("camera") or {}
    camera_ids = list(oracle.get("required_camera_events") or camera.get("must_show_events") or [])
    return {
        "required_entity_ids": [str(item) for item in entity_ids if str(item).strip()],
        "required_event_ids": [str(item) for item in event_ids if str(item).strip()],
        "required_camera_event_ids": [str(item) for item in camera_ids if str(item).strip()],
    }


def prepare_jobs(
    split: str,
    out_dir: str | Path,
    *,
    dataset_root: str | Path = "dataset",
    harness_version: str = "h1-real",
    evaluator_version: str = "deterministic-v1",
    case_ids: list[str] | None = None,
    render_settings: dict[str, Any] | None = None,
    generation_mode: Literal["agent", "template_baseline"] = "agent",
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    fallback_codegen: FallbackCodegen | None = None,
    code_cache_dir: str | Path | None = None,
    codegen_examples_root: str | Path | None = "dataset/codegen-examples-v1",
) -> dict[str, Any]:
    if split not in {"calibration", "train", "dev", "test"}:
        raise ValueError("split must be calibration, train, dev, or test")
    if generation_mode not in {"agent", "template_baseline"}:
        raise ValueError("generation_mode must be agent or template_baseline")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    # The legacy template is an explicit calibration/baseline path.  It must
    # not be routed through the provider-backed DirectorAgent, because the
    # baseline is intentionally allowed to render prompts that the strict
    # agent contract would reject.  In agent mode, a dynamic provider is
    # mandatory and all failures remain visible in the job index.
    director = director_agent if generation_mode == "agent" else None
    agent = code_agent or BlenderCodeAgent()
    library_signatures = {
        category: [FunctionSignature.model_validate(item) for item in entries]
        for category, entries in collect_library_signatures().items()
    }
    cache = CodeCache(code_cache_dir or output / "code_cache")
    effective_render_settings = {**RENDER_SETTINGS, **(render_settings or {})}
    context_examples = []
    context_status = "none"
    context_error = None
    if generation_mode == "agent" and codegen_examples_root is not None:
        try:
            context_examples, context_status = load_validated_context_examples(codegen_examples_root)
        except Exception as exc:
            context_status = "invalid"
            context_error = f"{type(exc).__name__}: {exc}"
    jobs = []
    frozen_sources: dict[str, str] = cache.frozen_source_hashes(harness_version) if generation_mode == "agent" else {}
    for record in _load_records(dataset_root, split, case_ids):
        case_id = record["case_id"]
        run_dir = output / case_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"real run directory already contains artifacts: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        if generation_mode == "agent" and context_status == "invalid":
            failure = {
                "status": "codegen_failed",
                "reason": "invalid_codegen_context",
                "context_root": str(Path(codegen_examples_root).resolve()) if codegen_examples_root is not None else None,
                "context_error": context_error,
                "harness_version": harness_version,
            }
            failure_path = run_dir / "codegen_failure.json"
            failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
            jobs.append(
                {
                    "case_id": case_id,
                    "run_dir": str(run_dir),
                    "plan_hash": None,
                    "director_plan_hash": None,
                    "generation_mode": generation_mode,
                    "codegen_context_status": context_status,
                    "codegen_context_example_ids": [],
                    "status": "codegen_failed",
                    "failure_path": str(failure_path.resolve()),
                }
            )
            continue
        if generation_mode == "template_baseline":
            # No silent fallback: this branch is selected only by the caller
            # as an explicit baseline experiment.
            try:
                # Preserve the historical deterministic plan when it can
                # parse the baseline prompt; only the explicitly selected
                # baseline arm may use the relaxed projection for prompts it
                # cannot execute (for example a still-life calibration case).
                try:
                    director_result = DirectorAgent().plan(
                        record["prompt"],
                        scene_id=case_id,
                        duration_s=record["duration_s"],
                        fps=record["fps"],
                    )
                except Exception:
                    director_result = DirectorAgent().plan_explicit_baseline(
                        record["prompt"],
                        scene_id=case_id,
                        duration_s=record["duration_s"],
                        fps=record["fps"],
                    )
                contract = director_result.scene_contract
                plan = director_result.trajectory_plan
                director_plan_hash = director_result.director_plan_hash
            except Exception as exc:
                failure = {
                    "status": "baseline_prepare_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "harness_version": harness_version,
                }
                failure_path = run_dir / "baseline_failure.json"
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
                jobs.append(
                    {
                        "case_id": case_id,
                        "run_dir": str(run_dir),
                        "plan_hash": None,
                        "director_plan_hash": None,
                        "generation_mode": generation_mode,
                        "status": "baseline_prepare_failed",
                        "failure_path": str(failure_path.resolve()),
                    }
                )
                continue
        else:
            if director is None:
                failure = {
                    "status": "director_failed",
                    "reason": "agent generation requires an injected dynamic DirectorAgent provider",
                    "harness_version": harness_version,
                }
                failure_path = run_dir / "director_failure.json"
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
                jobs.append(
                    {
                        "case_id": case_id,
                        "run_dir": str(run_dir),
                        "plan_hash": None,
                        "director_plan_hash": None,
                        "generation_mode": generation_mode,
                        "status": "director_failed",
                        "failure_path": str(failure_path.resolve()),
                    }
                )
                continue
            if isinstance(director, DirectorAgent) and director.mode != "dynamic":
                failure = {
                    "status": "director_failed",
                    "reason": "agent generation requires a dynamic provider-assisted DirectorAgent; deterministic baseline is explicit-only",
                    "harness_version": harness_version,
                }
                failure_path = run_dir / "director_failure.json"
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
                jobs.append(
                    {
                        "case_id": case_id,
                        "run_dir": str(run_dir),
                        "plan_hash": None,
                        "director_plan_hash": None,
                        "generation_mode": generation_mode,
                        "status": "director_failed",
                        "failure_path": str(failure_path.resolve()),
                    }
                )
                continue
            try:
                director_result = director.plan(
                    record["prompt"],
                    scene_id=case_id,
                    duration_s=record["duration_s"],
                    fps=record["fps"],
                    obligations=_case_obligations(record),
                )
            except Exception as exc:
                failure = {
                    "status": "director_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "harness_version": harness_version,
                }
                failure_path = run_dir / "director_failure.json"
                failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
                jobs.append(
                    {
                        "case_id": case_id,
                        "run_dir": str(run_dir),
                        "plan_hash": None,
                        "director_plan_hash": None,
                        "generation_mode": generation_mode,
                        "status": "director_failed",
                        "failure_path": str(failure_path.resolve()),
                    }
                )
                continue
            contract = director_result.scene_contract
            plan = director_result.trajectory_plan
            director_plan_hash = director_result.director_plan_hash
        prompt_hash = hash_prompt(record["prompt"])
        plan_hash = hash_payload(plan.model_dump(mode="json"))
        manifest = RealRunManifest(
            run_id=f"real-{case_id}-{prompt_hash[:10]}",
            case_id=case_id,
            split=split,
            prompt_hash=prompt_hash,
            plan_hash=plan_hash,
            director_plan_hash=director_plan_hash,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_version="pending-mcp",
            fps=plan.timebase.fps,
            frame_start=plan.timebase.frame_start,
            frame_end=plan.timebase.frame_end,
            render_settings=effective_render_settings,
            fingerprint=fingerprint_real_run(
                prompt_hash=prompt_hash,
                plan_hash=plan_hash,
                director_plan_hash=director_plan_hash,
                harness_version=harness_version,
                evaluator_version=evaluator_version,
                blender_version="pending-mcp",
                render_settings=effective_render_settings,
            ),
            state="prepared",
        )
        if director_result is not None:
            (run_dir / "director_plan.json").write_text(
                json.dumps(director_result.director_plan.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        (run_dir / "scene_contract.json").write_text(json.dumps(contract.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "proxy_scene.json").write_text(json.dumps(record.get("proxy_scene", {}), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "trajectory.json").write_text(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "camera_plan.json").write_text(json.dumps(plan.camera.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        write_manifest(manifest, run_dir / "run_manifest.json")
        job_path = run_dir / "blender_job.py"
        job_entry: dict[str, Any] = {
            "case_id": case_id,
            "run_dir": str(run_dir),
            "plan_hash": plan_hash,
            "director_plan_hash": director_plan_hash,
            "generation_mode": generation_mode,
            "codegen_context_status": context_status if generation_mode == "agent" else "none",
            "codegen_context_example_ids": [example.case_id for example in context_examples] if generation_mode == "agent" else [],
        }
        if generation_mode == "template_baseline":
            source = compile_real_proxy_job(
                plan,
                manifest,
                run_dir,
                proxy_spec=record.get("proxy_scene"),
                director_plan=director_result.director_plan,
                director_trajectories=director_result.director_trajectories,
                director_camera=director_result.director_camera,
            )
            code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            manifest = manifest.model_copy(update={"code_hash": code_hash})
            write_manifest(manifest, run_dir / "run_manifest.json")
            job_path.write_bytes(source.encode("utf-8"))
            job_entry.update({"status": "prepared", "job_path": str(job_path), "code_hash": code_hash})
        else:
            request = CodegenRequest(
                director_plan=director_result.director_plan.model_dump(mode="json"),
                library_signatures=library_signatures,
                context_examples=context_examples,
                harness_version=harness_version,
                constraints=[
                    "no_penetration",
                    "visibility_checked",
                    "preserve_prompt_entities_and_events",
                    "emit_telemetry_and_sample_frames",
                ],
            )
            cached_source = cache.lookup(director_plan_hash, harness_version)
            if cached_source is not None:
                cached_violations = validate_generated_source(
                    cached_source,
                    allowed_library_calls=request.available_library_calls,
                )
                response = (
                    CodegenResponse(
                        status="hard_uncertainty",
                        uncertainties=[
                            {
                                "id": "cached_source_static_gate",
                                "description": ",".join(cached_violations),
                                "severity": "hard",
                                "resolved": False,
                            }
                        ],
                        llm_call_id="cache-invalid",
                    )
                    if cached_violations
                    else CodegenResponse(
                        status="success",
                        generated_code=cached_source,
                        llm_call_id="cache-hit",
                    )
                )
            else:
                response = agent.generate(request)
                if response.status == "library_insufficient" and fallback_codegen is not None:
                    response = fallback_codegen.generate(request, response)
            if response.status != "success":
                failure = {
                    "status": "codegen_failed",
                    "codegen_response": response.model_dump(mode="json"),
                    "director_plan_hash": director_plan_hash,
                    "harness_version": harness_version,
                }
                (run_dir / "codegen_failure.json").write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
                job_entry.update({"status": "codegen_failed", "failure_path": str((run_dir / "codegen_failure.json").resolve())})
                jobs.append(job_entry)
                continue

            source = response.generated_code
            coverage = validate_case_coverage(
                record=record,
                director_plan=director_result.director_plan,
                director_trajectories=director_result.director_trajectories,
                director_camera=director_result.director_camera,
                generated_code=source,
                existing_code_hashes={
                    key: value for key, value in frozen_sources.items() if key != director_plan_hash
                },
            )
            (run_dir / "coverage_report.json").write_text(
                json.dumps(coverage.model_dump(mode="json"), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if coverage.status != "pass":
                job_entry.update({"status": "coverage_failed", "failure_path": str((run_dir / "coverage_report.json").resolve())})
                jobs.append(job_entry)
                continue
            if cached_source is None:
                cache.store(
                    director_plan_hash,
                    harness_version,
                    source,
                    llm_call_id=response.llm_call_id,
                    metadata={
                        "context_status": context_status,
                        "context_example_ids": [example.case_id for example in context_examples],
                    },
                )
            code_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
            manifest = manifest.model_copy(update={"code_hash": code_hash})
            write_manifest(manifest, run_dir / "run_manifest.json")
            job_path.write_bytes(source.encode("utf-8"))
            frozen_sources[director_plan_hash] = hashlib.sha256(source.encode("utf-8")).hexdigest()
            job_entry.update(
                {
                    "status": "prepared",
                    "job_path": str(job_path),
                    "code_hash": code_hash,
                    "codegen_call_id": response.llm_call_id,
                }
            )
        jobs.append(job_entry)
    index = {
        "split": split,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "generation_mode": generation_mode,
        "case_count": len(jobs),
        "prepared_count": sum(job.get("status") == "prepared" for job in jobs),
        "failed_count": sum(job.get("status") != "prepared" for job in jobs),
        "codegen_context_status": context_status if generation_mode == "agent" else "none",
        "codegen_context_example_ids": [example.case_id for example in context_examples] if generation_mode == "agent" else [],
        "jobs": jobs,
    }
    (output / "job_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["calibration", "train", "dev", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--harness-version", default="t2blendercodeharness-v5-executable-director")
    parser.add_argument("--evaluator-version", default="visual-primary-v6-independent-channels")
    parser.add_argument("--generation-mode", choices=["agent", "template_baseline"], default="agent")
    parser.add_argument("--code-cache-dir", default=None)
    parser.add_argument("--codegen-examples-root", default="dataset/codegen-examples-v1")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    render_settings = {}
    if args.resolution:
        render_settings["resolution"] = args.resolution
    if args.samples is not None:
        render_settings["samples"] = args.samples
    print(json.dumps(prepare_jobs(args.split, args.out_dir, dataset_root=args.dataset_root, harness_version=args.harness_version, evaluator_version=args.evaluator_version, case_ids=args.case_id, render_settings=render_settings, generation_mode=args.generation_mode, code_cache_dir=args.code_cache_dir, codegen_examples_root=args.codegen_examples_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
