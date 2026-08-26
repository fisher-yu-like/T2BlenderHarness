"""Bounded single-sample plan/execute/evaluate/repair loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blender.proxy_scene import compile_proxy_script
from evaluator.deterministic import DeterministicEvaluator

from .blender_adapter import BlenderAdapter
from .contracts import RunManifest, RunResult
from .director import DirectorAgent
from .run_manifest import hash_payload, hash_prompt, write_manifest


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_inner_loop(
    case: dict[str, Any],
    harness_snapshot: dict[str, Any],
    output_dir: str | Path,
    *,
    adapter: Any | None = None,
    max_attempts: int = 6,
) -> RunResult:
    if not 1 <= max_attempts <= 6:
        raise ValueError("max_attempts must be between 1 and 6")
    prompt = str(case.get("prompt", ""))
    case_id = str(case.get("case_id", "case"))
    duration_s = float(case.get("duration_s", 10.0))
    fps = int(case.get("fps", 24))
    harness_version = str(harness_snapshot.get("version", "dev"))
    evaluator_version = str(harness_snapshot.get("evaluator_version", "deterministic-v1"))
    root = Path(output_dir)
    attempts_root = root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)

    director_result = DirectorAgent().plan(
        prompt,
        scene_id=case_id,
        duration_s=duration_s,
        fps=fps,
    )
    contract = director_result.scene_contract
    plan = director_result.trajectory_plan
    prompt_hash = hash_prompt(prompt)
    plan_hash = hash_payload(plan.model_dump(mode="json"))
    run_id = f"{case_id}-{prompt_hash[:10]}"
    execution_adapter = adapter or BlenderAdapter()
    evaluator = DeterministicEvaluator()
    records: list[dict[str, Any]] = []
    all_findings = []

    for attempt_number in range(1, max_attempts + 1):
        attempt_dir = attempts_root / f"{attempt_number:02d}"
        attempt_dir.mkdir(parents=False, exist_ok=False)
        manifest = RunManifest(
            run_id=run_id,
            scene_id=contract.scene_id,
            prompt_hash=prompt_hash,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            plan_hash=plan_hash,
            backend="fake" if adapter is not None else "mcp",
            frame_start=plan.timebase.frame_start,
            frame_end=plan.timebase.frame_end,
            artifacts={
                "plan": "plan.json",
                "director_plan": "director_plan.json",
                "trajectory": "trajectory.json",
                "camera_plan": "camera_plan.json",
                "blender_script": "blender_script.py",
                "deterministic_report": "deterministic_report.json",
            },
        )
        _write_json(attempt_dir / "director_plan.json", director_result.director_plan.model_dump(mode="json"))
        _write_json(attempt_dir / "plan.json", plan.model_dump(mode="json"))
        _write_json(attempt_dir / "trajectory.json", plan.model_dump(mode="json"))
        _write_json(attempt_dir / "camera_plan.json", plan.camera.model_dump(mode="json"))
        script_path = attempt_dir / "blender_script.py"
        compile_proxy_script(plan, manifest, script_path)
        write_manifest(manifest, attempt_dir / "run_manifest.json")
        _write_json(
            attempt_dir / "attempt_manifest.json",
            {
                "attempt": attempt_number,
                "manifest_hash": manifest.content_hash(),
                "run_manifest": manifest.model_dump(mode="json"),
            },
        )

        execution = execution_adapter.run(script_path, attempt_dir, prefer="mcp")
        report = evaluator.evaluate(contract, plan, execution=execution)
        _write_json(attempt_dir / "deterministic_report.json", report.model_dump(mode="json"))
        (attempt_dir / "mcp_calls.jsonl").touch(exist_ok=True)
        record = {
            "attempt": attempt_number,
            "status": report.terminal_status,
            "score": report.score,
            "execution_status": execution.status,
            "manifest_hash": manifest.content_hash(),
            "report_path": str(attempt_dir / "deterministic_report.json"),
        }
        records.append(record)
        all_findings.extend(report.findings)

        if report.terminal_status == "pass":
            final_dir = root / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                final_dir / "selection.json",
                {
                    "run_id": run_id,
                    "selected_attempt": attempt_number,
                    "score": report.score,
                    "prompt_hash": prompt_hash,
                    "harness_version": harness_version,
                    "plan_hash": plan_hash,
                    "director_plan_hash": director_result.director_plan_hash,
                    "attempt_manifest": str(attempt_dir / "attempt_manifest.json"),
                },
            )
            return RunResult(
                run_id=run_id,
                status="success",
                selected_attempt=attempt_number,
                attempts=records,
                findings=all_findings,
                final_score=report.score,
            )

    return RunResult(
        run_id=run_id,
        status="exhausted",
        attempts=records,
        findings=all_findings,
        final_score=None,
    )
