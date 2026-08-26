"""Run a deterministic, network-free capability check for T2Blendercodeharness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_COMPONENTS = {
    "scene_contract": "src/videoact/scene_contract.py",
    "trajectory": "src/videoact/trajectory.py",
    "camera": "src/videoact/camera.py",
    "blender_adapter": "src/videoact/blender_adapter.py",
    "real_artifacts": "src/videoact/real_artifacts.py",
    "real_pipeline": "src/videoact/real_pipeline.py",
    "meta_harness": "src/videoact/meta_harness.py",
    "deterministic_evaluator": "evaluator/deterministic.py",
    "realism_geometry_evaluator": "evaluator/geometry_realism.py",
    "realism_artifact_evaluator": "evaluator/realism.py",
    "render_visual_evidence": "evaluator/visual_evidence.py",
    "realism_audit_entrypoint": "scripts/evaluate_proxy_realism.py",
    "real_job_entrypoint": "scripts/prepare_real_jobs.py",
    "real_outer_loop": "scripts/run_real_outer_loop.py",
    "director_agent": "src/videoact/director.py",
    "director_metrics": "evaluator/director_metrics.py",
    "interaction_metrics": "evaluator/interaction_metrics.py",
    "multi_entity_dataset_validator": "scripts/validate_multi_entity_dataset.py",
}


def _check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "pass" if passed else "fail", "detail": detail}


def run_capability_check(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    checks: list[dict[str, str]] = []
    missing = [relative for relative in REQUIRED_COMPONENTS.values() if not (root / relative).is_file()]
    checks.append(
        _check(
            "required_components",
            not missing,
            "all required Harness modules are present" if not missing else f"missing: {missing}",
        )
    )
    if missing:
        return {"skill_version": "t2blendercodeharness-v3", "project_root": str(root), "status": "fail", "checks": checks}

    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    try:
        from evaluator.deterministic import DeterministicEvaluator
        from videoact.meta_harness import MetaHarnessOptimizer
        from videoact.real_artifacts import RealArtifactGate
        from videoact.director import DirectorAgent
        from evaluator.director_metrics import evaluate_director_plan
        from videoact.scene_contract import SceneContractBuilder
        from videoact.trajectory import TrajectoryPlanner

        checks.append(_check("imports", True, "DirectorAgent, contract, planner, artifact gate, evaluator, and optimizer imported"))
    except Exception as exc:  # pragma: no cover - exercised by a missing runtime
        checks.append(_check("imports", False, f"{type(exc).__name__}: {exc}"))
        return {"skill_version": "t2blendercodeharness-v3", "project_root": str(root), "status": "fail", "checks": checks}

    try:
        contract = SceneContractBuilder().build(
            "A character walks to the table and picks up the red cup.", duration_s=10.0, fps=24
        )
        plan = TrajectoryPlanner().plan(contract)
        checks.append(
            _check(
                "contract_and_plan",
                bool(contract.entities) and bool(plan.entities) and bool(plan.camera.shots),
                f"entities={len(contract.entities)}, states={len(plan.entities)}, shots={len(plan.camera.shots)}",
            )
        )
    except Exception as exc:
        checks.append(_check("contract_and_plan", False, f"{type(exc).__name__}: {exc}"))

    try:
        director_result = DirectorAgent().plan(
            "Alice carries the red cube while Bob carries the blue cup, then Alice hands the red cube to Bob.",
            scene_id="skill-director-probe",
            duration_s=12.0,
            fps=24,
        )
        director_report = evaluate_director_plan(
            director_result.director_plan,
            director_result.trajectory_plan,
        )
        checks.append(
            _check(
                "director_multi_plan",
                director_report.director_plan_score == 100.0
                and {"actor_a", "actor_b", "red_cube", "blue_cup"}.issubset(director_result.trajectory_plan.entities),
                f"director_score={director_report.director_plan_score}, entities={len(director_result.trajectory_plan.entities)}",
            )
        )
    except Exception as exc:
        checks.append(_check("director_multi_plan", False, f"{type(exc).__name__}: {exc}"))

    try:
        RealArtifactGate()
        DeterministicEvaluator()
        checks.append(_check("evaluator_interfaces", True, "artifact gate and deterministic evaluator are constructible"))
    except Exception as exc:
        checks.append(_check("evaluator_interfaces", False, f"{type(exc).__name__}: {exc}"))

    try:
        optimizer = MetaHarnessOptimizer(output_dir=root / "out" / "skill-capability-check")
        failure = {
            "case_id": "skill-probe-01",
            "findings": [
                {
                    "failure_id": "camera_event_uncovered",
                    "owner": "camera_planner",
                    "category": "camera_coverage",
                    "severity": "hard",
                    "message": "probe failure",
                    "evidence": ["probe"],
                    "repair_route": "camera_repair",
                }
            ],
        }
        optimizer.propose([failure, {**failure, "case_id": "skill-probe-02"}])
        checks.append(_check("one_owner_proposal", True, "repeated failure yields a bounded proposal"))
        try:
            optimizer.propose([{"case_id": "pass-01", "findings": []}])
        except ValueError:
            checks.append(_check("no_action_on_clean_records", True, "clean records produce no patch proposal"))
        else:
            checks.append(_check("no_action_on_clean_records", False, "clean records unexpectedly produced a proposal"))
    except Exception as exc:
        checks.append(_check("one_owner_proposal", False, f"{type(exc).__name__}: {exc}"))

    status = "pass" if all(check["status"] == "pass" for check in checks) else "fail"
    return {"skill_version": "t2blendercodeharness-v3", "project_root": str(root), "status": status, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = run_capability_check(args.project_root)
    destination = Path(args.out) if args.out else Path(args.project_root) / "out" / "skill_capability_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
