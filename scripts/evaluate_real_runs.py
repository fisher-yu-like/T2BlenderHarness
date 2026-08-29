"""Assemble real proxy videos and run artifact plus deterministic evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator.deterministic import DeterministicEvaluator  # noqa: E402
from evaluator.director_metrics import evaluate_director_plan  # noqa: E402
from evaluator.independent_oracle import evaluate_independent_oracle  # noqa: E402
from evaluator.interaction_metrics import evaluate_interactions  # noqa: E402
from evaluator.findings import deduplicate_findings  # noqa: E402
from evaluator.realism import REALISM_EVALUATOR_VERSION  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION  # noqa: E402
from videoact.contracts import Finding, SceneContract, TrajectoryPlan  # noqa: E402
from videoact.director_contracts import DirectorPlan  # noqa: E402
from videoact.real_artifacts import RealArtifactGate  # noqa: E402
from videoact.real_pipeline import RealRunStateMachine  # noqa: E402
from videoact.real_video import assemble_mp4_from_pngs  # noqa: E402
from scripts.evaluate_proxy_realism import audit_run  # noqa: E402


def discover_run_dirs(root: str | Path) -> list[Path]:
    """Find prepared real runs without assuming a synthetic ``case-*`` prefix."""
    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_dir() and (path / "run_manifest.json").is_file()
    )


def evaluate_real_run(
    run_dir: str | Path,
    *,
    record: dict[str, Any] | None = None,
    blender_bin: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    contract = SceneContract.model_validate(json.loads((root / "scene_contract.json").read_text(encoding="utf-8")))
    plan = TrajectoryPlan.model_validate(json.loads((root / "trajectory.json").read_text(encoding="utf-8")))
    director_plan = None
    director_plan_path = root / "director_plan.json"
    if director_plan_path.is_file():
        director_plan = DirectorPlan.model_validate(json.loads(director_plan_path.read_text(encoding="utf-8")))
    telemetry_path = root / "telemetry.json"
    telemetry = {}
    telemetry_failure: str | None = None
    if telemetry_path.is_file():
        try:
            loaded_telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_telemetry, dict):
                telemetry_failure = "telemetry_not_an_object"
            else:
                telemetry = loaded_telemetry
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            telemetry_failure = f"telemetry_invalid:{type(exc).__name__}"
    else:
        telemetry_failure = "telemetry_missing"
    machine = RealRunStateMachine(root, case_id=manifest["case_id"])
    if machine.state == "executing" and telemetry_failure is None:
        machine.record_mcp_response({"status": "success", "blender_version": telemetry.get("blender_version")})

    animation_frames = sorted((root / "frames" / "animation").glob("frame_*.png"))
    if not (root / "proxy.mp4").exists() and animation_frames:
        assemble_mp4_from_pngs(animation_frames, root / "proxy.mp4", fps=manifest["fps"])

    artifacts = machine.validate_artifacts(RealArtifactGate())
    (root / "artifact_report.json").write_text(
        json.dumps(artifacts.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = DeterministicEvaluator().evaluate_real(
        contract,
        plan,
        telemetry,
        artifacts,
        director_plan=director_plan,
    )
    if telemetry_failure is not None:
        telemetry_finding = Finding(
            failure_id="telemetry_unavailable",
            owner="blender_executor",
            category="telemetry",
            severity="hard",
            root_cause_id="runtime_telemetry_missing",
            message=f"real Blender telemetry cannot be read: {telemetry_failure}",
            evidence=[telemetry_failure],
            repair_route="runtime_repair",
        )
        findings = deduplicate_findings([*report.findings, telemetry_finding])
        report = report.model_copy(
            update={
                "terminal_status": "fail",
                "hard_gate_failed": True,
                "score": DeterministicEvaluator._score_findings(findings),
                "findings": findings,
                "metrics": {
                    **report.metrics,
                    "finding_count": float(len(findings)),
                    "hard_count": float(sum(finding.severity == "hard" for finding in findings)),
                },
            }
        )
    director_findings = []
    interaction_findings = []
    if record is not None:
        oracle_findings = evaluate_independent_oracle(record, contract, plan, telemetry=telemetry)
        if director_plan is not None:
            director_report = evaluate_director_plan(director_plan, plan, telemetry=telemetry)
            interaction_findings = evaluate_interactions(director_plan, plan, telemetry=telemetry)
            director_findings = deduplicate_findings([*director_report.findings, *interaction_findings])
            all_findings = deduplicate_findings([*report.findings, *director_findings, *oracle_findings])
            report = report.model_copy(
                update={
                    "evaluator_version": "real-v5-independent-oracle-director",
                    "terminal_status": "fail" if any(finding.severity == "hard" for finding in all_findings) else "pass",
                    "hard_gate_failed": any(finding.severity == "hard" for finding in all_findings),
                    "score": DeterministicEvaluator._score_findings(all_findings),
                    "findings": all_findings,
                    "director_plan_score": director_report.director_plan_score,
                    "director_findings": director_report.findings,
                    "interaction_findings": interaction_findings,
                    "metrics": {
                        **report.metrics,
                        "director_finding_count": float(len(director_report.findings)),
                        "interaction_finding_count": float(len(interaction_findings)),
                        "independent_oracle_finding_count": float(len(oracle_findings)),
                        "finding_count": float(len(all_findings)),
                        "hard_count": float(sum(finding.severity == "hard" for finding in all_findings)),
                    },
                }
            )
        else:
            all_findings = deduplicate_findings([*report.findings, *oracle_findings])
            report = report.model_copy(
                update={
                    "evaluator_version": "real-v3-declarative-independent-oracle",
                    "terminal_status": "fail" if any(finding.severity == "hard" for finding in all_findings) else "pass",
                    "hard_gate_failed": any(finding.severity == "hard" for finding in all_findings),
                    "score": DeterministicEvaluator._score_findings(all_findings),
                    "findings": all_findings,
                    "metrics": {
                        **report.metrics,
                        "independent_oracle_finding_count": float(len(oracle_findings)),
                        "finding_count": float(len(all_findings)),
                        "hard_count": float(sum(finding.severity == "hard" for finding in all_findings)),
                        "error_count": float(sum(finding.severity == "error" for finding in all_findings)),
                        "warning_count": float(sum(finding.severity == "warning" for finding in all_findings)),
                        "unique_root_cause_count": float(len(all_findings)),
                    },
                }
            )
    (root / "deterministic_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8"
    )
    # Geometry and sampled-frame evidence share this real-run pass.  It is
    # deliberately separate from the task score and never calls a VLM.
    realism = {
        "evaluator_version": REALISM_EVALUATOR_VERSION,
        "status": "unavailable",
        "reason": "proxy.blend or configured Blender binary is unavailable",
        "score": None,
        "score_kind": "artifact_only_proxy_unavailable",
    }
    blender_path = Path(blender_bin) if blender_bin else None
    if (root / "proxy.blend").is_file() and blender_path and blender_path.is_file():
        try:
            audit = audit_run(root, blender_bin=str(blender_path))
            realism = audit.get("realism") or realism
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            realism["reason"] = f"combined realism audit failed: {type(exc).__name__}: {exc}"
    (root / "combined_evaluator.json").write_text(
        json.dumps(
            {
                "evaluator_version": VISUAL_PRIMARY_VERSION,
                "artifact_report": str((root / "artifact_report.json").resolve()),
                "artifact_hash": artifacts.artifact_hash,
                "deterministic_report": str((root / "deterministic_report.json").resolve()),
                "geometry_report": str((root / "geometry_report.json").resolve()) if (root / "geometry_report.json").is_file() else None,
                "visual_evidence": str((root / "visual_evidence.json").resolve()) if (root / "visual_evidence.json").is_file() else None,
                "real_video_evidence": str((root / "local_video_evidence.json").resolve()) if (root / "local_video_evidence.json").is_file() else None,
                "realism_report": str((root / "realism_report.json").resolve()) if (root / "realism_report.json").is_file() else None,
                "vlm_policy": "one shared local visual review after artifact, deterministic, geometry, and MP4 gates; task, visual, physical, trajectory, camera, and realism channels remain separate",
                "real_video_evidence": "proxy.mp4 decoded plus Blender runtime_observations; missing either source is unavailable",
                "scores_are_added": False,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if report.terminal_status == "pass" and machine.state == "artifact_valid":
        machine.transition("evaluated", {"score": report.score})
    elif report.terminal_status == "fail" and machine.state not in {"failed", "evaluated"}:
        machine.transition(
            "failed",
            {
                "findings": [
                    finding.failure_id
                    for finding in [*report.findings, *director_findings]
                ]
            },
        )
    return {
        "case_id": manifest["case_id"],
        "status": report.terminal_status,
        "score": report.score,
        "state": machine.state,
        "artifact_status": artifacts.artifact_status,
        "hard_failures": artifacts.hard_failures,
        "artifact_hash": artifacts.artifact_hash,
        "artifact_hashes": artifacts.artifact_hashes,
        "code_hash": manifest.get("code_hash"),
        "findings": [finding.failure_id for finding in report.findings],
        "finding_details": [finding.model_dump(mode="json") for finding in report.findings],
        "director_plan_score": report.director_plan_score,
        "director_findings": [finding.model_dump(mode="json") for finding in report.director_findings],
        "interaction_findings": [finding.model_dump(mode="json") for finding in report.interaction_findings],
        "realism": realism,
    }


def evaluate_real_split(
    root: str | Path,
    dataset_root: str | Path | None = None,
    *,
    blender_bin: str | Path | None = None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if dataset_root is not None:
        records = {
            record["case_id"]: record
            for record in (
                json.loads(line)
                for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
    return [
        evaluate_real_run(
            path,
            record=records.get(json.loads((path / "run_manifest.json").read_text(encoding="utf-8")).get("case_id")),
            blender_bin=blender_bin,
        )
        for path in discover_run_dirs(root)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_real_split(args.run_root, args.dataset_root, blender_bin=args.blender_bin),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
