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
from evaluator.physics_oracle import PHYSICS_ORACLE_VERSION, evaluate_physics_oracle  # noqa: E402
from evaluator.schemas import VLMJudgeResponse  # noqa: E402
from evaluator.findings import deduplicate_findings  # noqa: E402
from evaluator.realism import REALISM_EVALUATOR_VERSION  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION  # noqa: E402
from evaluator.formal_config import FormalEvaluatorConfig  # noqa: E402
from videoact.contracts import Finding, SceneContract, TrajectoryPlan  # noqa: E402
from videoact.director_contracts import DirectorPlan  # noqa: E402
from videoact.real_artifacts import RealArtifactGate  # noqa: E402
from videoact.real_pipeline import RealRunStateMachine  # noqa: E402
from videoact.real_video import assemble_mp4_from_pngs  # noqa: E402
from videoact.observer_contract import read_trusted_observer_output  # noqa: E402
from videoact.experiment_fingerprint import build_from_run_dir, hash_file  # noqa: E402
from blender.lib.__meta__ import collect_library_signatures  # noqa: E402
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


def _formal_score_policy_payload(config_path: str | Path | None) -> tuple[dict[str, Any], str]:
    """Load the evaluator policy that belongs to a trusted real run."""

    path = Path(config_path) if config_path is not None else ROOT / "config" / "formal-evaluator-v1.json"
    if not path.is_absolute():
        path = ROOT / path
    formal = FormalEvaluatorConfig.from_path(path)
    payload = {
        "config_path": str(path.resolve()),
        "config": formal.as_dict(),
        "config_fingerprint": formal.fingerprint(),
        "scoring_source_sha256": hash_file(ROOT / "evaluator" / "scoring_v7.py"),
        "evidence_source_sha256": hash_file(ROOT / "evaluator" / "evidence.py"),
        "visual_primary_source_sha256": hash_file(ROOT / "evaluator" / "visual_primary.py"),
    }
    model_identity = (
        f"primary={formal.primary_judge_model_id};"
        f"audit={formal.audit_judge_model_id};"
        f"director={formal.director_provider_kind}:{formal.director_model_id};"
        f"codegen={formal.codegen_provider_kind}:{formal.codegen_model_id};"
        f"generator={formal.generator_model_id}"
    )
    return payload, model_identity


def evaluate_real_run(
    run_dir: str | Path,
    *,
    record: dict[str, Any] | None = None,
    blender_bin: str | Path | None = None,
    dataset_fingerprint: str | None = None,
    evaluator_model_id: str = "gpt-5.6-luna",
    patch_hash: str | None = None,
    formal_evaluator_config: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    contract = SceneContract.model_validate(json.loads((root / "scene_contract.json").read_text(encoding="utf-8")))
    plan = TrajectoryPlan.model_validate(json.loads((root / "trajectory.json").read_text(encoding="utf-8")))
    director_plan = None
    director_plan_path = root / "director_plan.json"
    if director_plan_path.is_file():
        director_plan = DirectorPlan.model_validate(json.loads(director_plan_path.read_text(encoding="utf-8")))
    telemetry = {}
    telemetry_failure: str | None = None
    observer_report: dict[str, Any] | None = None
    if manifest.get("trusted_observer_required") is True:
        observer_report = read_trusted_observer_output(
            root,
            observer_source_path=ROOT / "blender" / "trusted_observer.py",
        )
        if observer_report.get("status") == "pass":
            telemetry = observer_report.get("telemetry") or {}
            declared_source_hash = str(manifest.get("observer_source_hash") or "")
            observed_source_hash = str((observer_report.get("manifest") or {}).get("observer_source_hash") or "")
            if declared_source_hash and declared_source_hash != observed_source_hash:
                telemetry = {}
                telemetry_failure = "trusted_observer:run_manifest_source_hash_mismatch"
        else:
            telemetry_failure = "trusted_observer:" + ",".join(observer_report.get("failures", []))
        (root / "observer_report.json").write_text(
            json.dumps(observer_report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    else:
        # Legacy diagnostic artifacts may still contain telemetry, but this
        # branch is never used for a run explicitly requiring the trusted
        # observer.  Formal runs set the flag in run_manifest.json.
        telemetry_path = root / "telemetry.json"
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

    experiment_fingerprint = None
    experiment_fingerprint_error = None
    if manifest.get("trusted_observer_required") is True:
        try:
            score_policy_payload, configured_model_identity = _formal_score_policy_payload(formal_evaluator_config)
            experiment_fingerprint = build_from_run_dir(
                root,
                dataset_fingerprint=str(dataset_fingerprint or ""),
                blender_binary=blender_bin or "",
                observer_source_path=ROOT / "blender" / "trusted_observer.py",
                python_lock_path=ROOT / "uv.lock",
                library_payload=collect_library_signatures(),
                evaluator_prompt_payload={
                    "source_sha256": hash_file(ROOT / "evaluator" / "vlm_providers.py"),
                    "blind_review_version": "primary-blind-v1",
                    "formal_config_fingerprint": score_policy_payload["config_fingerprint"],
                },
                evaluator_schema_payload=VLMJudgeResponse.model_json_schema(),
                evaluator_model_id=configured_model_identity or evaluator_model_id,
                score_policy_payload=score_policy_payload,
                frame_sampler_version="event-aligned-uniform-v1",
                patch_hash=patch_hash,
            )
            fingerprint_payload = experiment_fingerprint.model_dump(mode="json")
            (root / "experiment_fingerprint.json").write_text(
                json.dumps(fingerprint_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest["experiment_fingerprint"] = fingerprint_payload
            (root / "run_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            experiment_fingerprint_error = f"experiment_fingerprint:{type(exc).__name__}:{exc}"

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
    if experiment_fingerprint_error is not None:
        fingerprint_finding = Finding(
            failure_id="experiment_fingerprint_unavailable",
            owner="blender_executor",
            category="provenance",
            severity="hard",
            root_cause_id="experiment_fingerprint_incomplete",
            message=experiment_fingerprint_error,
            evidence=[experiment_fingerprint_error],
            repair_route="runtime_repair",
        )
        findings = deduplicate_findings([*report.findings, fingerprint_finding])
        report = report.model_copy(
            update={
                "terminal_status": "fail",
                "hard_gate_failed": True,
                "score": DeterministicEvaluator._score_findings(findings),
                "findings": findings,
                "metrics": {**report.metrics, "finding_count": float(len(findings)), "hard_count": float(sum(finding.severity == "hard" for finding in findings))},
            }
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
    physics_report: dict[str, Any] | None = None
    if record is not None:
        physics_report = evaluate_physics_oracle(
            record,
            telemetry,
            contract=contract,
            trajectory_plan=plan,
        )
        (root / "physics_oracle.json").write_text(
            json.dumps(physics_report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        oracle_findings = evaluate_independent_oracle(
            record,
            contract,
            plan,
            telemetry=telemetry,
            physics_report=physics_report,
        )
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
                "observer_report": str((root / "observer_report.json").resolve()) if (root / "observer_report.json").is_file() else None,
                "physics_oracle": str((root / "physics_oracle.json").resolve()) if (root / "physics_oracle.json").is_file() else None,
                "physics_oracle_version": PHYSICS_ORACLE_VERSION,
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
        "physics_oracle": physics_report,
        "experiment_fingerprint": experiment_fingerprint.model_dump(mode="json") if experiment_fingerprint is not None else None,
        "realism": realism,
    }


def evaluate_real_split(
    root: str | Path,
    dataset_root: str | Path | None = None,
    *,
    blender_bin: str | Path | None = None,
    formal_evaluator_config: str | Path | None = None,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    dataset_fingerprint = None
    if dataset_root is not None:
        metadata_path = Path(dataset_root) / "metadata.json"
        if metadata_path.is_file():
            try:
                dataset_fingerprint = json.loads(metadata_path.read_text(encoding="utf-8")).get("fingerprint")
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                dataset_fingerprint = None
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
            dataset_fingerprint=dataset_fingerprint,
            formal_evaluator_config=formal_evaluator_config,
        )
        for path in discover_run_dirs(root)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--formal-evaluator-config", default=None)
    args = parser.parse_args()
    split_kwargs: dict[str, Any] = {"blender_bin": args.blender_bin}
    if args.formal_evaluator_config is not None:
        split_kwargs["formal_evaluator_config"] = args.formal_evaluator_config
    print(
        json.dumps(
            evaluate_real_split(args.run_root, args.dataset_root, **split_kwargs),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
