"""Build an evidence-only readiness report for the six-round protocol.

This module deliberately treats readiness as a gate matrix rather than a
score.  A template render, a numeric placeholder, or an unavailable provider
cannot satisfy an agent-training gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping
from typing_extensions import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


GateStatus = Literal["pass", "pending", "blocked", "fail"]
VALID_STATUSES = {"pass", "pending", "blocked", "fail"}
AUTOMATED_GATE_NAMES = ("full_test", "capability", "dataset", "frozen_eval")
REQUIRED_GATE_NAMES = AUTOMATED_GATE_NAMES + (
    "real_blender_smoke",
    "golden_review",
    "dynamic_agent_provider",
    "paired_gate",
)


def _normalise_gate(name: str, evidence: Any) -> tuple[dict[str, Any], list[str]]:
    numeric_substitutions: list[str] = []
    if evidence is None:
        return {"status": "pending", "reason": "evidence_not_supplied"}, numeric_substitutions
    if isinstance(evidence, (int, float)) and not isinstance(evidence, bool):
        numeric_substitutions.append(name)
        return {
            "status": "blocked",
            "reason": "numeric_value_is_not_gate_evidence",
            "observed_type": type(evidence).__name__,
        }, numeric_substitutions
    if isinstance(evidence, bool):
        numeric_substitutions.append(name)
        return {
            "status": "blocked",
            "reason": "boolean_value_is_not_a_provenance_report",
            "observed_type": "bool",
        }, numeric_substitutions

    if isinstance(evidence, str):
        raw_status = evidence.strip().lower()
        details: dict[str, Any] = {"status": raw_status}
    elif isinstance(evidence, Mapping):
        details = dict(evidence)
        raw_status = str(details.get("status") or "pending").strip().lower()
        details["status"] = raw_status
    else:
        return {"status": "blocked", "reason": "unsupported_gate_evidence_type"}, numeric_substitutions

    if raw_status == "unavailable":
        details["status"] = "blocked"
        details["reason"] = details.get("reason") or "provider_unavailable"
    elif raw_status in {"not_established", "needs_human_review"}:
        details["status"] = "pending"
        details["reason"] = details.get("reason") or raw_status
    elif raw_status not in VALID_STATUSES:
        details["status"] = "blocked"
        details["reason"] = f"unknown_gate_status:{raw_status}"

    if name == "real_blender_smoke" and details["status"] == "pass":
        if details.get("generation_mode") != "agent":
            details["status"] = "blocked"
            details["reason"] = "template_baseline_is_not_agent_smoke"
        elif details.get("artifact_status") != "complete":
            details["status"] = "blocked"
            details["reason"] = "agent_smoke_artifact_evidence_missing_or_incomplete"
    if name == "dynamic_agent_provider" and details["status"] == "pass":
        if details.get("director") != "pass" or details.get("blender_code") != "pass":
            details["status"] = "blocked"
            details["reason"] = "director_and_blender_code_pair_are_both_required"
        elif details.get("provider_mode") == "rule_template_baseline":
            details["status"] = "blocked"
            details["reason"] = "rule_template_baseline_is_diagnostic_only"
        elif details.get("provider_mode") in {"model", "glm"}:
            required_identity = (
                "generator_model_id",
                "primary_judge_model_id",
                "audit_judge_model_id",
            )
            missing_identity = [field for field in required_identity if not str(details.get(field) or "").strip()]
            if missing_identity:
                details["status"] = "blocked"
                details["reason"] = f"formal_model_identity_missing:{','.join(missing_identity)}"
            elif len({details[field] for field in required_identity}) != len(required_identity):
                details["status"] = "blocked"
                details["reason"] = "generator_and_judge_model_snapshots_must_be_distinct"
    if name == "golden_review" and details["status"] == "pass":
        annotators = details.get("annotators_per_sample")
        if annotators is not None and (not isinstance(annotators, int) or annotators < 2):
            details["status"] = "blocked"
            details["reason"] = "golden_review_needs_two_annotators_per_sample"
    return details, numeric_substitutions


def build_training_readiness(
    *,
    automated_checks: Mapping[str, Any],
    real_blender_smoke: Any,
    golden_review: Any,
    dynamic_agent_provider: Any,
    paired_gate: Any,
    formal_release_report: Any | None = None,
    experiment_contract: Any | None = None,
) -> dict[str, Any]:
    """Combine independent evidence without converting any value to a score."""

    evidence_by_name: dict[str, Any] = {
        **{name: automated_checks.get(name) for name in AUTOMATED_GATE_NAMES},
        "real_blender_smoke": real_blender_smoke,
        "golden_review": golden_review,
        "dynamic_agent_provider": dynamic_agent_provider,
        "paired_gate": paired_gate,
    }
    gate_names = list(REQUIRED_GATE_NAMES)
    formal_release_verification: dict[str, Any] | None = None
    if formal_release_report is not None:
        from videoact.release_gates import validate_formal_release_report

        formal_release_verification = validate_formal_release_report(formal_release_report)
        formal_release_evidence = dict(formal_release_report) if isinstance(formal_release_report, Mapping) else {}
        formal_release_evidence.update(
            {
                "status": formal_release_verification.get("status", "blocked"),
                "training_allowed": formal_release_verification.get("training_allowed") is True,
                "verification": formal_release_verification,
            }
        )
        evidence_by_name["formal_release"] = formal_release_evidence
        gate_names.append("formal_release")
    if experiment_contract is not None:
        try:
            from videoact.real_artifacts import validate_experiment_contract

            contract = validate_experiment_contract(experiment_contract)
            evidence_by_name["experiment_contract"] = {
                "status": "pass",
                "experiment_id": contract.experiment_id,
                "experiment_fingerprint": contract.experiment_fingerprint,
                "contract_version": contract.contract_version,
            }
        except (TypeError, ValueError):
            evidence_by_name["experiment_contract"] = {
                "status": "blocked",
                "reason": "experiment_contract_invalid",
            }
        gate_names.append("experiment_contract")
    gates: dict[str, dict[str, Any]] = {}
    numeric_substitutions: list[str] = []
    for name in gate_names:
        gate, numeric = _normalise_gate(name, evidence_by_name.get(name))
        gates[name] = gate
        numeric_substitutions.extend(numeric)
    blocking = [name for name, gate in gates.items() if gate["status"] != "pass"]
    training_allowed = not blocking
    result = {
        "status": "pass" if training_allowed else "blocked",
        "training_allowed": training_allowed,
        "gates": gates,
        "automated_checks": {name: gates[name] for name in AUTOMATED_GATE_NAMES},
        "blocking_gates": blocking,
        "numeric_substitutions": sorted(set(numeric_substitutions)),
        "protocol": {
            "round_count": 6,
            "max_attempts_per_round": 5,
            "attempt_train_cases": 10,
            "attempt_dev_cases": 10,
            "overall_train_cases": 60,
            "overall_dev_cases": 60,
            "maximum_real_video_executions": 1320,
        },
        "summary": [
            f"{name}: {gates[name]['status']}"
            + (f" ({gates[name]['reason']})" if gates[name].get("reason") else "")
            for name in gate_names
        ],
    }
    if formal_release_verification is not None:
        result["formal_release"] = formal_release_verification
        result["gate_report_hashes"] = formal_release_verification.get("gate_report_hashes", {})
    if experiment_contract is not None and gates["experiment_contract"]["status"] == "pass":
        result["experiment_contract"] = {
            "contract_version": gates["experiment_contract"]["contract_version"],
            "experiment_id": gates["experiment_contract"]["experiment_id"],
            "experiment_fingerprint": gates["experiment_contract"]["experiment_fingerprint"],
        }
    return result


def _read_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    if source.suffix.lower() == ".xml":
        try:
            xml_root = ET.parse(source).getroot()
            suites = [xml_root] if xml_root.tag == "testsuite" else list(xml_root.findall("testsuite"))
            failures = sum(int(item.attrib.get("failures", 0)) for item in suites)
            errors = sum(int(item.attrib.get("errors", 0)) for item in suites)
            skipped = sum(int(item.attrib.get("skipped", 0)) for item in suites)
            tests = sum(int(item.attrib.get("tests", 0)) for item in suites)
        except (OSError, ValueError, ET.ParseError):
            return {"status": "fail", "reason": "report_unreadable", "path": str(source.resolve())}
        return {
            "status": "pass" if failures == 0 and errors == 0 else "fail",
            "tests": tests,
            "failures": failures,
            "errors": errors,
            "skipped": skipped,
            "path": str(source.resolve()),
        }
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {"status": "fail", "reason": "report_unreadable", "path": str(source.resolve())}
    if not isinstance(value, dict):
        return {"status": "fail", "reason": "report_not_an_object", "path": str(source.resolve())}
    # A sealed release report is content-addressed.  Adding a convenience
    # path field would invalidate its hash and make a valid report unusable.
    if "report_hash" not in value:
        value.setdefault("path", str(source.resolve()))
    return value


def _report_or_pending(path: str | Path | None, reason: str) -> dict[str, Any]:
    report = _read_report(path)
    return report if report is not None else {"status": "pending", "reason": reason, "path": str(path) if path else None}


def _dataset_evidence(root: str | Path, references: list[str | Path] | None = None) -> dict[str, Any]:
    destination = Path(root)
    metadata_path = destination / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "fail",
            "reason": f"training_dataset_must_be_benchmark_prompt_index:metadata_unreadable:{type(exc).__name__}",
            "path": str(destination.resolve()),
        }
    if not isinstance(metadata, Mapping) or metadata.get("dataset_kind") != "benchmark_prompt_index":
        return {
            "status": "fail",
            "reason": "training_dataset_must_be_benchmark_prompt_index; self-built prompt datasets are ineligible",
            "dataset_kind": metadata.get("dataset_kind") if isinstance(metadata, Mapping) else None,
            "path": str(destination.resolve()),
        }
    try:
        from scripts.validate_benchmark_prompt_index import validate_benchmark_prompt_index

        destination_resolved = destination.resolve()
        reference_paths = [
            Path(reference).resolve()
            for reference in (references or [])
            if Path(reference).resolve() != destination_resolved
        ]
        return validate_benchmark_prompt_index(destination, reference_roots=reference_paths)
    except Exception as exc:
        return {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}


def _frozen_evidence(root: str | Path, references: list[str | Path]) -> dict[str, Any]:
    try:
        from scripts.validate_frozen_eval_set import validate_frozen_eval_set

        return validate_frozen_eval_set(root, reference_roots=references)
    except Exception as exc:
        return {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}


def _smoke_evidence(root: str | Path | None, project_root: Path) -> dict[str, Any]:
    if root is None:
        return {"status": "pending", "reason": "agent_smoke_root_not_supplied"}
    smoke_root = Path(root)
    if not smoke_root.exists():
        return {"status": "pending", "reason": "agent_smoke_root_missing", "path": str(smoke_root)}
    job_index_path = smoke_root / "job_index.json"
    if not job_index_path.is_file():
        return {"status": "pending", "reason": "smoke_job_index_missing", "path": str(smoke_root.resolve())}
    try:
        index = json.loads(job_index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "fail", "reason": f"smoke_job_index_invalid:{type(exc).__name__}"}
    generation_mode = index.get("generation_mode")
    jobs = index.get("jobs") or []
    if generation_mode != "agent":
        return {
            "status": "pass" if index.get("prepared_count", 0) else "fail",
            "generation_mode": generation_mode,
            "artifact_status": "unknown",
            "case_count": len(jobs),
            "path": str(smoke_root.resolve()),
        }
    if not jobs or any(job.get("status") != "prepared" for job in jobs):
        return {
            "status": "fail",
            "generation_mode": "agent",
            "reason": "agent_smoke_has_failed_jobs",
            "case_count": len(jobs),
            "path": str(smoke_root.resolve()),
        }
    try:
        from videoact.real_artifacts import RealArtifactGate

        artifact_reports = []
        for job in jobs:
            run_dir = Path(str(job.get("run_dir") or ""))
            if not run_dir.is_absolute():
                run_dir = project_root / run_dir
            artifact_reports.append(RealArtifactGate(minimum_readable_frames=3).validate(run_dir))
        complete = all(report.artifact_status == "complete" for report in artifact_reports)
    except Exception as exc:
        return {"status": "fail", "generation_mode": "agent", "reason": f"artifact_gate_error:{type(exc).__name__}"}
    return {
        "status": "pass" if complete else "fail",
        "generation_mode": "agent",
        "artifact_status": "complete" if complete else "incomplete",
        "case_count": len(jobs),
        "path": str(smoke_root.resolve()),
    }


def _provider_evidence(
    root: str | Path | None,
    *,
    formal_evaluator_config: str | Path | None = None,
) -> dict[str, Any]:
    if root is None:
        return {"status": "pending", "reason": "dynamic_provider_report_not_supplied"}
    provider_root = Path(root)
    if not provider_root.exists():
        return {"status": "pending", "reason": "dynamic_provider_report_missing", "path": str(provider_root)}
    report = _read_report(provider_root) if provider_root.is_file() else None
    if report is not None:
        return report
    job_index = provider_root / "job_index.json"
    if job_index.is_file():
        try:
            index = json.loads(job_index.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {"status": "fail", "reason": "dynamic_provider_job_index_invalid"}
        jobs = index.get("jobs") or []
        prepared = bool(jobs) and all(job.get("status") == "prepared" for job in jobs)
        evidence: dict[str, Any] = {
            "status": "pass" if prepared and index.get("generation_mode") == "agent" else "fail",
            "director": "pass" if prepared else "fail",
            "blender_code": "pass" if prepared else "fail",
            "generation_mode": index.get("generation_mode"),
            "provider_mode": index.get("provider_mode"),
            "case_count": len(jobs),
            "path": str(provider_root.resolve()),
        }
        if index.get("provider_mode") not in {"model", "glm"}:
            return evidence

        try:
            from scripts.train_real_harness import audit_dynamic_agent_index

            audit = audit_dynamic_agent_index(
                index,
                run_root=provider_root,
                expected_case_ids=[str(job.get("case_id")) for job in jobs],
            )
        except Exception as exc:
            evidence.update({"status": "fail", "reason": f"provider_audit_error:{type(exc).__name__}"})
            return evidence
        evidence["provider_audit"] = audit
        if audit.get("status") != "pass":
            evidence.update({"status": "fail", "reason": "provider_manifest_audit_failed"})
            return evidence

        generation_identities: set[tuple[str, str, str, str]] = set()
        manifest_errors: list[str] = []
        for job in jobs:
            manifest_path = Path(
                str(job.get("provider_manifest_path") or provider_root / str(job.get("case_id")) / "provider_manifest.json")
            )
            if not manifest_path.is_absolute():
                manifest_path = provider_root / manifest_path
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                director_stage = payload["stages"]["director"]
                codegen_stage = payload["stages"]["blender_code"]
                generation_identities.add(
                    (
                        str(director_stage["provider_kind"]),
                        str(director_stage["model_id"]),
                        str(codegen_stage["provider_kind"]),
                        str(codegen_stage["model_id"]),
                    )
                )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                manifest_errors.append(f"{job.get('case_id')}:model_identity_unreadable:{type(exc).__name__}")
        if manifest_errors:
            evidence.update(
                {
                    "status": "fail",
                    "reason": "provider_model_identity_unreadable",
                    "failures": manifest_errors,
                }
            )
            return evidence
        if len(generation_identities) != 1:
            evidence.update(
                {
                    "status": "fail",
                    "reason": "generator_boundary_identity_not_constant",
                    "generation_identities": [list(item) for item in sorted(generation_identities)],
                }
            )
            return evidence
        observed_director_provider, observed_director_id, observed_codegen_provider, observed_codegen_id = next(
            iter(generation_identities)
        )
        observed_generator_id = (
            f"{observed_director_provider}:{observed_director_id}|"
            f"{observed_codegen_provider}:{observed_codegen_id}"
        )
        evidence.update(
            {
                "observed_generator_model_id": observed_generator_id,
                "observed_director_provider_kind": observed_director_provider,
                "observed_director_model_id": observed_director_id,
                "observed_codegen_provider_kind": observed_codegen_provider,
                "observed_codegen_model_id": observed_codegen_id,
            }
        )

        config_path = Path(formal_evaluator_config) if formal_evaluator_config is not None else provider_root / "formal-evaluator-v1.json"
        if not config_path.is_absolute():
            config_path = provider_root / config_path
        if not config_path.is_file():
            evidence.update(
                {
                    "status": "fail",
                    "reason": "formal_evaluator_config_missing",
                    "formal_evaluator_config": str(config_path.resolve()),
                }
            )
            return evidence
        try:
            from evaluator.formal_config import FormalEvaluatorConfig

            formal = FormalEvaluatorConfig.from_path(config_path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            evidence.update(
                {
                    "status": "fail",
                    "reason": f"formal_evaluator_config_invalid:{type(exc).__name__}",
                    "formal_evaluator_config": str(config_path.resolve()),
                }
            )
            return evidence
        evidence.update(
            {
                "generator_model_id": formal.generator_model_id,
                "director_model_id": formal.director_model_id,
                "codegen_model_id": formal.codegen_model_id,
                "director_provider_kind": formal.director_provider_kind,
                "codegen_provider_kind": formal.codegen_provider_kind,
                "primary_judge_model_id": formal.primary_judge_model_id,
                "audit_judge_model_id": formal.audit_judge_model_id,
                "formal_evaluator_config": str(config_path.resolve()),
                "formal_evaluator_config_fingerprint": formal.fingerprint(),
            }
        )
        mismatches = []
        if formal.generator_model_id != observed_generator_id:
            mismatches.append("generator_model_id")
        if formal.director_model_id != observed_director_id:
            mismatches.append("director_model_id")
        if formal.codegen_model_id != observed_codegen_id:
            mismatches.append("codegen_model_id")
        if formal.director_provider_kind != observed_director_provider:
            mismatches.append("director_provider_kind")
        if formal.codegen_provider_kind != observed_codegen_provider:
            mismatches.append("codegen_provider_kind")
        if mismatches:
            evidence.update(
                {
                    "status": "fail",
                    "reason": "formal_generation_identity_mismatch",
                    "mismatched_fields": mismatches,
                }
            )
        return evidence
    failure_files = list(provider_root.rglob("director_failure.json")) + list(provider_root.rglob("codegen_failure.json"))
    if failure_files:
        return {"status": "fail", "reason": "dynamic_provider_failure_artifact", "path": str(failure_files[0].resolve())}
    return {"status": "pending", "reason": "dynamic_provider_evidence_unrecognized", "path": str(provider_root.resolve())}


def build_training_readiness_from_project(
    *,
    project_root: str | Path = ".",
    full_test_report: str | Path | None = None,
    capability_report: str | Path | None = None,
    dataset_root: str | Path = "dataset/vbench2-agent-training-index-v1",
    frozen_root: str | Path = "dataset/frozen-eval-v2",
    frozen_reference_roots: list[str | Path] | None = None,
    blender_smoke_root: str | Path | None = None,
    golden_root: str | Path | None = "dataset/golden-review-exact-v2",
    dynamic_provider_root: str | Path | None = None,
    formal_evaluator_config: str | Path | None = "config/formal-evaluator-v1.json",
    paired_gate_report: str | Path | None = None,
    formal_release_report: str | Path | None = None,
    experiment_contract: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    def resolve_input(value: str | Path | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        return path if path.is_absolute() else root / path

    capability_path = resolve_input(capability_report) or root / "out" / "skill_capability_report.json"
    resolved_full_test = resolve_input(full_test_report)
    resolved_dataset = resolve_input(dataset_root)
    resolved_frozen = resolve_input(frozen_root)
    golden_path = resolve_input(golden_root)
    resolved_provider = resolve_input(dynamic_provider_root)
    resolved_formal_config = resolve_input(formal_evaluator_config)
    resolved_paired = resolve_input(paired_gate_report)
    resolved_release = resolve_input(formal_release_report)
    resolved_contract = resolve_input(experiment_contract)
    resolved_references = [path for value in (frozen_reference_roots or []) if (path := resolve_input(value)) is not None]
    golden = (
        _report_or_pending(golden_path, "golden_review_bundle_missing")
        if golden_path is None or not golden_path.exists()
        else None
    )
    if golden is None:
        try:
            from scripts.validate_golden_review_set import validate_golden_review_set

            golden = validate_golden_review_set(golden_path)
            errors = golden.get("errors", []) if isinstance(golden, dict) else []
            pending_golden_errors = all(
                any(
                    marker in str(error)
                    for marker in (
                        "human scores",
                        "independent annotators",
                        "inter-rater agreement",
                        "render prompt differs",
                    )
                )
                for error in errors
            )
            if golden.get("status") == "fail" and errors and pending_golden_errors:
                golden = {
                    **golden,
                    "status": "pending",
                    "reason": (
                        "golden_bundle_requires_exact_prompt_rerender"
                        if any("render prompt differs" in str(error) for error in errors)
                        else "golden_annotations_pending"
                    ),
                }
        except Exception as exc:
            golden = {"status": "fail", "reason": f"{type(exc).__name__}: {exc}"}
    frozen_refs = frozen_reference_roots or []
    readiness_kwargs: dict[str, Any] = {
        "automated_checks": {
            "full_test": _report_or_pending(resolved_full_test, "full_test_report_missing"),
            "capability": _report_or_pending(capability_path, "capability_report_missing"),
            "dataset": _dataset_evidence(
                resolved_dataset or root / "dataset" / "vbench2-agent-training-index-v1",
                resolved_references,
            ),
            "frozen_eval": _frozen_evidence(resolved_frozen or root / "dataset" / "frozen-eval-v2", resolved_references),
        },
        "real_blender_smoke": _smoke_evidence(resolve_input(blender_smoke_root), root),
        "golden_review": golden,
        "dynamic_agent_provider": _provider_evidence(
            resolved_provider,
            formal_evaluator_config=resolved_formal_config,
        ),
        "paired_gate": _report_or_pending(resolved_paired, "paired_gate_report_missing"),
    }
    if resolved_release is not None:
        readiness_kwargs["formal_release_report"] = _report_or_pending(
            resolved_release, "formal_release_report_missing"
        )
    if resolved_contract is not None:
        try:
            from videoact.real_artifacts import load_experiment_contract

            readiness_kwargs["experiment_contract"] = load_experiment_contract(resolved_contract).model_dump(mode="json")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            readiness_kwargs["experiment_contract"] = {"invalid": True}
    report = build_training_readiness(**readiness_kwargs)
    report["project_root"] = str(root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--full-test-report")
    parser.add_argument("--capability-report")
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--frozen-root", default="dataset/frozen-eval-v2")
    parser.add_argument("--frozen-reference-root", action="append", default=[])
    parser.add_argument("--blender-smoke-root")
    parser.add_argument("--golden-root", default="dataset/golden-review-exact-v2")
    parser.add_argument("--dynamic-provider-root")
    parser.add_argument("--formal-evaluator-config", default="config/formal-evaluator-v1.json")
    parser.add_argument("--paired-gate-report")
    parser.add_argument("--formal-release-report")
    parser.add_argument("--experiment-contract")
    parser.add_argument("--out")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    report = build_training_readiness_from_project(
        project_root=project_root,
        full_test_report=args.full_test_report,
        capability_report=args.capability_report,
        dataset_root=project_root / args.dataset_root,
        frozen_root=project_root / args.frozen_root,
        frozen_reference_roots=[project_root / value for value in args.frozen_reference_root],
        blender_smoke_root=project_root / args.blender_smoke_root if args.blender_smoke_root else None,
        golden_root=project_root / args.golden_root if args.golden_root else None,
        dynamic_provider_root=project_root / args.dynamic_provider_root if args.dynamic_provider_root else None,
        formal_evaluator_config=project_root / args.formal_evaluator_config if args.formal_evaluator_config else None,
        paired_gate_report=project_root / args.paired_gate_report if args.paired_gate_report else None,
        formal_release_report=project_root / args.formal_release_report if args.formal_release_report else None,
        experiment_contract=project_root / args.experiment_contract if args.experiment_contract else None,
    )
    destination = Path(args.out) if args.out else project_root / "out" / "training_readiness_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
