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
) -> dict[str, Any]:
    """Combine independent evidence without converting any value to a score."""

    evidence_by_name: dict[str, Any] = {
        **{name: automated_checks.get(name) for name in AUTOMATED_GATE_NAMES},
        "real_blender_smoke": real_blender_smoke,
        "golden_review": golden_review,
        "dynamic_agent_provider": dynamic_agent_provider,
        "paired_gate": paired_gate,
    }
    gates: dict[str, dict[str, Any]] = {}
    numeric_substitutions: list[str] = []
    for name in REQUIRED_GATE_NAMES:
        gate, numeric = _normalise_gate(name, evidence_by_name.get(name))
        gates[name] = gate
        numeric_substitutions.extend(numeric)
    blocking = [name for name, gate in gates.items() if gate["status"] != "pass"]
    training_allowed = not blocking
    return {
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
            for name in REQUIRED_GATE_NAMES
        ],
    }


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


def _provider_evidence(root: str | Path | None) -> dict[str, Any]:
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
        return {
            "status": "pass" if prepared and index.get("generation_mode") == "agent" else "fail",
            "director": "pass" if prepared else "fail",
            "blender_code": "pass" if prepared else "fail",
            "case_count": len(jobs),
            "path": str(provider_root.resolve()),
        }
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
    frozen_root: str | Path = "dataset/frozen-eval-v1",
    frozen_reference_roots: list[str | Path] | None = None,
    blender_smoke_root: str | Path | None = None,
    golden_root: str | Path | None = "dataset/golden-review-exact-v2",
    dynamic_provider_root: str | Path | None = None,
    paired_gate_report: str | Path | None = None,
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
    resolved_paired = resolve_input(paired_gate_report)
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
    report = build_training_readiness(
        automated_checks={
            "full_test": _report_or_pending(resolved_full_test, "full_test_report_missing"),
            "capability": _report_or_pending(capability_path, "capability_report_missing"),
            "dataset": _dataset_evidence(
                resolved_dataset or root / "dataset" / "vbench2-agent-training-index-v1",
                resolved_references,
            ),
            "frozen_eval": _frozen_evidence(resolved_frozen or root / "dataset" / "frozen-eval-v1", resolved_references),
        },
        real_blender_smoke=_smoke_evidence(resolve_input(blender_smoke_root), root),
        golden_review=golden,
        dynamic_agent_provider=_provider_evidence(resolved_provider),
        paired_gate=_report_or_pending(resolved_paired, "paired_gate_report_missing"),
    )
    report["project_root"] = str(root)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--full-test-report")
    parser.add_argument("--capability-report")
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--frozen-root", default="dataset/frozen-eval-v1")
    parser.add_argument("--frozen-reference-root", action="append", default=[])
    parser.add_argument("--blender-smoke-root")
    parser.add_argument("--golden-root", default="dataset/golden-review-exact-v2")
    parser.add_argument("--dynamic-provider-root")
    parser.add_argument("--paired-gate-report")
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
        paired_gate_report=project_root / args.paired_gate_report if args.paired_gate_report else None,
    )
    destination = Path(args.out) if args.out else project_root / "out" / "training_readiness_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["training_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
