"""Validate real reviewed plan/source pairs for BlenderCodeAgent context."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from videoact.real_artifacts import RealArtifactGate
from videoact.run_manifest import hash_payload


ELIGIBLE_REVIEW_SOURCES = {
    "human_review",
    "codex_local_visual_review",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _record_root(record: dict[str, Any], examples_root: Path) -> Path | None:
    raw = record.get("artifact_root") or record.get("artifact_path")
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else (examples_root / path)


def _validate_record(record: dict[str, Any], examples_root: Path) -> list[str]:
    case_id = str(record.get("case_id") or "unknown-case")
    errors: list[str] = []
    if record.get("generation_mode") != "agent":
        errors.append(f"{case_id}:generation_mode must be agent")
    if len(str(record.get("plan_hash") or "")) != 64:
        errors.append(f"{case_id}:plan_hash missing or invalid")
    if len(str(record.get("code_hash") or "")) != 64:
        errors.append(f"{case_id}:code_hash missing or invalid")
    review_source = str(record.get("review_source") or "")
    model = str(record.get("model") or "")
    if review_source not in ELIGIBLE_REVIEW_SOURCES and not (
        review_source == "external_vlm" and model in {"gpt-5.6-luna", "gpt-5.6-terra"}
    ):
        errors.append(f"{case_id}:review_source is not an eligible independent review")
    confidence = record.get("review_confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"{case_id}:review_confidence missing or invalid")
    try:
        deterministic_score = float(record["deterministic_score"])
    except (KeyError, TypeError, ValueError):
        deterministic_score = -1.0
    if deterministic_score < 80.0:
        errors.append(f"{case_id}:deterministic_score must be at least 80")

    run_root = _record_root(record, examples_root)
    if run_root is None or not run_root.is_dir():
        errors.append(f"{case_id}:artifact_root missing")
        return errors
    report = RealArtifactGate(minimum_readable_frames=3).validate(run_root)
    if record.get("artifact_status") != "complete" or report.artifact_status != "complete":
        details = ",".join(report.hard_failures) or "record_artifact_status_not_complete"
        errors.append(f"{case_id}:real artifact gate failed:{details}")
    source = run_root / "blender_job.py"
    if not source.is_file():
        errors.append(f"{case_id}:blender_job.py missing")
    else:
        actual_code_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_code_hash != str(record.get("code_hash")):
            errors.append(f"{case_id}:code_hash mismatch")
    plan_path = run_root / "director_plan.json"
    if not plan_path.is_file():
        errors.append(f"{case_id}:director_plan.json missing")
    else:
        try:
            director_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            actual_plan_hash = hash_payload(director_plan)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{case_id}:director_plan.json invalid:{type(exc).__name__}")
        else:
            if actual_plan_hash != str(record.get("plan_hash")):
                errors.append(f"{case_id}:plan_hash mismatch")
    return errors


def validate_codegen_examples(root: str | Path = "dataset/codegen-examples-v1") -> dict[str, Any]:
    examples_root = Path(root)
    manifest = examples_root / "manifest.jsonl"
    if not manifest.is_file():
        return {
            "status": "pending_external",
            "case_count": 0,
            "eligible_count": 0,
            "errors": [f"examples_manifest_missing:{manifest}"],
            "records": [],
        }
    records = _read_jsonl(manifest)
    if not records:
        return {
            "status": "pending_external",
            "case_count": 0,
            "eligible_count": 0,
            "errors": ["no_real_reviewed_examples"],
            "records": [],
        }
    errors: list[str] = []
    record_reports: list[dict[str, Any]] = []
    for record in records:
        record_errors = _validate_record(record, examples_root)
        case_id = str(record.get("case_id") or "unknown-case")
        record_reports.append({"case_id": case_id, "status": "pass" if not record_errors else "fail", "errors": record_errors})
        errors.extend(record_errors)
    return {
        "status": "pass" if not errors else "fail",
        "case_count": len(records),
        "eligible_count": len(records) - sum(report["status"] == "fail" for report in record_reports),
        "errors": errors,
        "records": record_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="dataset/codegen-examples-v1")
    args = parser.parse_args()
    report = validate_codegen_examples(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
