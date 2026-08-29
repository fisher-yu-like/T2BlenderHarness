"""Evaluate the raw-prompt ablation without importing the Director contracts.

The ordinary real-run evaluator intentionally validates SceneContract and
TrajectoryPlan.  That is correct for the two Director arms, but it would make
the no-Director ablation fail before its video can be compared.  This adapter
therefore owns only runtime/artifact eligibility and leaves semantic and
realism ranking to the shared blind visual-review aggregator.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_proxy_realism import audit_run  # noqa: E402
from videoact.real_artifacts import RealArtifactGate  # noqa: E402


DIRECT_EVALUATOR_VERSION = "three-arm-direct-runtime-v1"


def _json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def _report_dict(report: Any) -> dict[str, Any]:
    if hasattr(report, "model_dump"):
        return report.model_dump(mode="json")
    return dict(report or {})


def build_direct_runtime_report(artifact_report: Any, telemetry: dict[str, Any] | None) -> dict[str, Any]:
    """Create a non-semantic runtime report for the direct-code arm.

    A complete render receives a runtime score of 100 only as an eligibility
    diagnostic.  The three-arm ranking never uses this value as semantic
    quality; its task score is derived from the same blind visual review as the
    Director arms.  Missing artifacts are fail-closed and never imputed.
    """
    report = _report_dict(artifact_report)
    failures = list(report.get("hard_failures") or [])
    complete = report.get("artifact_status") == "complete" and not failures
    findings = [
        {
            "failure_id": "incomplete_direct_runtime_artifacts",
            "owner": "proxy_renderer",
            "category": "runtime",
            "severity": "hard",
            "root_cause_id": "direct_runtime_artifact_completeness",
            "message": f"raw-prompt ablation artifacts are incomplete: {failures}",
            "evidence": failures,
            "repair_route": "runtime_repair",
        }
    ] if not complete else []
    telemetry = telemetry or {}
    object_count = len(telemetry.get("objects") or {})
    camera_active = bool((telemetry.get("camera") or {}).get("active"))
    return {
        "evaluator_version": DIRECT_EVALUATOR_VERSION,
        "planning_mode": "direct_prompt_code",
        "terminal_status": "pass" if complete else "fail",
        "hard_gate_failed": not complete,
        "score": 100.0 if complete else 0.0,
        "score_kind": "runtime_artifact_only",
        "director_plan_score": None,
        "director_findings": [],
        "interaction_findings": [],
        "findings": findings,
        "metrics": {
            "artifact_complete": 1.0 if complete else 0.0,
            "readable_frame_count": float(report.get("readable_frame_count", 0) or 0),
            "video_duration_s": float(report.get("video_duration_s", 0.0) or 0.0),
            "telemetry_object_count": float(object_count),
            "telemetry_active_camera": 1.0 if camera_active else 0.0,
            "hard_count": float(len(findings)),
            "finding_count": float(len(findings)),
        },
    }


def evaluate_direct_run(
    run_dir: str | Path,
    *,
    record: dict[str, Any] | None = None,
    blender_bin: str | Path | None = None,
    audit_geometry: bool = True,
    timeout_s: int = 180,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _json(root / "run_manifest.json", {})
    telemetry = _json(root / "telemetry.json", {})
    artifact = RealArtifactGate().validate(root)
    deterministic = build_direct_runtime_report(artifact, telemetry)
    (root / "deterministic_report.json").write_text(
        json.dumps(deterministic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    realism: dict[str, Any] = {
        "status": "unavailable",
        "reason": "geometry audit not run",
        "score": None,
        "score_kind": "artifact_only_proxy_unavailable",
    }
    blender_path = Path(blender_bin) if blender_bin else None
    if audit_geometry and artifact.artifact_status == "complete" and blender_path and blender_path.is_file():
        try:
            audit = audit_run(root, blender_bin=str(blender_path), timeout_s=timeout_s)
            realism = audit.get("realism") or realism
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            realism["reason"] = f"geometry audit failed: {type(exc).__name__}: {exc}"

    combined = {
        "evaluator_version": DIRECT_EVALUATOR_VERSION,
        "planning_mode": "direct_prompt_code",
        "deterministic_report": str((root / "deterministic_report.json").resolve()),
        "geometry_report": str((root / "geometry_report.json").resolve()) if (root / "geometry_report.json").is_file() else None,
        "visual_evidence": str((root / "visual_evidence.json").resolve()) if (root / "visual_evidence.json").is_file() else None,
        "realism_report": str((root / "realism_report.json").resolve()) if (root / "realism_report.json").is_file() else None,
        "director_plan_score": None,
        "vlm_policy": "one shared blind review per case; no Director contract is required for this arm",
        "scores_are_added": False,
    }
    (root / "combined_evaluator.json").write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "case_id": manifest.get("case_id"),
        "split": manifest.get("split"),
        "run_dir": str(root.resolve()),
        "proxy_video": str((root / "proxy.mp4").resolve()) if (root / "proxy.mp4").is_file() else None,
        "planning_mode": "direct_prompt_code",
        "status": deterministic["terminal_status"],
        "artifact_status": artifact.artifact_status,
        "hard_failures": artifact.hard_failures,
        "deterministic_score": deterministic["score"],
        "deterministic_score_kind": deterministic["score_kind"],
        "director_plan_score": None,
        "realism": realism,
        "harness_issue": "raw prompt was compiled without Director event, trajectory, or camera contracts",
        "prompt": (record or {}).get("prompt"),
    }


def discover_direct_run_dirs(root: str | Path) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_dir() and (path / "run_manifest.json").is_file())


def evaluate_direct_root(
    run_root: str | Path,
    *,
    dataset_root: str | Path | None = None,
    blender_bin: str | Path | None = None,
    workers: int = 12,
    audit_geometry: bool = True,
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if dataset_root:
        manifest_path = Path(dataset_root) / "manifest.jsonl"
        if manifest_path.is_file():
            records = {
                row["case_id"]: row
                for row in (_json_line for _json_line in (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()))
            }
    paths = discover_direct_run_dirs(run_root)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [
            pool.submit(
                evaluate_direct_run,
                path,
                record=records.get((_json(path / "run_manifest.json", {}) or {}).get("case_id")),
                blender_bin=blender_bin,
                audit_geometry=audit_geometry,
            )
            for path in paths
        ]
        results = [future.result() for future in futures]
    results.sort(key=lambda item: str(item.get("case_id")))
    output = Path(run_root) / "direct_evaluation.json"
    output.write_text(
        json.dumps(
            {
                "evaluator_version": DIRECT_EVALUATOR_VERSION,
                "run_root": str(Path(run_root).resolve()),
                "case_count": len(results),
                "complete_count": sum(item["artifact_status"] == "complete" for item in results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--dataset-root")
    parser.add_argument("--blender-bin")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-geometry", action="store_true")
    args = parser.parse_args()
    results = evaluate_direct_root(
        args.run_root,
        dataset_root=args.dataset_root,
        blender_bin=args.blender_bin,
        workers=args.workers,
        audit_geometry=not args.skip_geometry,
    )
    print(json.dumps({"case_count": len(results), "complete_count": sum(item["artifact_status"] == "complete" for item in results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
