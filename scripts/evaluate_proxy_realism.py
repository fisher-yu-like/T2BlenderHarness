"""Audit actual Blender geometry and merge it into deterministic reports."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator.findings import deduplicate_findings, score_findings  # noqa: E402
from evaluator.geometry_realism import AUDIT_VERSION, evaluate_geometry_report  # noqa: E402
from evaluator.realism import score_realism  # noqa: E402
from evaluator.visual_evidence import inspect_render_frames  # noqa: E402
from videoact.contracts import Finding  # noqa: E402


def discover_run_dirs(root: str | Path) -> list[Path]:
    return sorted(path for path in Path(root).iterdir() if path.is_dir() and (path / "proxy.blend").is_file())


def _inspect(blender_bin: str, run_dir: Path, timeout_s: int) -> dict[str, Any]:
    output = run_dir / "geometry_raw.json"
    script = ROOT / "scripts" / "inspect_blend_geometry.py"
    command = [blender_bin, "-b", str(run_dir / "proxy.blend"), "--python", str(script), "--", "--output", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    if completed.returncode != 0 or not output.is_file():
        return {
            "audit_available": False,
            "error": f"Blender geometry inspection failed with return code {completed.returncode}",
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "meshes": [],
        }
    return json.loads(output.read_text(encoding="utf-8"))


def audit_run(run_dir: str | Path, *, blender_bin: str, merge_deterministic: bool = False, timeout_s: int = 180) -> dict[str, Any]:
    root = Path(run_dir)
    proxy_scene = json.loads((root / "proxy_scene.json").read_text(encoding="utf-8")) if (root / "proxy_scene.json").is_file() else {}
    raw = _inspect(blender_bin, root, timeout_s)
    report = evaluate_geometry_report(raw, proxy_scene)
    visual_report = inspect_render_frames(root)
    realism_report = score_realism(report, visual_report)
    (root / "geometry_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "visual_evidence.json").write_text(json.dumps(visual_report, indent=2, sort_keys=True), encoding="utf-8")
    (root / "realism_report.json").write_text(json.dumps(realism_report, indent=2, sort_keys=True), encoding="utf-8")
    if merge_deterministic and (root / "deterministic_report.json").is_file():
        deterministic = json.loads((root / "deterministic_report.json").read_text(encoding="utf-8"))
        existing = [Finding.model_validate(finding) for finding in deterministic.get("findings", [])]
        geometry_findings = [Finding.model_validate(finding) for finding in report["findings"]]
        all_findings = deduplicate_findings([*existing, *geometry_findings])
        deterministic["findings"] = [finding.model_dump(mode="json") for finding in all_findings]
        deterministic["score"] = score_findings(all_findings)
        deterministic["hard_gate_failed"] = any(finding.severity == "hard" for finding in all_findings)
        deterministic["terminal_status"] = "fail" if deterministic["hard_gate_failed"] else "pass"
        deterministic["evaluator_version"] = "real-v4-geometry-realism-v1"
        deterministic["metrics"] = {
            **deterministic.get("metrics", {}),
            "geometry_score": float(report["score"]),
            "geometry_hard_count": float(sum(finding["severity"] == "hard" for finding in report["findings"])),
            "geometry_mesh_count": float(report["mesh_count"]),
        }
        (root / "deterministic_report.json").write_text(json.dumps(deterministic, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "case_id": json.loads((root / "run_manifest.json").read_text(encoding="utf-8")).get("case_id"),
        **report,
        "visual_evidence": visual_report,
        "realism": realism_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--merge-deterministic", action="store_true")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    results = [audit_run(path, blender_bin=args.blender_bin, merge_deterministic=args.merge_deterministic, timeout_s=args.timeout_s) for path in discover_run_dirs(args.run_root)]
    payload = {"audit_version": AUDIT_VERSION, "run_root": str(Path(args.run_root).resolve()), "results": results}
    output = Path(args.output) if args.output else Path(args.run_root) / "geometry_audit_summary.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "case_count": len(results), "hard_fail_count": sum(result["hard_gate_failed"] for result in results)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
