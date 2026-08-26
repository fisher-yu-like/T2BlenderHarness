"""Pair baseline and candidate on the old evaluator and an independent realism score."""

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

from evaluator.findings import score_findings  # noqa: E402
from evaluator.realism import score_realism  # noqa: E402
from videoact.contracts import Finding  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def legacy_score(report: dict[str, Any]) -> float:
    """Recover the old score while ignoring the new independent geometry category."""
    findings = [
        Finding.model_validate(finding)
        for finding in report.get("findings", [])
        if finding.get("category") != "geometry_realism"
    ]
    return score_findings(findings)


def pair_evaluate(baseline_root: str | Path, candidate_root: str | Path, output: str | Path) -> dict[str, Any]:
    baseline_root, candidate_root = Path(baseline_root), Path(candidate_root)
    baseline_geom = _load(baseline_root / "geometry_audit_summary.json")
    candidate_geom = _load(candidate_root / "geometry_audit_summary.json")
    baseline_by_id = {item["case_id"]: item for item in baseline_geom["results"]}
    candidate_by_id = {item["case_id"]: item for item in candidate_geom["results"]}
    case_ids = sorted(set(baseline_by_id) & set(candidate_by_id))
    rows = []
    for case_id in case_ids:
        baseline_dir = baseline_root / case_id
        candidate_dir = candidate_root / case_id
        baseline_det = _load(baseline_dir / "deterministic_report.json")
        candidate_det = _load(candidate_dir / "deterministic_report.json")
        baseline_realism = score_realism(baseline_by_id[case_id], baseline_by_id[case_id].get("visual_evidence"))
        candidate_realism = score_realism(candidate_by_id[case_id], candidate_by_id[case_id].get("visual_evidence"))
        baseline_legacy = legacy_score(baseline_det)
        candidate_legacy = legacy_score(candidate_det)
        candidate_cli = _load(candidate_root / "cli_render_report.json")
        cli_by_id = {item["case_id"]: item for item in candidate_cli.get("results", [])}
        render_ok = cli_by_id.get(case_id, {}).get("status") == "success"
        row = {
            "case_id": case_id,
            "baseline_video": str((baseline_dir / "proxy.mp4").resolve()),
            "candidate_video": str((candidate_dir / "proxy.mp4").resolve()),
            "baseline_legacy_score": baseline_legacy,
            "candidate_legacy_score": candidate_legacy,
            "legacy_delta": round(candidate_legacy - baseline_legacy, 4),
            "baseline_realism_score": baseline_realism["score"],
            "candidate_realism_score": candidate_realism["score"],
            "realism_delta": round(candidate_realism["score"] - baseline_realism["score"], 4),
            "baseline_realism_band": baseline_realism["band"],
            "candidate_realism_band": candidate_realism["band"],
            "baseline_realism_score_kind": baseline_realism["score_kind"],
            "candidate_realism_score_kind": candidate_realism["score_kind"],
            "candidate_requires_independent_review": candidate_realism["requires_independent_review"],
            "baseline_geometry_hard_fail": bool(baseline_by_id[case_id].get("hard_gate_failed")),
            "candidate_geometry_hard_fail": bool(candidate_by_id[case_id].get("hard_gate_failed")),
            "candidate_render_ok": render_ok,
            "candidate_deterministic_status": candidate_det.get("terminal_status"),
            "candidate_deterministic_findings": [finding["failure_id"] for finding in candidate_det.get("findings", [])],
        }
        (candidate_dir / "realism_report.json").write_text(json.dumps(candidate_realism, indent=2, sort_keys=True), encoding="utf-8")
        rows.append(row)

    legacy_before = sum(row["baseline_legacy_score"] for row in rows) / len(rows) if rows else 0.0
    legacy_after = sum(row["candidate_legacy_score"] for row in rows) / len(rows) if rows else 0.0
    realism_before = sum(row["baseline_realism_score"] for row in rows) / len(rows) if rows else 0.0
    realism_after = sum(row["candidate_realism_score"] for row in rows) / len(rows) if rows else 0.0
    acceptance = {
        "same_case_count": len(rows) == len(baseline_by_id) == len(candidate_by_id),
        "all_candidate_renders_ok": all(row["candidate_render_ok"] for row in rows),
        "legacy_per_case_non_regression": all(row["legacy_delta"] >= 0 for row in rows),
        "legacy_mean_non_regression": legacy_after >= legacy_before,
        "realism_mean_improved": realism_after > realism_before,
        "candidate_geometry_hard_fail_count": sum(row["candidate_geometry_hard_fail"] for row in rows),
    }
    acceptance["accepted"] = all([
        acceptance["same_case_count"],
        acceptance["all_candidate_renders_ok"],
        acceptance["legacy_per_case_non_regression"],
        acceptance["legacy_mean_non_regression"],
        acceptance["realism_mean_improved"],
        acceptance["candidate_geometry_hard_fail_count"] == 0,
    ])
    report = {
        "evolution_version": "realism-evolution-v3",
        "baseline_root": str(baseline_root.resolve()),
        "candidate_root": str(candidate_root.resolve()),
        "old_evaluator": "deterministic-v3-declarative-independent-oracle",
        "independent_realism_evaluator": "realism-v3-independent-artifact-review",
        "weights": {"legacy_gate": "non-regression only", "artifact_only_proxy": "geometry .60 + sampled PNG evidence .40, ceiling 80; independent review fusion .20/.80 when available"},
        "aggregate": {
            "legacy_before": round(legacy_before, 4),
            "legacy_after": round(legacy_after, 4),
            "legacy_delta": round(legacy_after - legacy_before, 4),
            "realism_before": round(realism_before, 4),
            "realism_after": round(realism_after, 4),
            "realism_delta": round(realism_after - realism_before, 4),
        },
        "acceptance": acceptance,
        "cases": rows,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = pair_evaluate(args.baseline_root, args.candidate_root, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()), "aggregate": report["aggregate"], "acceptance": report["acceptance"]}, indent=2))
    return 0 if report["acceptance"]["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
