"""Normalize historical outer-loop evidence into proposal-tool JSONL records.

The converter is intentionally read-only with respect to the Harness, dataset,
evaluator, and skill. It turns accepted round patch manifests plus their new
train batch into auditable records for ``propose_skill_update.py``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DIRECTOR_OWNERS = {
    "director_prompt_interpreter",
    "director_event_scheduler",
    "director_trajectory",
    "director_camera",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_number(round_dir: Path) -> int:
    match = re.fullmatch(r"round-(\d+)", round_dir.name)
    if not match:
        raise ValueError(f"expected round-N directory, got {round_dir}")
    return int(match.group(1))


def _train_case_ids(round_dir: Path) -> list[str]:
    attempt_report = round_dir / "attempt_report.json"
    if attempt_report.is_file():
        batch = _load_json(attempt_report).get("batch", {})
        case_ids = [str(case_id) for case_id in batch.get("train", [])]
        if case_ids:
            return case_ids

    discovered = {
        path.parent.name
        for path in round_dir.glob("attempt-*/real/train/*/combined_evaluator.json")
    }
    return sorted(discovered)


def _evidence_paths(round_dir: Path, case_id: str) -> list[str]:
    paths = [round_dir / "patch_manifest.json"]
    attempt_report = round_dir / "attempt_report.json"
    if attempt_report.is_file():
        paths.append(attempt_report)
    case_reports = sorted(round_dir.glob(f"attempt-*/real/train/{case_id}/combined_evaluator.json"))
    paths.extend(case_reports)
    return [str(path.resolve()) for path in paths if path.is_file()]


def _normalize_finding(patch: dict[str, Any]) -> dict[str, str]:
    owner = str(patch.get("owner", "unknown"))
    if owner in {"director_prompt_interpreter", "director_event_scheduler"}:
        return {
            "failure_id": "implicit_event_order_not_preserved",
            "category": "event_order",
            "severity": "error",
        }
    return {
        "failure_id": str(patch.get("patch_id", "historical_patch")),
        "category": "historical_patch_evidence",
        "severity": "warning",
    }


def build_historical_records(round_root: str | Path) -> list[dict[str, Any]]:
    root = Path(round_root)
    records: list[dict[str, Any]] = []
    for round_dir in sorted(root.glob("round-*"), key=lambda path: _round_number(path)):
        patch_path = round_dir / "patch_manifest.json"
        if not patch_path.is_file():
            continue
        patch = _load_json(patch_path)
        owner = str(patch.get("owner", "unknown"))
        decision = str(patch.get("decision", ""))
        if decision != "accepted" or owner == "pretraining_evaluator_gate":
            continue

        normalized = _normalize_finding(patch)
        message = str(patch.get("detected_problem", "historical accepted patch"))
        if owner in {"director_prompt_interpreter", "director_event_scheduler"}:
            message = "implicit event order was not preserved across reveal, handoff, pause, or return clauses"
        for case_id in _train_case_ids(round_dir):
            evidence = _evidence_paths(round_dir, case_id)
            records.append(
                {
                    "case_id": case_id,
                    "round": _round_number(round_dir),
                    "record_type": "historical_patch_scope",
                    "source": "prior_real_training_artifacts",
                    "status": "fail",
                    "findings": [
                        {
                            **normalized,
                            "owner": owner,
                            "message": message,
                            "evidence": evidence,
                            "repair_route": str(patch.get("fix_location", "reviewed_patch")),
                        }
                    ],
                    "patch_id": str(patch.get("patch_id", "")),
                    "scope_note": "Case IDs are the round's new train batch; patch_manifest.json is the source of diagnosis.",
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = build_historical_records(args.round_root)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"record_count": len(records), "out": str(destination.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
