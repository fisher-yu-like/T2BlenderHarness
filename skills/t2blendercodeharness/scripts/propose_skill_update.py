"""Create reviewed, proposal-only updates from repeated Harness failures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


OWNER_SECTIONS = {
    "scene_parser": "contract handoff and scene parsing",
    "trajectory_planner": "trajectory handoff and continuity",
    "camera_planner": "camera coverage and observability",
    "director_prompt_interpreter": "Director prompt interpretation and event scheduling",
    "director_event_scheduler": "Director prompt interpretation and event scheduling",
    "director_trajectory": "Director trajectories and interaction lifecycles",
    "director_camera": "Director camera choreography and visibility",
    "blender_code_agent": "Blender code generation from DirectorPlan",
    "blender_executor": "controlled execution and artifact persistence",
    "proxy_renderer": "artifact gate and proxy rendering",
    "evaluator": "deterministic evaluation and score promotion",
    "meta_harness": "outer-loop aggregation and acceptance",
}


SKILL_VERSION = "t2blendercodeharness-v5-executable-director"
ALLOWED_CHANGE_TYPES = {"function_library", "owner_mapping", "memory_entry", "prose_guidance", "runtime_patch"}


def _skill_hash(skill_path: Path) -> str:
    return hashlib.sha256(skill_path.read_bytes()).hexdigest()


def _load_records(records_path: Path) -> list[dict[str, Any]]:
    records = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_update_proposal(
    records_path: str | Path,
    skill_path: str | Path,
    *,
    minimum_cases: int = 2,
    change_type: str = "runtime_patch",
    explicit_reason: str | None = None,
) -> dict[str, Any]:
    if change_type not in ALLOWED_CHANGE_TYPES:
        raise ValueError(f"unsupported change_type: {change_type}")
    records_file = Path(records_path)
    skill_file = Path(skill_path)
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"case_ids": set(), "messages": [], "evidence": set()}
    )
    ignored_vlm_unavailable = 0
    records = _load_records(records_file)
    for record in records:
        if record.get("status") in {"unavailable", "skipped"} or record.get("source") == "vlm":
            ignored_vlm_unavailable += 1
            continue
        for finding in record.get("findings", []):
            owner = str(finding.get("owner", "unknown"))
            key = (
                str(finding.get("failure_id", "unknown")),
                owner,
                str(finding.get("category", "unknown")),
                str(finding.get("severity", "unknown")),
            )
            group = groups[key]
            group["case_ids"].add(str(record.get("case_id", "unknown")))
            group["messages"].append(str(finding.get("message", "")))
            group["evidence"].update(str(item) for item in finding.get("evidence", []))

    proposals = []
    for (failure_id, owner, category, severity), group in sorted(
        groups.items(), key=lambda item: (-len(item[1]["case_ids"]), item[0])
    ):
        if len(group["case_ids"]) < minimum_cases:
            continue
        proposal = {
                "owner": owner,
                "failure_id": failure_id,
                "category": category,
                "severity": severity,
                "affected_case_ids": sorted(group["case_ids"]),
                "evidence_paths": sorted(group["evidence"]),
                "observed_pattern": group["messages"][0] if group["messages"] else failure_id,
                "target_section": OWNER_SECTIONS.get(owner, f"owner={owner}"),
                "required_checks": [
                    "run capability_check.py",
                    "run the full project test suite",
                    "rerun train and dev with stable fingerprints",
                    "keep test split frozen until final blind verification",
                ],
        }
        proposal["change_type"] = change_type
        proposal["runtime_change"] = change_type != "prose_guidance"
        proposal["requires_explicit_reason"] = change_type == "prose_guidance"
        if change_type == "prose_guidance":
            proposal["warning"] = (
                "prose_guidance is not counted as Harness training; provide an explicit reason "
                "and independent runtime evidence before applying it"
            )
            if explicit_reason:
                proposal["explicit_reason"] = explicit_reason
        proposals.append(proposal)

    return {
        "skill_version": SKILL_VERSION,
        "status": "proposal_ready" if proposals else "no_action",
        "requires_human_review": True,
        "skill_sha256": _skill_hash(skill_file),
        "records_path": str(records_file.resolve()),
        "record_count": len(records),
        "ignored_vlm_unavailable": ignored_vlm_unavailable,
        "proposals": proposals,
        "mutation_policy": "proposal-only; never edit SKILL.md, source, evaluator, or labels",
        "change_type": change_type,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--skill-file", default=str(Path(__file__).resolve().parents[1] / "SKILL.md"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-cases", type=int, default=2)
    parser.add_argument("--change-type", choices=sorted(ALLOWED_CHANGE_TYPES), default="runtime_patch")
    parser.add_argument("--explicit-reason")
    args = parser.parse_args()
    result = build_update_proposal(
        args.records,
        args.skill_file,
        minimum_cases=args.minimum_cases,
        change_type=args.change_type,
        explicit_reason=args.explicit_reason,
    )
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
