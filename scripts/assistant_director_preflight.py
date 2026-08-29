"""Offline Director preflight for assistant-authored interpretations.

Validates that every case's pre-authored interpretation passes the real
Director pipeline (interpret -> schedule -> trajectory -> camera -> gates)
before the expensive real run starts.  Missing preauth is a hard error, not a
template fallback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.assistant_session_provider import AssistantSessionProvider  # noqa: E402
from videoact.director import DirectorAgent  # noqa: E402


def dataset_records(dataset_root: Path, split: str) -> dict[str, dict]:
    records = {}
    for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") == split:
            records[record["case_id"]] = record
    return records


def case_obligations(record: dict) -> dict[str, list[str]]:
    return {
        "required_entity_ids": list(record.get("required_entity_ids") or []),
        "required_event_ids": list(record.get("required_event_ids") or []),
        "required_camera_event_ids": list(record.get("required_camera_event_ids") or []),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--case-ids", required=True, help="comma-separated case ids")
    parser.add_argument("--session-root", default="out/assistant-session")
    parser.add_argument("--out", help="optional report path")
    args = parser.parse_args()

    records = dataset_records(Path(args.dataset_root), args.split)
    wanted = [item.strip() for item in args.case_ids.split(",") if item.strip()]
    unknown = sorted(set(wanted) - set(records))
    if unknown:
        raise SystemExit(f"case ids missing from dataset split {args.split}: {unknown}")

    provider = AssistantSessionProvider.for_director(
        session_root=args.session_root,
        wait_timeout_s=1.0,
        poll_interval_s=0.05,
    )
    director = DirectorAgent.from_provider(
        provider,
        provider_name="assistant-session-glm-flash",
        policy="director-v5-glm-structured",
    )
    report = {"split": args.split, "cases": {}, "pass_count": 0, "fail_count": 0}
    for case_id in wanted:
        record = records[case_id]
        entry: dict = {"case_id": case_id}
        try:
            result = director.plan(
                record["prompt"],
                scene_id=case_id,
                duration_s=record["duration_s"],
                fps=record["fps"],
                obligations=case_obligations(record),
            )
            entry["status"] = "pass"
            entry["director_plan_hash"] = result.director_plan_hash
            entry["entity_ids"] = [
                entity.id for entity in result.director_plan.entities
            ]
            entry["event_ids"] = [event.id for event in result.director_plan.events]
            entry["camera_shots"] = len(result.director_plan.camera_plan.shots)
            report["pass_count"] += 1
        except Exception as exc:  # noqa: BLE001 - preflight reports every failure
            entry["status"] = "fail"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            report["fail_count"] += 1
        report["cases"][case_id] = entry
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report["fail_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
