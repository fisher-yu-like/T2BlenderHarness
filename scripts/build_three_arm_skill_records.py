"""Turn real trained-arm video reviews into proposal-only skill evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SOURCE = "actual_blender_video_local_review"


def _gm(values: list[float]) -> float:
    return math.prod(max(0.0, float(value)) for value in values) ** (1.0 / len(values)) if values else 0.0


def _trajectory(scores: dict[str, Any]) -> float:
    return _gm([float(scores[name]) for name in ("object_trajectory", "character_trajectory", "event_timing", "temporal_smoothness") if isinstance(scores.get(name), (int, float))])


def _camera(scores: dict[str, Any]) -> float:
    return _gm([float(scores[name]) for name in ("camera_coverage", "camera_innovation", "visual_clarity") if isinstance(scores.get(name), (int, float))])


def build_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit one normalized finding per train case, with one owner per record.

    The thresholds are diagnostic triggers, not score modifiers.  They only
    select repeated failure evidence for the existing proposal tool.
    """
    records = []
    for row in rows:
        if row.get("arm") != "trained" or row.get("split") != "train":
            continue
        if row.get("review_status") not in {None, "complete"}:
            continue
        scores = row.get("review_scores") or {}
        trajectory = _trajectory(scores)
        camera = _camera(scores)
        appearance = scores.get("appearance_detail")
        if trajectory < 50.0:
            failure_id = "trajectory_event_readability_low"
            owner = "director_trajectory"
            category = "trajectory_planning"
            message = f"local review found weak visible character/object phase continuity (trajectory diagnostic={trajectory:.2f})"
            route = "director_trajectory_review"
        elif camera < 70.0:
            failure_id = "camera_choreography_visibility_low"
            owner = "director_camera"
            category = "camera_coverage"
            message = f"local review found weak multi-target camera visibility or innovation (camera diagnostic={camera:.2f})"
            route = "director_camera_review"
        elif isinstance(appearance, (int, float)) and float(appearance) < 55.0:
            failure_id = "proxy_visual_detail_low"
            owner = "proxy_renderer"
            category = "realism_presentation"
            message = f"local review found insufficient visible proxy detail (appearance_detail={float(appearance):.2f})"
            route = "proxy_detail_review"
        else:
            continue
        video = row.get("proxy_video")
        evidence = [f"proxy_video:{video}"] if video else []
        evidence.extend(f"run_dir:{item}" for item in (row.get("run_dir"), row.get("review_request")) if item)
        records.append(
            {
                "case_id": row.get("case_id"),
                "split": row.get("split"),
                "record_type": "three_arm_real_video_skill_evidence",
                "source": SOURCE,
                "status": "fail",
                "review_source": "assistant_local_review",
                "review_method": "codex-local-visual-frame-analysis-v1",
                "findings": [
                    {
                        "failure_id": failure_id,
                        "owner": owner,
                        "category": category,
                        "severity": "warning",
                        "root_cause_id": failure_id,
                        "message": message,
                        "evidence": evidence,
                        "repair_route": route,
                    }
                ],
                "observed_scores": {
                    "trajectory_diagnostic": round(trajectory, 4),
                    "camera_diagnostic": round(camera, 4),
                    "appearance_detail": appearance,
                    "task_vlm": row.get("task_vlm"),
                    "realism_final": row.get("realism_final"),
                },
                "proxy_video": video,
                "arm": "trained",
            }
        )
    return records


def build_from_aggregate(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return build_records(payload.get("flat_rows") or [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    records = build_from_aggregate(args.aggregate)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    print(json.dumps({"record_count": len(records), "out": str(destination.resolve()), "source": SOURCE}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
