"""Print the next pending assistant request with an authoring digest."""
import glob
import json
import sys
from pathlib import Path

session = Path("out/assistant-session")
items = []
for stage in ("director", "blender_code"):
    req_dir = session / "requests" / stage
    res_dir = session / "responses" / stage
    if not req_dir.is_dir():
        continue
    for path in sorted(req_dir.glob("*.json")):
        if (res_dir / path.name).is_file():
            continue
        req = json.loads(path.read_text(encoding="utf-8"))
        payload = req.get("payload") or {}
        if stage == "director":
            digest = {"prompt": payload.get("prompt"), "scene_id": payload.get("scene_id"),
                      "obligations": payload.get("obligations")}
        else:
            plan = payload.get("director_plan") or {}
            digest = {
                "scene_id": (plan.get("request") or {}).get("scene_id"),
                "entities": [(e.get("id"), e.get("kind"), e.get("label")) for e in plan.get("entities", [])],
                "events": [(e.get("id"), e.get("action"), e.get("participant_ids"), e.get("target_ids")) for e in plan.get("events", [])],
                "states": {eid: [s.get("position") for s in d.get("states", [])]
                            for eid, d in (plan.get("trajectory_summary") or {}).get("entities", {}).items()},
                "primitives": {eid: [(p.get("type"), p.get("start_frame"), p.get("end_frame"), p.get("parameters"))
                                      for p in d.get("motion_primitives", [])]
                                for eid, d in (plan.get("trajectory_summary") or {}).get("entities", {}).items()},
                "shots": plan.get("camera_plan", {}).get("shots", []),
                "interactions": plan.get("interactions", []),
                "job_hash": payload.get("director_plan_hash"),
            }
        items.append({"stage": stage, "call_id": req.get("call_id"), "path": str(path), "digest": digest})
if not items:
    print(json.dumps({"pending": 0}))
    sys.exit(0)
print(json.dumps({"pending": len(items), "next": items[0]}, ensure_ascii=False, indent=1, default=str))
