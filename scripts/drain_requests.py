"""Serve every pending assistant codegen request, then exit when idle.

Front-loop tool for the driving session: polls the session root, materializes
each pending blender_code request through the authored generator, and stops
after a few idle rounds so the host can watch renders.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

IDLE_ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
served = 0
idle = 0
while idle < IDLE_ROUNDS:
    listing = subprocess.run(
        [sys.executable, "scripts/assistant_respond.py", "--session-root", "out/assistant-session", "--list"],
        capture_output=True,
        text=True,
    )
    try:
        data = json.loads(listing.stdout)
    except json.JSONDecodeError:
        print("list failed:", listing.stdout[-200:], listing.stderr[-200:], flush=True)
        break
    pending = [item for item in data.get("pending", []) if item["stage"] == "blender_code"]
    if not pending:
        idle += 1
        time.sleep(4)
        continue
    idle = 0
    for item in pending:
        token = item["call_id"].split(":")[-1]
        req_path = f"out/assistant-session/requests/blender_code/assistant-blender_code-{token}.json"
        scene = item["scene_id"] or "unknown"
        result = subprocess.run(
            [sys.executable, "scripts/author_general_scene_source.py", "--request", req_path, "--source-out", f"out/assistant-session/authoring/{scene}.py"],
            capture_output=True,
            text=True,
        )
        status = "ok" if result.returncode == 0 else "FAIL"
        if result.returncode == 0:
            # Mock-Blender dry-run: catch runtime errors (undefined names,
            # frozen-dataclass writes) before a real render is spent.
            template = None
            for candidate in Path("out/training/glm-flash-diagnostic-v1/round-01/attempt-01/real/train").glob("vbench2-train-01-01"):
                template = candidate
            dry = subprocess.run(
                [sys.executable, "scripts/mock_blender_dryrun.py", f"out/assistant-session/authoring/{scene}.py", str(template) if template else ""],
                capture_output=True,
                text=True,
            )
            if dry.returncode != 0:
                status = "DRYRUN_FAIL"
                print((dry.stdout or "")[-500:], flush=True)
        served += 1
        print(f"{scene} {token[-8:]} {status}", flush=True)
        if result.returncode != 0:
            print(result.stdout[-600:], result.stderr[-300:], flush=True)
print("drained", served, "requests", flush=True)
