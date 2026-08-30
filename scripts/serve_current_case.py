"""Serve the current pending case end-to-end.

Answers the pending director request with an authored spec, waits for the
matching blender_code request of the same case, assembles the authored scene
code with the host-contract wrapper, and prints the next pending state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def next_pending() -> dict:
    out = subprocess.run(
        [sys.executable, "scripts/next_pending.py"], capture_output=True, text=True
    )
    return json.loads(out.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--director-spec", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--case", required=True, help="expected scene_id, e.g. vbench2-dev-02-12")
    parser.add_argument("--author-out", required=True)
    parser.add_argument("--dryrun-template", default="out/training/glm-flash-diagnostic-v1/round-01/attempt-01/real/train/vbench2-train-01-01")
    parser.add_argument("--wait-s", type=float, default=120.0)
    args = parser.parse_args()

    pending = next_pending()
    entry = pending.get("next") or {}
    if entry.get("stage") != "director" or (entry.get("digest") or {}).get("scene_id") != args.case:
        print(json.dumps({"error": "pending is not the expected director request", "pending": pending}))
        return 2
    subprocess.run(
        [sys.executable, "scripts/assistant_author_director.py",
         "--request", entry["path"], "--spec", args.director_spec],
        capture_output=True, text=True,
    )
    deadline = time.monotonic() + args.wait_s
    codegen_path = None
    while time.monotonic() < deadline:
        pending = next_pending()
        entry = pending.get("next") or {}
        if entry.get("stage") == "blender_code" and (entry.get("digest") or {}).get("scene_id") == args.case:
            codegen_path = entry["path"]
            break
        if pending.get("pending", 0) == 0:
            time.sleep(1.0)
            continue
        time.sleep(1.0)
    if codegen_path is None:
        print(json.dumps({"error": "no matching codegen request appeared"}))
        return 2
    result = subprocess.run(
        [sys.executable, "scripts/assemble_case_source.py",
         "--request", codegen_path, "--scene-code", args.scene,
         "--source-out", args.author_out, "--dryrun-template", args.dryrun_template],
        capture_output=True, text=True,
    )
    print(result.stdout[-500:] if result.returncode else result.stdout[-120:])
    after = next_pending()
    follow = after.get("next") or {}
    print(json.dumps({
        "case": args.case,
        "served": result.returncode == 0,
        "next_pending": after.get("pending"),
        "next_stage": follow.get("stage"),
        "next_scene": (follow.get("digest") or {}).get("scene_id"),
        "next_prompt": (follow.get("digest") or {}).get("prompt"),
        "next_path": follow.get("path"),
    }, ensure_ascii=False))
    return 0 if result.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
