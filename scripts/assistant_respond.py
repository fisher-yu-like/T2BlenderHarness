"""Respond to pending assistant-session provider requests.

The driving coding agent authors Director interpretations and Blender sources;
this tool materializes them as provider response files.  It never fabricates
content: responses come from explicit files supplied by the agent session.

Usage:
  python scripts/assistant_respond.py --list [--session-root out/assistant-session]
  python scripts/assistant_respond.py --stage director --call-id <id> \
      --response-file my_interpretation.json
  python scripts/assistant_respond.py --stage blender_code --call-id <id> \
      --source-file my_blender_job.py [--library-calls box,cylinder]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def pending_requests(session_root: Path) -> list[dict]:
    pending = []
    for stage in ("director", "blender_code"):
        requests_dir = session_root / "requests" / stage
        responses_dir = session_root / "responses" / stage
        if not requests_dir.is_dir():
            continue
        for request_path in sorted(requests_dir.glob("*.json")):
            if (responses_dir / request_path.name).is_file():
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            payload = request.get("payload") or {}
            plan = payload.get("director_plan") or {}
            request_obj = plan.get("request") or payload.get("request") or {}
            pending.append(
                {
                    "stage": stage,
                    "call_id": request.get("call_id"),
                    "scene_id": request.get("scene_id"),
                    "prompt": (request_obj.get("prompt") or payload.get("prompt") or "")[:160],
                    "request_path": str(request_path),
                    "respond_to": request.get("respond_to"),
                }
            )
    return pending


def extract_library_calls(source: str, allowed: set[str]) -> list[str]:
    """Collect verified-library primitives actually called in the source."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in allowed:
                calls.add(function.id)
            elif isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
                if function.attr in allowed:
                    calls.add(function.attr)
    return sorted(calls)


def write_response(session_root: Path, stage: str, call_id: str, response: dict) -> Path:
    file_token = call_id.replace(":", "-")
    target = session_root / "responses" / stage / f"{file_token}.json"
    if not target.parent.is_dir():
        raise SystemExit(f"unknown request (no pending {stage} request dir): {call_id}")
    target.write_text(
        json.dumps(response, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", default="out/assistant-session")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--stage", choices=["director", "blender_code"])
    parser.add_argument("--call-id")
    parser.add_argument("--response-file", help="director: JSON interpretation file")
    parser.add_argument("--source-file", help="blender_code: Python source file")
    parser.add_argument("--library-calls", help="comma-separated extra library call names")
    parser.add_argument(
        "--uncertainties",
        help="JSON list of uncertainty dicts for hard_uncertainty responses",
    )
    parser.add_argument("--fallback-reason")
    args = parser.parse_args()
    session_root = Path(args.session_root)

    if args.list:
        pending = pending_requests(session_root)
        print(json.dumps({"pending_count": len(pending), "pending": pending}, ensure_ascii=False, indent=2))
        return 0 if not pending else 3

    if not args.stage or not args.call_id:
        parser.error("--stage and --call-id are required unless --list is used")
    # Request/response files use the Windows-safe token (':' replaced).
    file_token = args.call_id.replace(":", "-")
    request_path = session_root / "requests" / args.stage / f"{file_token}.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))

    if args.stage == "director":
        if not args.response_file:
            parser.error("--response-file is required for the director stage")
        interpretation = json.loads(Path(args.response_file).read_text(encoding="utf-8"))
        response = dict(interpretation)
    else:
        if not args.source_file:
            parser.error("--source-file is required for the blender_code stage")
        source = Path(args.source_file).read_text(encoding="utf-8")
        allowed: set[str] = set()
        for entries in ((request.get("payload") or {}).get("library_signatures") or {}).values():
            for entry in entries or []:
                name = str((entry or {}).get("name") or "").strip()
                if name:
                    allowed.add(name)
        calls = extract_library_calls(source, allowed)
        for extra in filter(None, [item.strip() for item in (args.library_calls or "").split(",")]):
            if extra not in allowed:
                raise SystemExit(f"library call not in verified signatures: {extra}")
            calls.append(extra) if extra not in calls else None
        if args.fallback_reason:
            response = {
                "status": "library_insufficient",
                "fallback_reason": args.fallback_reason,
                "library_calls": calls,
            }
        elif args.uncertainties:
            response = {
                "status": "hard_uncertainty",
                "uncertainties": json.loads(args.uncertainties),
                "library_calls": calls,
            }
        else:
            response = {
                "status": "success",
                "generated_code": source,
                "library_calls": calls,
            }
    target = write_response(session_root, args.stage, args.call_id, response)
    print(json.dumps({"written": str(target), "stage": args.stage, "call_id": args.call_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
