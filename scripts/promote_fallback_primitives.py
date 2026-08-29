"""Create human-reviewable L4 primitive promotion candidates.

This tool is intentionally read-only.  Three occurrences create a candidate
report, never an automatic library edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


ELIGIBLE_REVIEW_SOURCES = {
    "human_review",
    "codex_local_visual_review",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    elif path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).replace("\\", "/").encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def build_promotion_report(
    manifest_paths: Sequence[str | Path],
    *,
    library_roots: Sequence[str | Path] = (),
    minimum_occurrences: int = 3,
) -> dict[str, Any]:
    occurrences: dict[str, set[str]] = {}
    scanned_entries = 0
    eligible_entries = 0
    for raw_path in manifest_paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        for entry in _read_jsonl(path):
            scanned_entries += 1
            if entry.get("status") not in {"success", "artifact_valid", "evaluated"}:
                continue
            if entry.get("artifact_status") not in {None, "complete"}:
                continue
            review_source = str(entry.get("review_source") or "")
            model = str(entry.get("model") or "")
            if review_source not in ELIGIBLE_REVIEW_SOURCES and not (
                review_source == "external_vlm" and model in {"gpt-5.6-luna", "gpt-5.6-terra"}
            ):
                continue
            case_id = str(entry.get("case_id") or "")
            if not case_id:
                continue
            eligible_entries += 1
            primitive_values = entry.get("new_primitives") or []
            for value in primitive_values:
                primitive = str(value.get("name") if isinstance(value, dict) else value).strip()
                if primitive:
                    occurrences.setdefault(primitive, set()).add(case_id)
    candidates = [
        {
            "primitive": primitive,
            "case_ids": sorted(case_ids),
            "case_count": len(case_ids),
            "requires_human_approval": True,
            "next_steps": ["review implementation and side effects", "add signature/docstring", "add unit and Blender smoke tests", "update signatures.json and bump harness_version"],
        }
        for primitive, case_ids in sorted(occurrences.items())
        if len(case_ids) >= minimum_occurrences
    ]
    before = {str(Path(path).resolve()): _hash_path(Path(path)) for path in library_roots if Path(path).exists()}
    return {
        "status": "promotion_candidates" if candidates else "no_candidates",
        "minimum_occurrences": minimum_occurrences,
        "scanned_entries": scanned_entries,
        "eligible_entries": eligible_entries,
        "candidates": candidates,
        "library_hashes_before": before,
        "library_mutation": "none",
        "requires_human_approval": bool(candidates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--library-root", action="append", default=[])
    parser.add_argument("--minimum-occurrences", type=int, default=3)
    args = parser.parse_args()
    report = build_promotion_report(
        args.manifest,
        library_roots=args.library_root,
        minimum_occurrences=args.minimum_occurrences,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
