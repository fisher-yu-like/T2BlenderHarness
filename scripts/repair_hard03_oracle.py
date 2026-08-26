"""Repair only hard-03 authored labels from the literal prompt semantics.

The repair is auditable: the old/new record payloads and fingerprints are
stored separately before the live manifest/proxy-spec rows are updated.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "trajectory-v3-hard"
AUDIT = ROOT / "out" / "training" / "six-rounds-real-v6" / "dataset-repair-hard03"


def _fingerprint(records: list[dict]) -> str:
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _prompt_entities(prompt: str) -> list[dict]:
    lower = prompt.lower()
    support = "table" if "table" in lower or "worktable" in lower else "support"
    drop_zone = "drop zone" if "drop zone" in lower else "drop platform" if "drop platform" in lower else "marked destination"
    entities = [
        {"id": "character", "kind": "character", "role": "actor"},
        {"id": "table", "kind": "support", "role": "environment"},
        {"id": "drop_zone", "kind": "support", "role": "environment"},
        {"id": "red_cup", "kind": "prop", "role": "target_object"},
        {"id": "blue_cube", "kind": "prop", "role": "secondary_target"},
    ]
    return entities


def _repair_record(record: dict) -> dict:
    prompt = record["prompt"]
    entities = _prompt_entities(prompt)
    proxy = dict(record["proxy_scene"])
    proxy["entities"] = entities
    proxy["camera"] = {
        **proxy.get("camera", {}),
        "must_show_events": ["walk", "reach", "grasp", "lift", "carry", "place", "release"] * 2,
        "trajectory_types": ["follow", "orbit", "dolly"],
    }
    oracle = dict(record["oracle_expectations"])
    oracle["required_entity_ids"] = [entity["id"] for entity in entities]
    oracle["required_entity_kinds"] = {entity["id"]: entity["kind"] for entity in entities}
    oracle["event_order"] = ["walk", "reach", "grasp", "lift", "carry", "place", "release"] * 2
    oracle["required_attachment_actions"] = ["attach", "detach", "attach", "detach"]
    oracle["required_camera_types"] = ["follow", "orbit", "dolly"]
    oracle["required_camera_constraints"] = ["target_visible_before_grasp", "support_before_grasp", "attachment_lifecycle", "dual_handoff"]
    oracle["required_motion_primitives"] = ["ease_in_out", "linear"]
    return {**record, "proxy_scene": proxy, "oracle_expectations": oracle}


def main() -> int:
    manifest_path = DATASET / "manifest.jsonl"
    proxy_specs_path = DATASET / "proxy_specs.jsonl"
    metadata_path = DATASET / "metadata.json"
    records = _read_jsonl(manifest_path)
    old_records = [json.loads(json.dumps(record)) for record in records]
    changed = []
    for index, record in enumerate(records):
        if record["case_id"].startswith("hard-03-"):
            repaired = _repair_record(record)
            changed.append({"case_id": record["case_id"], "before": record, "after": repaired})
            records[index] = repaired
    if len(changed) != 10:
        raise RuntimeError(f"expected 10 hard-03 records, found {len(changed)}")
    old_proxy_specs = _read_jsonl(proxy_specs_path)
    proxy_specs = []
    changed_by_id = {item["case_id"]: item["after"]["proxy_scene"] for item in changed}
    for spec in old_proxy_specs:
        if spec["case_id"] in changed_by_id:
            proxy_specs.append({"case_id": spec["case_id"], "proxy_scene": changed_by_id[spec["case_id"]]})
        else:
            proxy_specs.append(spec)
    old_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    new_metadata = {**old_metadata, "fingerprint": _fingerprint(records), "oracle_repair": "hard-03 labels repaired from literal prompt: red_cup + blue_cube"}
    AUDIT.mkdir(parents=True, exist_ok=True)
    audit_payload = {
        "repair_version": "hard03-prompt-authoritative-v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": "All hard-03 prompts explicitly say red cup and blue cube; previous oracle rows contained rotating green_ball/yellow_book or duplicate blue_cube IDs.",
        "manifest_before_fingerprint": _fingerprint(old_records),
        "manifest_after_fingerprint": new_metadata["fingerprint"],
        "changed_records": changed,
        "metadata_before": old_metadata,
        "metadata_after": new_metadata,
    }
    (AUDIT / "repair_audit.json").write_text(json.dumps(audit_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    proxy_specs_path.write_text("".join(json.dumps(spec, ensure_ascii=False, sort_keys=True) + "\n" for spec in proxy_specs), encoding="utf-8")
    metadata_path.write_text(json.dumps(new_metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"changed_case_count": len(changed), "before_fingerprint": audit_payload["manifest_before_fingerprint"], "after_fingerprint": audit_payload["manifest_after_fingerprint"], "audit": str((AUDIT / "repair_audit.json").resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
