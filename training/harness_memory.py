"""Append-only memory for Harness update proposals, evaluations, and decisions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EVENTS = {"proposal", "patch_applied", "train_evaluated", "dev_evaluated", "test_evaluated", "accepted", "rejected", "rollback", "no_patch"}


class HarnessMemoryStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "harness_updates.jsonl"

    def begin_update(
        self,
        *,
        parent_version: str,
        candidate_version: str,
        owner: str,
        dataset_fingerprint: str,
        evaluator_fingerprint: str,
        affected_case_ids: Iterable[str],
        forbidden_case_ids: set[str] | None = None,
    ) -> str:
        case_ids = sorted({str(case_id) for case_id in affected_case_ids})
        leaked = set(case_ids) & (forbidden_case_ids or set())
        if leaked:
            raise ValueError(f"test case IDs cannot enter Harness memory proposal: {sorted(leaked)}")
        if not owner or owner == "unknown":
            raise ValueError("Harness update memory requires exactly one owner")
        payload = {
            "parent_version": parent_version,
            "candidate_version": candidate_version,
            "owner": owner,
            "dataset_fingerprint": dataset_fingerprint,
            "evaluator_fingerprint": evaluator_fingerprint,
            "affected_case_ids": case_ids,
        }
        memory_id = "memory-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        self._append(
            memory_id,
            "proposal",
            base=payload,
            payload={"status": "proposal"},
        )
        return memory_id

    def append_event(self, memory_id: str, event: str, **payload: Any) -> dict[str, Any]:
        if event not in EVENTS:
            raise ValueError(f"unknown Harness memory event: {event}")
        history = self.history(memory_id)
        if not history:
            raise ValueError(f"unknown Harness memory ID: {memory_id}")
        if event == "accepted":
            required = {"train_before", "train_after", "dev_before", "dev_after", "hard_regression"}
            missing = required - set(payload)
            if missing:
                raise ValueError(f"accepted memory event missing {sorted(missing)}")
            if float(payload["train_after"]) <= float(payload["train_before"]):
                raise ValueError("accepted memory event requires train improvement")
            if float(payload["dev_after"]) < float(payload["dev_before"]) or payload["hard_regression"]:
                raise ValueError("accepted memory event requires non-regressing dev")
        base = {key: history[0][key] for key in ("parent_version", "candidate_version", "owner", "dataset_fingerprint", "evaluator_fingerprint", "affected_case_ids")}
        return self._append(memory_id, event, base=base, payload=payload)

    def record_no_patch(
        self,
        *,
        harness_version: str,
        dataset_fingerprint: str,
        evaluator_fingerprint: str,
        reason: str,
    ) -> str:
        memory_id = self.begin_update(
            parent_version=harness_version,
            candidate_version=harness_version,
            owner="none",
            dataset_fingerprint=dataset_fingerprint,
            evaluator_fingerprint=evaluator_fingerprint,
            affected_case_ids=[],
        )
        self.append_event(memory_id, "no_patch", reason=reason)
        return memory_id

    def history(self, memory_id: str) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            record
            for record in self._read_all()
            if record.get("memory_id") == memory_id
        ]

    def retrieve(
        self,
        *,
        owner: str | None = None,
        failure_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self._read_all()
        if owner:
            records = [record for record in records if record.get("owner") == owner]
        if failure_id:
            records = [record for record in records if record.get("failure_id") == failure_id or record.get("payload", {}).get("failure_id") == failure_id]
        return records[-limit:] if limit else records

    def _append(self, memory_id: str, event: str, *, base: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        event_record = {
            "memory_id": memory_id,
            "event_index": len(self.history(memory_id)),
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **base,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_record, sort_keys=True) + "\n")
        return event_record

    def _read_all(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
