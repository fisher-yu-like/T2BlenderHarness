"""Generate-once-freeze cache for per-plan Blender job sources."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


class CodeCache:
    """Persist immutable source by ``plan_hash`` and ``harness_version``."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "code_manifest.jsonl"

    @staticmethod
    def _safe_part(value: str) -> str:
        if not str(value).strip():
            raise ValueError("cache key parts must be non-empty")
        return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))

    def _path(self, plan_hash: str, harness_version: str) -> Path:
        key = f"{self._safe_part(plan_hash)}_{self._safe_part(harness_version)}"
        return self.cache_dir / f"{key}.py"

    def lookup(self, plan_hash: str, harness_version: str) -> str | None:
        """Return frozen source for a matching plan/version, or ``None``."""

        path = self._path(plan_hash, harness_version)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def frozen_source_hashes(self, harness_version: str) -> dict[str, str]:
        """Return prior plan-to-source hashes for duplicate detection."""

        if not self.manifest_path.is_file():
            return {}
        result: dict[str, str] = {}
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if entry.get("harness_version") != harness_version:
                continue
            plan_hash = str(entry.get("plan_hash") or "")
            code_hash = str(entry.get("code_hash") or "")
            if plan_hash and len(code_hash) == 64:
                result[plan_hash] = code_hash
        return result

    def store(
        self,
        plan_hash: str,
        harness_version: str,
        code: str,
        *,
        llm_call_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Freeze source and append an auditable manifest entry.

        Existing content is never silently overwritten.  A same-key identical
        source is idempotent; a different source requires a new version or an
        explicit cache cleanup by the caller.
        """

        if not code.strip():
            raise ValueError("cannot freeze empty code")
        path = self._path(plan_hash, harness_version)
        if path.is_file():
            existing = path.read_text(encoding="utf-8")
            if existing != code:
                raise ValueError(f"frozen source already exists with different content: {path}")
            return path
        path.write_text(code, encoding="utf-8")
        code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
        entry = {
            "plan_hash": str(plan_hash),
            "harness_version": str(harness_version),
            "code_hash": code_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "llm_call_id": str(llm_call_id),
            "frozen_path": str(path.relative_to(self.cache_dir.parent)),
        }
        if metadata:
            reserved = set(entry) & set(metadata)
            if reserved:
                raise ValueError(f"cache metadata cannot overwrite core fields: {sorted(reserved)}")
            entry.update(dict(metadata))
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return path


__all__ = ["CodeCache"]
