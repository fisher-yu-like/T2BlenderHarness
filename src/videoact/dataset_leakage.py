"""Deterministic train/dev/frozen leakage checks for benchmark prompts."""

from __future__ import annotations

import hashlib
import itertools
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


LEAKAGE_AUDIT_VERSION = "dataset-leakage-v1-four-layer"
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+", flags=re.UNICODE)


def normalize_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", str(prompt or "")).casefold()
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ngrams(text: str, size: int = 3) -> dict[str, int]:
    if not text:
        return {}
    if len(text) < size:
        return {text: 1}
    result: dict[str, int] = {}
    for index in range(len(text) - size + 1):
        gram = text[index : index + size]
        result[gram] = result.get(gram, 0) + 1
    return result


def cosine_similarity(first: str, second: str) -> float:
    left, right = _ngrams(normalize_prompt(first)), _ngrams(normalize_prompt(second))
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _identity(record: Mapping[str, Any], ordinal: int, source_name: str) -> dict[str, Any]:
    prompt = str(record.get("prompt") or record.get("source_prompt") or "")
    source_dataset = str(record.get("source_dataset") or source_name)
    source_dimension = str(record.get("source_dimension") or record.get("category") or "uncategorized")
    try:
        source_index: int | None = int(record["source_index"])
    except (KeyError, TypeError, ValueError):
        source_index = None
    return {
        "source_name": source_name,
        "ordinal": ordinal,
        "case_id": str(record.get("case_id") or f"{source_name}:{ordinal}"),
        "prompt": prompt,
        "prompt_hash": str(record.get("prompt_hash") or _hash(prompt)),
        "normalized_prompt_hash": _hash(normalize_prompt(prompt)),
        "source_dataset": source_dataset,
        "source_index": source_index,
        "source_family": (source_dataset, source_dimension),
    }


def audit_leakage(
    current_records: Iterable[Mapping[str, Any]],
    reference_records: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    *,
    near_duplicate_threshold: float = 0.92,
) -> dict[str, Any]:
    """Audit current records against each other and named reference sets."""

    if not 0.0 < near_duplicate_threshold <= 1.0:
        raise ValueError("near_duplicate_threshold must be between zero and one")
    current = [_identity(record, index, "current") for index, record in enumerate(current_records)]
    references = {
        str(name): [_identity(record, index, str(name)) for index, record in enumerate(records)]
        for name, records in (reference_records or {}).items()
    }
    all_records = current + [item for rows in references.values() for item in rows]
    collisions: list[dict[str, Any]] = []
    counts = {
        "exact_prompt": 0,
        "normalized_prompt": 0,
        "semantic_near_duplicate": 0,
        "source_family": 0,
        "source_identity": 0,
        "case_id": 0,
    }
    for left, right in itertools.combinations(all_records, 2):
        if left["source_name"] != "current" and right["source_name"] != "current":
            continue
        if left["source_name"] == right["source_name"] == "current":
            scope = "within_current"
        elif left["source_name"] == right["source_name"]:
            scope = "within_reference"
        else:
            scope = "current_vs_reference"
        pair_base = {
            "left": left["case_id"],
            "right": right["case_id"],
            "left_source": left["source_name"],
            "right_source": right["source_name"],
            "scope": scope,
        }
        if left["case_id"] == right["case_id"]:
            counts["case_id"] += 1
            collisions.append({**pair_base, "kind": "case_id"})
        if left["prompt_hash"] == right["prompt_hash"]:
            counts["exact_prompt"] += 1
            collisions.append({**pair_base, "kind": "exact_prompt"})
        if left["normalized_prompt_hash"] == right["normalized_prompt_hash"]:
            counts["normalized_prompt"] += 1
            collisions.append({**pair_base, "kind": "normalized_prompt"})
        if (
            left["source_dataset"] == right["source_dataset"]
            and left["source_index"] is not None
            and left["source_index"] == right["source_index"]
        ):
            counts["source_identity"] += 1
            collisions.append({**pair_base, "kind": "source_identity", "value": [left["source_dataset"], left["source_index"]]})
        if left["source_name"] != right["source_name"] and left["source_family"] == right["source_family"]:
            counts["source_family"] += 1
            collisions.append({**pair_base, "kind": "source_family", "value": list(left["source_family"])})
        similarity = cosine_similarity(left["prompt"], right["prompt"])
        if (
            similarity >= near_duplicate_threshold
            and left["normalized_prompt_hash"] != right["normalized_prompt_hash"]
            and left["prompt"]
            and right["prompt"]
        ):
            counts["semantic_near_duplicate"] += 1
            collisions.append({**pair_base, "kind": "semantic_near_duplicate", "similarity": round(similarity, 6)})
    return {
        "version": LEAKAGE_AUDIT_VERSION,
        "status": "fail" if collisions else "pass",
        "current_count": len(current),
        "reference_counts": {name: len(records) for name, records in references.items()},
        "pairs_checked": len(all_records) * max(0, len(all_records) - 1) // 2,
        "near_duplicate_threshold": float(near_duplicate_threshold),
        "collision_counts": counts,
        "collisions": collisions,
    }


__all__ = ["LEAKAGE_AUDIT_VERSION", "audit_leakage", "cosine_similarity", "normalize_prompt"]
