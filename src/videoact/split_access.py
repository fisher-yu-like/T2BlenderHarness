"""Train/Dev/Test access policy for closed-loop controller inputs.

The policy stores only identities, hashes, source-family labels, and compact
prompt n-gram profiles.  It is sufficient to reject leakage checks without
making test prompts or test scores available to the controller.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


SPLIT_ACCESS_SCHEMA_VERSION = "split-access-policy-v1"
_NON_WORD = re.compile(r"[^\w\u4e00-\u9fff]+", flags=re.UNICODE)


def normalize_prompt(prompt: str) -> str:
    text = unicodedata.normalize("NFKC", str(prompt or "")).casefold()
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def _profile(prompt: str, size: int = 3) -> dict[str, int]:
    text = normalize_prompt(prompt)
    if not text:
        return {}
    if len(text) < size:
        return {text: 1}
    result: dict[str, int] = {}
    for index in range(len(text) - size + 1):
        gram = text[index : index + size]
        result[gram] = result.get(gram, 0) + 1
    return result


def _similarity(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = sum(value * value for value in left.values()) ** 0.5
    right_norm = sum(value * value for value in right.values()) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _record_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    prompt = str(record.get("prompt") or record.get("source_prompt") or "")
    source_dataset = str(record.get("source_dataset") or "")
    source_dimension = str(record.get("source_dimension") or record.get("category") or "")
    source_index = record.get("source_index")
    try:
        source_index = int(source_index) if source_index is not None else None
    except (TypeError, ValueError):
        source_index = None
    return {
        "case_id": str(record.get("case_id") or ""),
        "prompt_hash": str(record.get("prompt_hash") or prompt_hash(prompt)),
        "normalized_prompt_hash": hashlib.sha256(normalize_prompt(prompt).encode("utf-8")).hexdigest(),
        "profile": _profile(prompt),
        "source_family": [source_dataset, source_dimension],
        "source_identity": [source_dataset, source_index] if source_dataset and source_index is not None else None,
    }


class SplitAccessPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SPLIT_ACCESS_SCHEMA_VERSION
    train_case_ids: list[str] = Field(default_factory=list)
    dev_case_ids: list[str] = Field(default_factory=list)
    test_case_ids: list[str] = Field(default_factory=list)
    train_prompt_hashes: list[str] = Field(default_factory=list)
    dev_prompt_hashes: list[str] = Field(default_factory=list)
    test_prompt_hashes: list[str] = Field(default_factory=list)
    train_normalized_prompt_hashes: list[str] = Field(default_factory=list)
    dev_normalized_prompt_hashes: list[str] = Field(default_factory=list)
    test_normalized_prompt_hashes: list[str] = Field(default_factory=list)
    dev_test_source_families: list[list[str]] = Field(default_factory=list)
    dev_test_source_identities: list[list[Any]] = Field(default_factory=list)
    dev_test_prompt_profiles: list[dict[str, int]] = Field(default_factory=list)
    near_duplicate_threshold: float = Field(default=0.92, gt=0.0, le=1.0)

    @classmethod
    def from_records(
        cls,
        train_records: Iterable[Mapping[str, Any]],
        dev_records: Iterable[Mapping[str, Any]],
        test_records: Iterable[Mapping[str, Any]],
        *,
        near_duplicate_threshold: float = 0.92,
    ) -> "SplitAccessPolicy":
        groups = {
            "train": [_record_identity(item) for item in train_records],
            "dev": [_record_identity(item) for item in dev_records],
            "test": [_record_identity(item) for item in test_records],
        }
        forbidden = groups["dev"] + groups["test"]
        return cls(
            train_case_ids=sorted(item["case_id"] for item in groups["train"] if item["case_id"]),
            dev_case_ids=sorted(item["case_id"] for item in groups["dev"] if item["case_id"]),
            test_case_ids=sorted(item["case_id"] for item in groups["test"] if item["case_id"]),
            train_prompt_hashes=sorted(item["prompt_hash"] for item in groups["train"]),
            dev_prompt_hashes=sorted(item["prompt_hash"] for item in groups["dev"]),
            test_prompt_hashes=sorted(item["prompt_hash"] for item in groups["test"]),
            train_normalized_prompt_hashes=sorted(item["normalized_prompt_hash"] for item in groups["train"]),
            dev_normalized_prompt_hashes=sorted(item["normalized_prompt_hash"] for item in groups["dev"]),
            test_normalized_prompt_hashes=sorted(item["normalized_prompt_hash"] for item in groups["test"]),
            dev_test_source_families=sorted({tuple(item["source_family"]) for item in forbidden if any(item["source_family"])}),
            dev_test_source_identities=sorted({tuple(item["source_identity"]) for item in forbidden if item["source_identity"] is not None}),
            dev_test_prompt_profiles=[item["profile"] for item in forbidden if item["profile"]],
            near_duplicate_threshold=near_duplicate_threshold,
        )

    @property
    def forbidden_case_ids(self) -> set[str]:
        return set(self.dev_case_ids) | set(self.test_case_ids)

    @property
    def forbidden_prompt_hashes(self) -> set[str]:
        return set(self.dev_prompt_hashes) | set(self.test_prompt_hashes)

    def validate_records(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Reject any train record colliding with the held-out identity policy."""

        for record in records:
            if str(record.get("split") or "train").casefold() != "train":
                raise ValueError("split access policy accepts train records only")
            identity = _record_identity(record)
            case_id = identity["case_id"]
            if case_id in self.forbidden_case_ids:
                raise ValueError(f"split leakage: forbidden dev/test case ID {case_id}")
            if identity["prompt_hash"] in self.forbidden_prompt_hashes:
                raise ValueError(f"split leakage: forbidden exact prompt hash for {case_id}")
            if identity["normalized_prompt_hash"] in set(self.dev_normalized_prompt_hashes) | set(self.test_normalized_prompt_hashes):
                raise ValueError(f"split leakage: forbidden normalized prompt for {case_id}")
            if tuple(identity["source_family"]) in {tuple(item) for item in self.dev_test_source_families}:
                raise ValueError(f"split leakage: forbidden source family for {case_id}")
            if identity["source_identity"] is not None and tuple(identity["source_identity"]) in {tuple(item) for item in self.dev_test_source_identities}:
                raise ValueError(f"split leakage: forbidden source identity for {case_id}")
            if any(
                _similarity(identity["profile"], profile) >= self.near_duplicate_threshold
                and identity["normalized_prompt_hash"] != hashlib.sha256("".encode()).hexdigest()
                for profile in self.dev_test_prompt_profiles
            ):
                raise ValueError(f"split leakage: semantic near-duplicate prompt for {case_id}")

    def validate_payload(self, payload: Mapping[str, Any]) -> None:
        """Validate one extractor payload without exposing held-out content."""

        manifest = payload.get("manifest")
        record = {
            "case_id": payload.get("case_id") or (manifest.get("case_id") if isinstance(manifest, Mapping) else None),
            "split": payload.get("split") or (manifest.get("split") if isinstance(manifest, Mapping) else "train"),
            "prompt_hash": payload.get("prompt_hash") or (manifest.get("prompt_hash") if isinstance(manifest, Mapping) else None),
            # Preserve source text only long enough for the local normalized /
            # semantic check.  The policy itself stores hashes and compact
            # profiles, so held-out prompt content never crosses into the
            # controller context.
            "prompt": payload.get("prompt") or payload.get("source_prompt") or (manifest.get("prompt") if isinstance(manifest, Mapping) else None),
            "source_prompt": payload.get("source_prompt") or (manifest.get("source_prompt") if isinstance(manifest, Mapping) else None),
            "source_dataset": payload.get("source_dataset"),
            "source_dimension": payload.get("source_dimension"),
            "source_index": payload.get("source_index"),
        }
        self.validate_records([record])


def build_split_access_policy(
    train_records: Iterable[Mapping[str, Any]],
    dev_records: Iterable[Mapping[str, Any]],
    test_records: Iterable[Mapping[str, Any]],
    **kwargs: Any,
) -> SplitAccessPolicy:
    return SplitAccessPolicy.from_records(train_records, dev_records, test_records, **kwargs)


__all__ = [
    "SPLIT_ACCESS_SCHEMA_VERSION",
    "SplitAccessPolicy",
    "build_split_access_policy",
    "normalize_prompt",
    "prompt_hash",
]
