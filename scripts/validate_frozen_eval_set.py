"""Validate an independent frozen evaluation manifest.

This validator checks identity and leakage boundaries only.  It never reads
runtime scores and never exposes a proposal owner, so the frozen set cannot
silently become a training signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_FROZEN_DATASET_IDS = {"frozen-eval-v1", "frozen-eval-v2"}

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: str(item.get("case_id", "")))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _coverage_matrix(
    records: list[dict[str, Any]], metadata: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Materialize declared evaluation slices and enforce their minimum size."""

    slices = metadata.get("evaluation_slices")
    if not isinstance(slices, dict) or not slices:
        return {}, []
    matrix: dict[str, Any] = {}
    errors: list[str] = []
    for slice_name, raw_slice in sorted(slices.items()):
        if not isinstance(raw_slice, dict):
            errors.append(f"evaluation slice {slice_name} must be an object")
            continue
        dimensions = [str(value) for value in raw_slice.get("dimensions", []) or []]
        try:
            declared_case_count = int(raw_slice.get("case_count"))
            minimum_case_count = int(raw_slice.get("min_case_count") or 1)
        except (TypeError, ValueError) as exc:
            errors.append(f"evaluation slice {slice_name} has invalid case-count metadata")
            continue
        if declared_case_count < 1 or minimum_case_count < 1:
            errors.append(f"evaluation slice {slice_name} must declare positive case counts")
        case_ids = raw_slice.get("case_ids")
        if case_ids is None:
            scoped_records = records
        elif isinstance(case_ids, list):
            requested_ids = {str(value) for value in case_ids}
            scoped_records = [record for record in records if str(record.get("case_id")) in requested_ids]
            if len(scoped_records) != len(requested_ids):
                errors.append(f"evaluation slice {slice_name} references unknown case IDs")
        else:
            errors.append(f"evaluation slice {slice_name} case_ids must be a list")
            scoped_records = []
        dimension_counts = Counter(
            str(record.get("source_dimension") or record.get("category") or "uncategorized")
            for record in scoped_records
        )
        missing_dimensions = sorted(set(dimensions) - set(dimension_counts))
        below_minimum = {
            dimension: int(dimension_counts.get(dimension, 0))
            for dimension in dimensions
            if dimension_counts.get(dimension, 0) < minimum_case_count
        }
        actual_case_count = len(scoped_records)
        slice_status = "pass"
        if actual_case_count != declared_case_count or missing_dimensions or below_minimum:
            slice_status = "fail"
            errors.append(
                f"evaluation slice {slice_name} coverage mismatch: "
                f"actual={actual_case_count}, declared={declared_case_count}, "
                f"missing={missing_dimensions}, below_minimum={below_minimum}"
            )
        matrix[slice_name] = {
            "status": slice_status,
            "case_count": actual_case_count,
            "declared_case_count": declared_case_count,
            "min_case_count": minimum_case_count,
            "dimensions": dimensions,
            "dimension_counts": {
                dimension: int(dimension_counts[dimension])
                for dimension in sorted(dimension_counts)
            },
            "missing_dimensions": missing_dimensions,
            "below_minimum": below_minimum,
        }
    return matrix, errors


def _reference_identity_sets(
    records: list[dict[str, Any]], metadata: dict[str, Any], *, fallback_dataset: str
) -> dict[str, set[Any]]:
    """Extract comparable identities from current and historical manifests."""

    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    source_identities: set[tuple[str, int]] = set()
    semantic_signatures: set[str] = set()
    default_dataset = str(metadata.get("dataset_id") or fallback_dataset)
    for index, record in enumerate(records):
        case_ids.add(str(record.get("case_id", "")))
        prompt = str(record.get("prompt", ""))
        prompt_hashes.add(str(record.get("prompt_hash") or hashlib.sha256(prompt.encode("utf-8")).hexdigest()))
        source_dataset = str(record.get("source_dataset") or default_dataset)
        try:
            source_index = int(record.get("source_index", index))
        except (TypeError, ValueError):
            source_index = index
        source_identities.add((source_dataset, source_index))
        semantic = record.get("semantic_signature") or record.get("composition_signature")
        if not semantic:
            semantic_payload = {
                "category": record.get("category"),
                "template_family": record.get("template_family"),
                "required_events": record.get("required_events"),
                "camera": (record.get("proxy_scene") or {}).get("camera"),
            }
            semantic = hashlib.sha256(
                json.dumps(semantic_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        semantic_signatures.add(str(semantic))
    return {
        "case_id": case_ids,
        "prompt_hash": prompt_hashes,
        "source_identity": source_identities,
        "semantic_signature": semantic_signatures,
    }


def validate_frozen_eval_set(
    root: str | Path = "dataset/frozen-eval-v1",
    *,
    reference_roots: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    destination = Path(root)
    required = ("manifest.jsonl", "metadata.json")
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        raise ValueError(f"missing frozen-eval files: {missing}")
    records = _read_jsonl(destination / "manifest.jsonl")
    metadata = json.loads((destination / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("frozen-eval metadata must be an object")
    if metadata.get("dataset_id") not in SUPPORTED_FROZEN_DATASET_IDS:
        raise ValueError(f"metadata dataset_id must be one of {sorted(SUPPORTED_FROZEN_DATASET_IDS)}")
    expected_count = int(metadata.get("case_count") or len(records))
    if len(records) != expected_count or len(records) < 10:
        raise ValueError(f"frozen-eval case count must be at least 10 and match metadata: {len(records)}")

    required_fields = {"case_id", "prompt", "prompt_hash", "source_dataset", "source_index", "semantic_signature"}
    ids: list[str] = []
    prompt_hashes: list[str] = []
    source_keys: list[tuple[str, int]] = []
    signatures: list[str] = []
    for record in records:
        missing_fields = sorted(required_fields - set(record))
        case_id = str(record.get("case_id", ""))
        if missing_fields:
            raise ValueError(f"frozen case {case_id} missing fields: {missing_fields}")
        prompt = str(record["prompt"])
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if record["prompt_hash"] != expected_hash:
            raise ValueError(f"prompt hash mismatch for frozen case {case_id}")
        try:
            source_index = int(record["source_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source_index must be an integer for {case_id}") from exc
        if source_index < 0:
            raise ValueError(f"source_index must be non-negative for {case_id}")
        ids.append(case_id)
        prompt_hashes.append(expected_hash)
        source_keys.append((str(record["source_dataset"]), source_index))
        signatures.append(str(record["semantic_signature"]))
        if not case_id or not str(record["source_dataset"]).strip() or not signatures[-1].strip():
            raise ValueError(f"frozen case {case_id} has empty identity metadata")

    if len(ids) != len(set(ids)):
        raise ValueError("frozen case IDs must be unique")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        raise ValueError("frozen prompt hashes must be unique")
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("frozen source identities must be unique")
    if len(signatures) != len(set(signatures)):
        raise ValueError("frozen semantic signatures must be unique")

    train_dev_hashes = {str(value) for value in metadata.get("train_dev_prompt_hashes", [])}
    overlap = sorted(set(prompt_hashes) & train_dev_hashes)
    if overlap:
        raise ValueError(f"frozen prompt hash overlap with train/dev: {overlap}")
    train_dev_sources = {
        (str(item.get("source_dataset")), int(item["source_index"]))
        for item in metadata.get("train_dev_source_identities", [])
        if isinstance(item, dict) and "source_dataset" in item and "source_index" in item
    }
    source_overlap = sorted(set(source_keys) & train_dev_sources)
    if source_overlap:
        raise ValueError(f"frozen source identity overlap with train/dev: {source_overlap}")
    if metadata.get("patch_selection_allowed") is not False:
        raise ValueError("frozen evaluation must declare patch_selection_allowed=false")
    if metadata.get("proposal_generation_allowed") is not False:
        raise ValueError("frozen evaluation must declare proposal_generation_allowed=false")
    if metadata.get("split") != "frozen_eval":
        raise ValueError("frozen evaluation must declare split=frozen_eval")
    if metadata.get("fingerprint") != _fingerprint(records):
        raise ValueError("frozen metadata fingerprint does not match manifest")
    slices = metadata.get("evaluation_slices")
    if metadata.get("dataset_id") == "frozen-eval-v2":
        ood = slices.get("ood_unseen_dimensions") if isinstance(slices, dict) else None
        if not isinstance(ood, dict) or int(ood.get("case_count") or 0) != len(records):
            raise ValueError("frozen-eval-v2 must declare a complete OOD unseen-dimensions slice")
        if not isinstance(ood.get("dimensions"), list) or not ood["dimensions"]:
            raise ValueError("frozen-eval-v2 OOD slice must declare source dimensions")
        if int(ood.get("min_case_count") or 0) < 1:
            raise ValueError("frozen-eval-v2 OOD slice must declare a positive minimum case count")
    coverage_matrix, coverage_errors = _coverage_matrix(records, metadata)
    frozen_identity_sets = {
        "case_id": set(ids),
        "prompt_hash": set(prompt_hashes),
        "source_identity": set(source_keys),
        "semantic_signature": set(signatures),
    }
    reference_checks: list[dict[str, Any]] = []
    reference_errors: list[str] = []
    reference_records_by_name: dict[str, list[dict[str, Any]]] = {}
    for reference in reference_roots or ():
        reference_path = Path(reference)
        manifest_path = reference_path / "manifest.jsonl"
        metadata_path = reference_path / "metadata.json"
        if not manifest_path.is_file():
            reference_errors.append(f"reference manifest missing: {reference_path}")
            continue
        reference_records = _read_jsonl(manifest_path)
        reference_records_by_name[str(reference_path.resolve())] = reference_records
        reference_metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded_metadata, dict):
                reference_metadata = loaded_metadata
        reference_identities = _reference_identity_sets(
            reference_records,
            reference_metadata,
            fallback_dataset=reference_path.name,
        )
        overlaps = {
            kind: sorted(frozen_identity_sets[kind] & reference_identities[kind], key=str)
            for kind in frozen_identity_sets
            if frozen_identity_sets[kind] & reference_identities[kind]
        }
        check = {
            "reference_root": str(reference_path.resolve()),
            "reference_dataset": str(reference_metadata.get("dataset_id") or reference_path.name),
            "case_count": len(reference_records),
            "overlaps": overlaps,
            "status": "fail" if overlaps else "pass",
        }
        reference_checks.append(check)
        for kind, values in overlaps.items():
            reference_errors.append(
                f"{kind} overlap with reference {check['reference_dataset']}: {values}"
            )
    from videoact.dataset_leakage import audit_leakage

    leakage = audit_leakage(records, reference_records_by_name)
    if leakage.get("status") != "pass":
        reference_errors.append("four-layer leakage audit detected collisions")
    reference_errors.extend(coverage_errors)
    status = "fail" if reference_errors else "pass"
    return {
        "status": status,
        "dataset_id": metadata["dataset_id"],
        "case_count": len(records),
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "unique_source_identities": len(set(source_keys)),
        "unique_semantic_signatures": len(set(signatures)),
        "patch_selection_allowed": metadata.get("patch_selection_allowed"),
        "proposal_generation_allowed": metadata.get("proposal_generation_allowed"),
        "fingerprint": metadata["fingerprint"],
        "coverage_matrix": coverage_matrix,
        "reference_checks": reference_checks,
        "leakage_audit": leakage,
        "errors": reference_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "--dataset-root", dest="root", default="dataset/frozen-eval-v1")
    parser.add_argument(
        "--reference-root",
        action="append",
        default=[],
        help="historical/training dataset root to compare for identity leakage; repeatable",
    )
    args = parser.parse_args()
    report = validate_frozen_eval_set(args.root, reference_roots=args.reference_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
