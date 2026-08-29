"""Validate that a training index contains only verbatim benchmark prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_SPLITS = {"train": 60, "dev": 60, "test": 20}
EXPECTED_DATASET_KIND = "benchmark_prompt_index"
TRAIN_DIMENSIONS = (
    "Camera_Motion",
    "Human_Interaction",
    "Motion_Order_Understanding",
    "Complex_Plot",
    "Dynamic_Spatial_Relationship",
    "Mechanics",
)
TEST_DIMENSIONS = ("Motion_Rationality", "Human_Clothes")


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _fingerprint(records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(_json_line(record) for record in sorted(records, key=lambda item: item["case_id"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} is not an object")
        records.append(value)
    return records


def _reference_identities(reference_roots: Iterable[str | Path]) -> tuple[set[str], set[tuple[str, int]]]:
    prompt_hashes: set[str] = set()
    source_keys: set[tuple[str, int]] = set()
    for raw_root in reference_roots:
        root = Path(raw_root)
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            continue
        for record in _read_jsonl(manifest):
            prompt = record.get("source_prompt") or record.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompt_hashes.add(_prompt_hash(prompt))
            dimension = str(record.get("source_dimension") or "")
            try:
                source_index = int(record["source_index"])
            except (KeyError, TypeError, ValueError):
                continue
            source_keys.add((dimension, source_index))
    return prompt_hashes, source_keys


def validate_benchmark_prompt_index(
    root: str | Path = "dataset/vbench2-agent-training-index-v1",
    *,
    source_path: str | Path | None = None,
    reference_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    destination = Path(root)
    errors: list[str] = []
    metadata_path = destination / "metadata.json"
    manifest_path = destination / "manifest.jsonl"
    splits_path = destination / "splits.json"
    if not destination.is_dir():
        return {"status": "fail", "training_eligible": False, "errors": [f"index_missing:{destination}"]}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        records = _read_jsonl(manifest_path)
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "fail", "training_eligible": False, "errors": [f"index_unreadable:{type(exc).__name__}:{exc}"]}
    if not isinstance(metadata, dict) or not isinstance(splits, dict):
        return {"status": "fail", "training_eligible": False, "errors": ["metadata_or_splits_not_object"]}
    if metadata.get("dataset_kind") != EXPECTED_DATASET_KIND:
        errors.append("dataset must be a benchmark_prompt_index; self-built prompt datasets are not eligible")
    if metadata.get("source_dataset") != "VBench-2.0":
        errors.append("source_dataset must be VBench-2.0")
    if set(splits) != set(EXPECTED_SPLITS):
        errors.append(f"splits must be exactly {sorted(EXPECTED_SPLITS)}")
    counts = {name: len(values) for name, values in splits.items()}
    if counts != EXPECTED_SPLITS:
        errors.append(f"expected split counts {EXPECTED_SPLITS}, got {counts}")
    if len(records) != sum(EXPECTED_SPLITS.values()):
        errors.append(f"expected 140 benchmark records, got {len(records)}")
    source_value = source_path or metadata.get("source_path") or metadata.get("source_file")
    source = Path(str(source_value)) if source_value else PROJECT_ROOT / "data/vbench-source/VBench2_full_info.json"
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source_rows: list[dict[str, Any]] = []
    if not source.is_file():
        errors.append(f"benchmark source missing:{source}")
    else:
        try:
            source_rows = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(source_rows, list):
                raise ValueError("source is not a list")
            actual_sha256 = _sha256(source)
            if actual_sha256 != metadata.get("source_sha256"):
                errors.append("source_sha256 mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"benchmark source unreadable:{type(exc).__name__}:{exc}")
    case_ids = [str(record.get("case_id") or "") for record in records]
    if len(case_ids) != len(set(case_ids)) or "" in case_ids:
        errors.append("case IDs must be non-empty and unique")
    by_id = {record.get("case_id"): record for record in records}
    for split, ids in splits.items():
        if not isinstance(ids, list) or any(case_id not in by_id for case_id in ids):
            errors.append(f"{split} split does not reference manifest cases exactly")
        for case_id in ids if isinstance(ids, list) else []:
            if by_id.get(case_id, {}).get("split") != split:
                errors.append(f"split mismatch for {case_id}")
    if set().union(*(set(values) for values in splits.values() if isinstance(values, list))) != set(case_ids):
        errors.append("splits must cover every manifest case")
    expected_families = {
        "train": tuple(f"vbench2-train-{index:02d}" for index in range(1, len(TRAIN_DIMENSIONS) + 1)),
        "dev": tuple(f"vbench2-dev-{index:02d}" for index in range(1, len(TRAIN_DIMENSIONS) + 1)),
        "test": tuple(f"vbench2-test-{index:02d}" for index in range(1, len(TEST_DIMENSIONS) + 1)),
    }
    expected_dimensions = {
        "train": TRAIN_DIMENSIONS,
        "dev": TRAIN_DIMENSIONS,
        "test": TEST_DIMENSIONS,
    }
    for metadata_key, split in (("train_families", "train"), ("dev_families", "dev"), ("test_families", "test")):
        if metadata.get(metadata_key) != list(expected_families[split]):
            errors.append(f"{metadata_key} must match the six-round family layout")
    for split, families in expected_families.items():
        ids = splits.get(split, [])
        observed = Counter(str(case_id).rsplit("-", 1)[0] for case_id in ids) if isinstance(ids, list) else Counter()
        expected_counts = {family: 10 for family in families}
        if observed != expected_counts:
            errors.append(f"{split} case families must be exactly six/ten (test two/ten): got {dict(observed)}")
        for family_index, family in enumerate(families):
            for case_id in ids if isinstance(ids, list) else []:
                if str(case_id).rsplit("-", 1)[0] != family:
                    continue
                record = by_id.get(case_id, {})
                if record.get("protocol_family") != family:
                    errors.append(f"{case_id}:protocol_family does not match its split family")
                expected_dimension = expected_dimensions[split][family_index]
                if record.get("source_dimension") != expected_dimension:
                    errors.append(f"{case_id}:source_dimension does not match its benchmark family")
    prompt_hashes: list[str] = []
    source_keys: list[tuple[str, int]] = []
    forbidden_label_fields = {"entities", "event_graph", "required_events", "oracle_expectations", "proxy_scene", "camera_evidence"}
    for record in records:
        case_id = str(record.get("case_id") or "unknown")
        required = {"prompt", "source_prompt", "prompt_hash", "source_dimension", "source_index", "source_dataset", "source_sha256", "prompt_origin", "benchmark_prompt_only"}
        missing = sorted(required - set(record))
        if missing:
            errors.append(f"{case_id}:missing benchmark provenance fields:{missing}")
            continue
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            errors.append(f"{case_id}:prompt must be non-empty text")
            continue
        if record.get("source_prompt") != prompt:
            errors.append(f"{case_id}:prompt is not identical to source_prompt")
        if record.get("prompt_origin") != "benchmark_verbatim":
            errors.append(f"{case_id}:prompt_origin is not benchmark_verbatim")
        if record.get("benchmark_prompt_only") is not True:
            errors.append(f"{case_id}:benchmark_prompt_only must be true")
        if forbidden_label_fields & set(record):
            errors.append(f"{case_id}:contains locally authored scene/label fields")
        expected_hash = _prompt_hash(prompt)
        if record.get("prompt_hash") != expected_hash:
            errors.append(f"{case_id}:prompt_hash mismatch")
        prompt_hashes.append(expected_hash)
        try:
            source_index = int(record["source_index"])
        except (TypeError, ValueError):
            errors.append(f"{case_id}:source_index is not an integer")
            continue
        dimension = str(record["source_dimension"])
        source_keys.append((dimension, source_index))
        if source_rows:
            if source_index < 0 or source_index >= len(source_rows):
                errors.append(f"{case_id}:source_index outside benchmark source")
            else:
                source_item = source_rows[source_index]
                if source_item.get("prompt_en") != prompt:
                    errors.append(f"{case_id}:prompt differs from raw VBench prompt_en")
                if dimension not in source_item.get("dimension", []):
                    errors.append(f"{case_id}:source_dimension not present at source_index")
    if len(prompt_hashes) != len(set(prompt_hashes)):
        errors.append("benchmark prompts must be unique")
    if len(source_keys) != len(set(source_keys)):
        errors.append("benchmark source identities must be unique")
    if metadata.get("fingerprint") != _fingerprint(records):
        errors.append("metadata fingerprint does not match manifest")
    reference_checks = []
    for reference in reference_roots:
        reference_prompt_hashes, reference_source_keys = _reference_identities([reference])
        overlaps = {
            "prompt_hash": sorted(set(prompt_hashes) & reference_prompt_hashes),
            "source_identity": sorted(set(source_keys) & reference_source_keys, key=str),
        }
        overlaps = {kind: values for kind, values in overlaps.items() if values}
        reference_checks.append({"reference_root": str(Path(reference).resolve()), "overlaps": overlaps, "status": "fail" if overlaps else "pass"})
        if overlaps:
            errors.append(f"benchmark overlap with reference {reference}: {overlaps}")
    return {
        "status": "pass" if not errors else "fail",
        "training_eligible": not errors,
        "dataset_id": metadata.get("dataset_id"),
        "dataset_kind": metadata.get("dataset_kind"),
        "case_count": len(records),
        "splits": counts,
        "unique_prompt_hashes": len(set(prompt_hashes)),
        "fingerprint": metadata.get("fingerprint"),
        "source": str(source.resolve()),
        "reference_checks": reference_checks,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "--dataset-root", dest="root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--source")
    parser.add_argument("--reference-root", action="append", default=[])
    args = parser.parse_args()
    report = validate_benchmark_prompt_index(args.root, source_path=args.source, reference_roots=args.reference_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
