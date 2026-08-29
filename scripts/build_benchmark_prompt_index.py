"""Index verbatim prompts from a public benchmark for Harness training.

The output is an execution index, not a prompt generator: ``prompt`` is copied
byte-for-byte from VBench-2.0 ``prompt_en`` and no events, entities, oracle
labels, or scene descriptions are synthesized here. DirectorAgent creates the
executable plan at run time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SOURCE_DATASET = "VBench-2.0"
SOURCE_URL = (
    "https://raw.githubusercontent.com/Vchitect/VBench/master/"
    "VBench-2.0/vbench2/VBench2_full_info.json"
)
TRAIN_DIMENSIONS = (
    "Camera_Motion",
    "Human_Interaction",
    "Motion_Order_Understanding",
    "Complex_Plot",
    "Dynamic_Spatial_Relationship",
    "Mechanics",
)
TEST_DIMENSIONS = ("Motion_Rationality", "Human_Clothes")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _json_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def _fingerprint(records: Iterable[dict[str, Any]]) -> str:
    payload = "".join(_json_line(record) for record in sorted(records, key=lambda item: item["case_id"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_source(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("benchmark source must be a JSON list of objects")
    return value


def _reference_identities(reference_roots: Iterable[str | Path]) -> tuple[set[str], set[tuple[str, int]]]:
    prompt_hashes: set[str] = set()
    source_keys: set[tuple[str, int]] = set()
    for raw_root in reference_roots:
        root = Path(raw_root)
        manifest = root / "manifest.jsonl"
        if not manifest.is_file():
            continue
        metadata: dict[str, Any] = {}
        metadata_path = root / "metadata.json"
        if metadata_path.is_file():
            try:
                value = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    metadata = value
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        default_dimension = str(metadata.get("source_dimension") or "")
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict):
                continue
            source_prompt = record.get("source_prompt") or record.get("prompt")
            if isinstance(source_prompt, str) and source_prompt:
                prompt_hashes.add(_prompt_hash(source_prompt))
            dimension = str(record.get("source_dimension") or default_dimension)
            try:
                source_index = int(record["source_index"])
            except (KeyError, TypeError, ValueError):
                continue
            source_keys.add((dimension, source_index))
    return prompt_hashes, source_keys


def _pick_spread(candidates: list[tuple[int, dict[str, Any]]], count: int) -> list[tuple[int, dict[str, Any]]]:
    if len(candidates) < count:
        raise ValueError(f"not enough unused benchmark prompts: need {count}, got {len(candidates)}")
    positions = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    return [candidates[position] for position in positions]


def _slug(dimension: str) -> str:
    return dimension.lower().replace("_", "-")


def build_index(
    source_path: str | Path,
    output_root: str | Path,
    *,
    reference_roots: Iterable[str | Path] = (),
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    source_rows = _load_source(source)
    source_sha256 = _sha256(source)
    reference_prompt_hashes, reference_source_keys = _reference_identities(reference_roots)
    selected_prompt_hashes = set(reference_prompt_hashes)
    selected_source_keys = set(reference_source_keys)
    records: list[dict[str, Any]] = []
    selected_sources: list[dict[str, Any]] = []

    families = [*TRAIN_DIMENSIONS, *TEST_DIMENSIONS]
    for family_number, dimension in enumerate(families, 1):
        candidates: list[tuple[int, dict[str, Any]]] = []
        for source_index, item in enumerate(source_rows):
            if dimension not in item.get("dimension", []):
                continue
            prompt = item.get("prompt_en")
            if not isinstance(prompt, str) or not prompt:
                continue
            key = (dimension, source_index)
            prompt_hash = _prompt_hash(prompt)
            if key in selected_source_keys or prompt_hash in selected_prompt_hashes:
                continue
            candidates.append((source_index, item))
        is_test_dimension = dimension in TEST_DIMENSIONS
        picked = _pick_spread(candidates, 10 if is_test_dimension else 20)
        for slot, (source_index, item) in enumerate(picked, 1):
            # Compute the split per record.  Keeping this as local state is
            # important: a previous family must never turn the next family
            # into a test family (or change its case-id namespace).
            record_split = "test" if is_test_dimension else ("train" if slot <= 10 else "dev")
            prompt = item["prompt_en"]
            family_id = family_number if not is_test_dimension else family_number - len(TRAIN_DIMENSIONS)
            case_id = f"vbench2-{record_split}-{family_id:02d}-{slot:02d}"
            record = {
                "case_id": case_id,
                "split": record_split,
                "protocol_family": f"vbench2-{record_split}-{family_id:02d}",
                "benchmark_prompt_only": True,
                "prompt_origin": "benchmark_verbatim",
                "prompt": prompt,
                "source_prompt": prompt,
                "prompt_hash": _prompt_hash(prompt),
                "source_dataset": SOURCE_DATASET,
                "source_file": source.name,
                "source_sha256": source_sha256,
                "source_url": SOURCE_URL,
                "source_dimension": dimension,
                "source_index": source_index,
                "source_auxiliary_info": item.get("auxiliary_info"),
                # These are execution defaults, not benchmark annotations.
                "duration_s": 10.0,
                "fps": 12,
            }
            records.append(record)
            selected_sources.append(
                {
                    "case_id": case_id,
                    "source_dimension": dimension,
                    "source_index": source_index,
                }
            )
            selected_prompt_hashes.add(record["prompt_hash"])
            selected_source_keys.add((dimension, source_index))

    records.sort(key=lambda item: item["case_id"])
    splits = {
        "train": [record["case_id"] for record in records if record["split"] == "train"],
        "dev": [record["case_id"] for record in records if record["split"] == "dev"],
        "test": [record["case_id"] for record in records if record["split"] == "test"],
    }
    (destination / "manifest.jsonl").write_text("".join(_json_line(record) for record in records), encoding="utf-8")
    (destination / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "dataset_id": "vbench2-agent-training-index-v1",
        "dataset_kind": "benchmark_prompt_index",
        "schema_version": "benchmark-prompt-index-v1",
        "generator_version": "benchmark-prompt-index-builder-v1",
        "source_dataset": SOURCE_DATASET,
        "source_file": source.name,
        "source_path": str(source.relative_to(PROJECT_ROOT)) if source.is_relative_to(PROJECT_ROOT) else str(source),
        "source_sha256": source_sha256,
        "source_url": SOURCE_URL,
        "prompt_policy": "prompt equals source prompt_en exactly; no generated or augmented prompt text",
        "annotation_policy": "no authored event/entity/oracle labels; DirectorAgent creates the executable plan",
        "selection_policy": "six benchmark dimensions provide ten train and ten dev prompts; two dimensions provide ten frozen test prompts",
        "cases": len(records),
        "splits": {name: len(values) for name, values in splits.items()},
        "round_count": 6,
        "train_families": [f"vbench2-train-{index:02d}" for index in range(1, 7)],
        "dev_families": [f"vbench2-dev-{index:02d}" for index in range(1, 7)],
        "test_families": ["vbench2-test-01", "vbench2-test-02"],
        "selected_sources": sorted(selected_sources, key=lambda item: item["case_id"]),
        "reference_roots": [str(Path(value).resolve()) for value in reference_roots],
        "fingerprint": _fingerprint(records),
        "test_policy": "test prompts are benchmark-verbatim and milestone-only; they cannot create proposals or acceptance decisions",
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/vbench-source/VBench2_full_info.json")
    parser.add_argument("--out", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--reference-root", action="append", default=[])
    args = parser.parse_args()
    report = build_index(args.source, args.out, reference_roots=args.reference_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
