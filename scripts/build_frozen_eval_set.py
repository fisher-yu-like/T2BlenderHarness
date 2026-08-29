"""Build an independent frozen evaluation manifest.

The source is the raw VBench corpus (or an explicitly supplied manifest), not
the already-derived 100-case comparison set.  Reference datasets are used only
to exclude identities; this script never reads scores or creates Harness
proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from videoact.dataset_leakage import cosine_similarity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_metadata(root: Path) -> dict[str, Any]:
    path = root / "metadata.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_source(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load either a JSONL dataset manifest or the raw VBench JSON list."""

    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("raw source JSON must be a list")
        records: list[dict[str, Any]] = []
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt_en") or item.get("prompt") or "").strip()
            dimensions = item.get("dimension")
            if not prompt or not isinstance(dimensions, list) or not dimensions:
                continue
            records.append(
                {
                    "prompt": prompt,
                    "source_prompt": prompt,
                    "source_dataset": "VBench-2.0",
                    "source_index": index,
                    "source_dimension": str(dimensions[0]),
                    "source_auxiliary_info": item.get("auxiliary_info"),
                }
            )
        return records, "raw_json"
    manifest = path / "manifest.jsonl"
    if not manifest.is_file():
        raise ValueError(f"source manifest missing: {manifest}")
    return _read_jsonl(manifest), "dataset_manifest"


def _fingerprint(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: item["case_id"])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_sets(
    records: list[dict[str, Any]], metadata: dict[str, Any], *, fallback_dataset: str
) -> dict[str, set[Any]]:
    case_ids: set[str] = set()
    prompt_hashes: set[str] = set()
    source_identities: set[tuple[str, int]] = set()
    source_dimensions: set[str] = set()
    default_dataset = str(metadata.get("dataset_id") or fallback_dataset)
    for index, record in enumerate(records):
        case_id = str(record.get("case_id") or "")
        if case_id:
            case_ids.add(case_id)
        prompt = str(record.get("prompt") or record.get("source_prompt") or "")
        if record.get("prompt_hash"):
            prompt_hashes.add(str(record["prompt_hash"]))
        if prompt:
            prompt_hashes.add(hashlib.sha256(prompt.encode("utf-8")).hexdigest())
        source_dataset = str(
            record.get("source_dataset")
            or ("VBench-2.0" if "vbench" in default_dataset.casefold() else default_dataset)
        )
        try:
            source_index = int(record.get("source_index", index))
        except (TypeError, ValueError):
            source_index = index
        source_identities.add((source_dataset, source_index))
        dimension = record.get("source_dimension") or record.get("category")
        if dimension:
            source_dimensions.add(str(dimension))
    return {
        "case_id": case_ids,
        "prompt_hash": prompt_hashes,
        "source_identity": source_identities,
        "source_dimension": source_dimensions,
    }


def _collect_exclusions(
    roots: Sequence[str | Path],
) -> tuple[dict[str, set[Any]], int, list[str]]:
    exclusions = {
        "case_id": set(),
        "prompt_hash": set(),
        "source_identity": set(),
        "source_dimension": set(),
    }
    record_count = 0
    datasets: list[str] = []
    for raw_root in roots:
        root = Path(raw_root)
        manifest = root if root.is_file() else root / "manifest.jsonl"
        if not manifest.is_file():
            raise ValueError(f"reference manifest missing: {manifest}")
        metadata_root = root.parent if root.is_file() else root
        metadata = _read_metadata(metadata_root)
        records = _read_jsonl(manifest)
        identities = _identity_sets(records, metadata, fallback_dataset=metadata_root.name)
        for key in exclusions:
            exclusions[key].update(identities[key])
        record_count += len(records)
        datasets.append(str(metadata.get("dataset_id") or metadata_root.name))
    return exclusions, record_count, datasets


def _source_dimension(record: dict[str, Any]) -> str:
    dimension = record.get("source_dimension") or record.get("category")
    if dimension:
        return str(dimension)
    dimensions = record.get("dimension")
    if isinstance(dimensions, list) and dimensions:
        return str(dimensions[0])
    return "uncategorized"


def _source_identity(record: dict[str, Any], index: int) -> tuple[str, int]:
    source_dataset = str(record.get("source_dataset") or "VBench-2.0")
    try:
        source_index = int(record.get("source_index", index))
    except (TypeError, ValueError):
        source_index = index
    return source_dataset, source_index


def build_frozen_eval_set(
    *,
    source_root: str | Path = "data/vbench-source/VBench2_full_info.json",
    training_root: str | Path = "dataset/trajectory-v5-agent-codegen",
    reference_roots: Sequence[str | Path] | None = None,
    output_root: str | Path = "dataset/frozen-eval-v1",
    dataset_id: str = "frozen-eval-v1",
    case_id_prefix: str | None = None,
    per_category: int = 4,
    category_count: int = 5,
) -> dict[str, Any]:
    source = Path(source_root)
    source_records, source_kind = _load_source(source)
    if per_category <= 0 or category_count <= 0:
        raise ValueError("per_category and category_count must be positive")

    default_references: list[str | Path] = []
    if reference_roots is not None:
        default_references = list(reference_roots)
    elif source_kind == "raw_json":
        for candidate in ("dataset/vbench-derived-100-v1", "dataset/trajectory-v4-multi"):
            if (ROOT / candidate / "manifest.jsonl").is_file():
                default_references.append(ROOT / candidate)
    exclusions, excluded_reference_count, excluded_datasets = _collect_exclusions(default_references)

    training_path = Path(training_root)
    training_records: list[dict[str, Any]] = []
    if training_path.is_file():
        training_records = _read_jsonl(training_path)
    elif (training_path / "manifest.jsonl").is_file():
        training_records = _read_jsonl(training_path / "manifest.jsonl")
    training_metadata = _read_metadata(training_path if training_path.is_dir() else training_path.parent)
    training_exclusions = _identity_sets(
        training_records,
        training_metadata,
        fallback_dataset=training_path.name,
    )
    for key in exclusions:
        exclusions[key].update(training_exclusions[key])

    eligible = source_records
    if source_kind == "dataset_manifest" and any(record.get("split") for record in source_records):
        eligible = [record for record in source_records if record.get("split") == "dev"]

    source_dataset_names = {str(record.get("source_dataset") or "VBench-2.0") for record in eligible}
    candidates_by_dimension: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, record in enumerate(eligible):
        identity = _source_identity(record, index)
        prompt = str(record.get("prompt") or record.get("source_prompt") or "").strip()
        prompt_hashes = {
            str(record.get("prompt_hash")),
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else "",
        }
        dimension = _source_dimension(record)
        if identity in exclusions["source_identity"]:
            continue
        if prompt_hashes & exclusions["prompt_hash"]:
            continue
        if identity[0] in source_dataset_names and dimension in exclusions["source_dimension"]:
            continue
        candidates_by_dimension.setdefault(dimension, []).append((index, record))

    selected: list[tuple[int, dict[str, Any]]] = []
    selected_dimensions: list[str] = []
    selected_prompt_hashes: set[str] = set()
    selected_prompts: list[str] = []
    for dimension in sorted(candidates_by_dimension):
        candidates = sorted(candidates_by_dimension[dimension], key=lambda item: item[0])
        unique_candidates: list[tuple[int, dict[str, Any]]] = []
        for source_position, candidate in candidates:
            prompt = str(candidate.get("prompt") or candidate.get("source_prompt") or "").strip()
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else ""
            if prompt_hash and prompt_hash in selected_prompt_hashes:
                continue
            unique_candidates.append((source_position, candidate))
        if len(unique_candidates) < per_category:
            continue
        selected_dimensions.append(dimension)
        chosen: list[tuple[int, dict[str, Any]]] = []
        for source_position, candidate in unique_candidates:
            candidate_prompt = str(candidate.get("prompt") or candidate.get("source_prompt") or "").strip()
            if any(cosine_similarity(candidate_prompt, previous) >= 0.92 for previous in selected_prompts):
                continue
            chosen.append((source_position, candidate))
            selected_prompts.append(candidate_prompt)
            if len(chosen) >= per_category:
                break
        if len(chosen) < per_category:
            continue
        selected.extend(chosen)
        selected_prompt_hashes.update(
            hashlib.sha256(
                str(candidate.get("prompt") or candidate.get("source_prompt") or "").strip().encode("utf-8")
            ).hexdigest()
            for _, candidate in chosen
        )
        if len(selected_dimensions) == category_count:
            break
    if len(selected_dimensions) < category_count:
        raise ValueError(
            f"not enough independent source dimensions: need {category_count}, got {len(selected_dimensions)}"
        )
    if len(selected) < 10:
        raise ValueError("independent frozen source is too small")

    records: list[dict[str, Any]] = []
    effective_case_id_prefix = case_id_prefix or ("frozen-vbench" if dataset_id == "frozen-eval-v1" else dataset_id)
    for ordinal, (source_position, source_record) in enumerate(selected, 1):
        prompt = str(source_record.get("prompt") or source_record.get("source_prompt") or "").strip()
        source_prompt = str(source_record.get("source_prompt") or prompt)
        source_dataset, source_index = _source_identity(source_record, source_position)
        source_dimension = _source_dimension(source_record)
        records.append(
            {
                "case_id": f"{effective_case_id_prefix}-{ordinal:03d}",
                "split": "frozen_eval",
                "prompt": prompt,
                "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "source_dataset": source_dataset,
                "source_index": source_index,
                "source_dimension": source_dimension,
                "source_prompt": source_prompt,
                "source_case_id": source_record.get("case_id"),
                "semantic_signature": hashlib.sha256(
                    f"{source_dataset}:{source_index}:{source_dimension}:{source_prompt}".encode("utf-8")
                ).hexdigest(),
            }
        )

    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.jsonl").write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    training_hashes = sorted(training_exclusions["prompt_hash"])
    training_sources = [
        {"source_dataset": dataset, "source_index": index}
        for dataset, index in sorted(training_exclusions["source_identity"])
    ]
    metadata = {
        "dataset_id": dataset_id,
        "schema_version": "frozen-eval-manifest-v1",
        "split": "frozen_eval",
        "case_count": len(records),
        "source_dataset": "VBench-2.0-unused-dimension-holdout",
        "source_policy": "raw VBench source; excludes all explicitly referenced dataset identities, prompt hashes, and used dimensions",
        "source_root": str(source.resolve()),
        "source_kind": source_kind,
        "selected_dimensions": selected_dimensions,
        "evaluation_slices": {
            "ood_unseen_dimensions": {
                "dimensions": selected_dimensions,
                "case_count": len(records),
                "min_case_count": per_category,
                "selection_rule": "source dimensions absent from active train/dev and prior frozen references",
            }
        },
        "excluded_reference_datasets": excluded_datasets,
        "excluded_reference_count": excluded_reference_count,
        "train_dev_prompt_hashes": training_hashes,
        "train_dev_source_identities": training_sources,
        "patch_selection_allowed": False,
        "proposal_generation_allowed": False,
        "fingerprint": _fingerprint(records),
    }
    (destination / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "pass",
        "dataset_id": metadata["dataset_id"],
        "case_count": len(records),
        "fingerprint": metadata["fingerprint"],
        "selected_dimensions": selected_dimensions,
        "excluded_reference_count": excluded_reference_count,
        "output_root": str(destination.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="data/vbench-source/VBench2_full_info.json")
    parser.add_argument("--training-root", default="dataset/trajectory-v5-agent-codegen")
    parser.add_argument("--reference-root", action="append", default=[])
    parser.add_argument("--out", default="dataset/frozen-eval-v1")
    parser.add_argument("--dataset-id", default="frozen-eval-v1")
    parser.add_argument("--case-id-prefix")
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--category-count", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            build_frozen_eval_set(
                source_root=args.source_root,
                training_root=args.training_root,
                reference_roots=args.reference_root or None,
                output_root=args.out,
                dataset_id=args.dataset_id,
                case_id_prefix=args.case_id_prefix,
                per_category=args.per_category,
                category_count=args.category_count,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
