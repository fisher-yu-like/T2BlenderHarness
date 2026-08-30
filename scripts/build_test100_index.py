"""Build the 100-case test index for per-round generalization scoring.

Selects 100 verbatim VBench-2.0 prompts unused by the training index
(10 dimensions x 10 prompts: the six training dimensions plus Dynamic_Attribute,
Human_Anatomy, Motion_Rationality, Human_Clothes), emitting the same
benchmark-prompt-index schema (manifest.jsonl, splits.json, metadata.json).
No authored event/entity/oracle labels are added.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "vbench-source" / "VBench2_full_info.json"
TRAINING_INDEX = ROOT / "dataset" / "vbench2-agent-training-index-v1" / "manifest.jsonl"
OUT = ROOT / "dataset" / "vbench2-agent-test-100-v1"
DATASET_ID = "vbench2-agent-test-100-v1"
SOURCE_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/VBench-2.0/vbench2/VBench2_full_info.json"

DIMENSION_PLAN = [
    "Camera_Motion",
    "Human_Interaction",
    "Motion_Order_Understanding",
    "Complex_Plot",
    "Dynamic_Spatial_Relationship",
    "Mechanics",
    "Dynamic_Attribute",
    "Human_Anatomy",
    "Motion_Rationality",
    "Human_Clothes",
]
PER_DIMENSION = 10


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def main() -> int:
    src = json.loads(SOURCE.read_text(encoding="utf-8"))
    used = {
        json.loads(line)["prompt"]
        for line in TRAINING_INDEX.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    by_dimension: dict[str, list[dict]] = defaultdict(list)
    for index, item in enumerate(src):
        prompt = item.get("prompt_en")
        dimension = (item.get("dimension") or [""])[0]
        if prompt in used:
            continue
        by_dimension[dimension].append({"index": index, "item": item})
    selected: list[tuple[str, dict]] = []
    selected_prompts: set[str] = set()
    for dimension in DIMENSION_PLAN:
        taken = 0
        for entry in by_dimension.get(dimension, []):
            if taken >= PER_DIMENSION:
                break
            prompt = entry["item"]["prompt_en"]
            # The same prompt can appear under several dimensions in the
            # source; keep each prompt exactly once in the index.
            if prompt in selected_prompts:
                continue
            selected_prompts.add(prompt)
            selected.append((dimension, entry))
            taken += 1
        if taken < PER_DIMENSION:
            raise SystemExit(f"not enough unused unique prompts for {dimension}: {taken}")
    if len(selected) != 100:
        raise SystemExit(f"expected 100 selected prompts, got {len(selected)}")

    OUT.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(SOURCE)
    manifest_lines: list[str] = []
    case_ids: list[str] = []
    per_family: dict[str, int] = defaultdict(int)
    for dimension, entry in selected:
        item = entry["item"]
        per_family[dimension] += 1
        family_id = f"{DIMENSION_PLAN.index(dimension) + 1:02d}"
        case_id = f"vbench2-test100-{family_id}-{per_family[dimension]:02d}"
        case_ids.append(case_id)
        record = {
            "benchmark_prompt_only": True,
            "case_id": case_id,
            "duration_s": 10.0,
            "fps": 12,
            "prompt": item["prompt_en"],
            "prompt_hash": prompt_hash(item["prompt_en"]),
            "prompt_origin": "benchmark_verbatim",
            "protocol_family": f"vbench2-test100-{family_id}",
            "source_auxiliary_info": item.get("auxiliary_info"),
            "source_dataset": "VBench-2.0",
            "source_dimension": dimension,
            "source_index": entry["index"],
            "source_file": "VBench2_full_info.json",
            "source_prompt": item["prompt_en"],
            "source_sha256": source_sha,
            "source_url": SOURCE_URL,
            "split": "test",
        }
        manifest_lines.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
    (OUT / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    (OUT / "splits.json").write_text(
        json.dumps({"test": case_ids}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    records = [json.loads(line) for line in manifest_lines]
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: item["case_id"])
    )
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    metadata = {
        "annotation_policy": "no authored event/entity/oracle labels; DirectorAgent creates the executable plan",
        "cases": len(case_ids),
        "dataset_id": DATASET_ID,
        "dataset_kind": "benchmark_prompt_index",
        "dimension_plan": DIMENSION_PLAN,
        "fingerprint": fingerprint,
        "generator_version": "benchmark-prompt-index-builder-v1",
        "prompt_policy": "prompt equals source prompt_en exactly; no generated or augmented prompt text",
        "purpose": "per-round generalization scoring for the glm-5.3-flash assistant-session training",
        "schema_version": "benchmark-prompt-index-v1",
        "source_dataset": "VBench-2.0",
        "source_file": "VBench2_full_info.json",
        "source_sha256": source_sha,
        "split_policy": "single frozen test split; prompts disjoint from vbench2-agent-training-index-v1",
        "test_families": [f"vbench2-test100-{i + 1:02d}" for i in range(len(DIMENSION_PLAN))],
    }
    # Self-checks for this standalone test index (the shared validator
    # enforces the six-round training shape and therefore does not apply).
    checks = {
        "cases_100": len(case_ids) == 100,
        "unique_prompts": len({json.loads(line)["prompt"] for line in manifest_lines}) == 100,
        "disjoint_from_training": not (selected_prompts & used),
        "verbatim_source_prompts": all(
            record["prompt"] == src[record["source_index"]]["prompt_en"]
            for record in records
        ),
        "single_test_split": all(record["split"] == "test" for record in records),
        "source_sha_matches": source_sha == sha256_file(SOURCE),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"test index self-checks failed: {failed}")
    (OUT / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cases": len(case_ids), "dataset_id": DATASET_ID, "fingerprint": fingerprint[:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
