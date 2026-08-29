"""Prepare immutable one-step raw-prompt Blender jobs for the ablation arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blender.direct_prompt_code import build_direct_spec, compile_direct_prompt_job


DEFAULT_RENDER_SETTINGS = {
    "engine": "BLENDER_EEVEE_NEXT",
    "resolution": [128, 128],
    "samples": 1,
}


def _hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _records(dataset_root: str | Path, split: str) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [row for row in rows if row.get("split") == split]


def _manifest(
    record: dict[str, Any],
    *,
    plan_hash: str,
    harness_version: str,
    evaluator_version: str,
    render_settings: dict[str, Any],
) -> dict[str, Any]:
    prompt_hash = _hash_prompt(record["prompt"])
    return {
        "run_id": f"direct-{record['case_id']}-{prompt_hash[:10]}",
        "case_id": record["case_id"],
        "split": record.get("split", "train"),
        "prompt_hash": prompt_hash,
        "plan_hash": plan_hash,
        "director_plan_hash": None,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "blender_version": "pending-direct-code",
        "fps": int(record["fps"]),
        "frame_start": 1,
        "frame_end": max(2, round(float(record["duration_s"]) * int(record["fps"]))),
        "render_settings": render_settings,
        "fingerprint": _hash_payload(
            {
                "prompt_hash": prompt_hash,
                "plan_hash": plan_hash,
                "director_plan_hash": None,
                "harness_version": harness_version,
                "evaluator_version": evaluator_version,
                "blender_version": "pending-direct-code",
                "render_settings": render_settings,
            }
        ),
        "state": "prepared",
    }


def prepare_direct_job(
    record: dict[str, Any],
    out_dir: str | Path,
    *,
    harness_version: str = "direct-prompt-code-v1",
    evaluator_version: str = "three-arm-v1",
    render_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(out_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"direct run directory already contains artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    settings = {**DEFAULT_RENDER_SETTINGS, **(render_settings or {})}
    spec = build_direct_spec(
        record["prompt"],
        duration_s=float(record["duration_s"]),
        fps=int(record["fps"]),
        seed=int(record.get("proxy_scene", {}).get("scene_seed", 0) or 0),
    )
    plan_hash = _hash_payload(spec)
    manifest = _manifest(
        record,
        plan_hash=plan_hash,
        harness_version=harness_version,
        evaluator_version=evaluator_version,
        render_settings=settings,
    )
    source, code_metadata = compile_direct_prompt_job(
        record["prompt"],
        case_id=record["case_id"],
        output_dir=output,
        duration_s=float(record["duration_s"]),
        fps=int(record["fps"]),
        seed=int(record.get("proxy_scene", {}).get("scene_seed", 0) or 0),
        render_settings=settings,
        manifest=manifest,
    )
    (output / "direct_code_manifest.json").write_text(
        json.dumps({**code_metadata, "plan_hash": plan_hash}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "scene_contract.json").write_text(
        json.dumps(
            {
                "planning_mode": "direct_prompt_code",
                "prompt": record["prompt"],
                "fps": spec["fps"],
                "entities": [
                    {"id": entity_id, "kind": data["kind"], "label": data["label"]}
                    for entity_id, data in spec["entities"].items()
                ],
                "events": [],
                "must_show": [],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "trajectory.json").write_text(
        json.dumps({"planning_mode": "direct_prompt_code", "entities": spec["entities"], "camera": {"shots": []}}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "camera_plan.json").write_text(
        json.dumps({"planning_mode": "direct_prompt_code", "shots": []}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Keep the dataset's authored geometry policy available to the independent
    # Blender audit.  The direct arm is not allowed to consume event_graph or
    # oracle labels when compiling code, but the evaluator may still test the
    # resulting .blend against the same realism gate.
    proxy_scene = record.get("proxy_scene") or {}
    (output / "proxy_scene.json").write_text(
        json.dumps(proxy_scene, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    job_path = output / "blender_job.py"
    job_path.write_text(source, encoding="utf-8")
    return {
        "case_id": record["case_id"],
        "run_dir": str(output),
        "job_path": str(job_path),
        "planning_mode": "direct_prompt_code",
        "plan_hash": plan_hash,
    }


def prepare_split(
    split: str,
    out_dir: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str = "direct-prompt-code-v1",
    evaluator_version: str = "three-arm-v1",
    render_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(out_dir)
    jobs = [
        prepare_direct_job(
            record,
            output / record["case_id"],
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            render_settings=render_settings,
        )
        for record in _records(dataset_root, split)
    ]
    index = {
        "split": split,
        "planning_mode": "direct_prompt_code",
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "case_count": len(jobs),
        "jobs": jobs,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "job_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-root", default="dataset/vbench-derived-100-v1")
    parser.add_argument("--harness-version", default="direct-prompt-code-v1")
    parser.add_argument("--evaluator-version", default="three-arm-v1")
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    settings: dict[str, Any] = {}
    if args.resolution:
        settings["resolution"] = args.resolution
    if args.samples is not None:
        settings["samples"] = args.samples
    print(json.dumps(prepare_split(args.split, args.out_dir, dataset_root=args.dataset_root, harness_version=args.harness_version, evaluator_version=args.evaluator_version, render_settings=settings), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
