"""Prepare immutable Blender MCP jobs for real proxy rendering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blender.real_proxy_job import compile_real_proxy_job
from videoact.director import DirectorAgent
from videoact.real_artifacts import RealRunManifest, fingerprint_real_run
from videoact.run_manifest import hash_payload, hash_prompt, write_manifest


RENDER_SETTINGS = {"engine": "BLENDER_EEVEE_NEXT", "resolution": [256, 256], "samples": 16}


def _load_records(dataset_root: str | Path, split: str, case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {record["case_id"]: record for record in records}
    if split == "calibration":
        labels = [json.loads(line) for line in (root / "labels.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        ids = [label["case_id"] for label in labels]
    else:
        ids = json.loads((root / "splits.json").read_text(encoding="utf-8"))[split]
    if case_ids is not None:
        unknown = set(case_ids) - set(ids)
        if unknown:
            raise ValueError(f"case IDs are not all in {split} split: {sorted(unknown)}")
        ids = case_ids
    return [by_id[case_id] for case_id in ids]


def prepare_jobs(
    split: str,
    out_dir: str | Path,
    *,
    dataset_root: str | Path = "dataset",
    harness_version: str = "h1-real",
    evaluator_version: str = "deterministic-v1",
    case_ids: list[str] | None = None,
    render_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if split not in {"calibration", "train", "dev", "test"}:
        raise ValueError("split must be calibration, train, dev, or test")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    director = DirectorAgent()
    effective_render_settings = {**RENDER_SETTINGS, **(render_settings or {})}
    jobs = []
    for record in _load_records(dataset_root, split, case_ids):
        case_id = record["case_id"]
        run_dir = output / case_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"real run directory already contains artifacts: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        director_result = director.plan(
            record["prompt"],
            scene_id=case_id,
            duration_s=record["duration_s"],
            fps=record["fps"],
        )
        contract = director_result.scene_contract
        plan = director_result.trajectory_plan
        prompt_hash = hash_prompt(record["prompt"])
        plan_hash = hash_payload(plan.model_dump(mode="json"))
        manifest = RealRunManifest(
            run_id=f"real-{case_id}-{prompt_hash[:10]}",
            case_id=case_id,
            split=split,
            prompt_hash=prompt_hash,
            plan_hash=plan_hash,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_version="pending-mcp",
            fps=plan.timebase.fps,
            frame_start=plan.timebase.frame_start,
            frame_end=plan.timebase.frame_end,
            render_settings=effective_render_settings,
            fingerprint=fingerprint_real_run(
                prompt_hash=prompt_hash,
                plan_hash=plan_hash,
                harness_version=harness_version,
                evaluator_version=evaluator_version,
                blender_version="pending-mcp",
                render_settings=effective_render_settings,
            ),
            state="prepared",
        )
        (run_dir / "director_plan.json").write_text(json.dumps(director_result.director_plan.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "scene_contract.json").write_text(json.dumps(contract.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "proxy_scene.json").write_text(json.dumps(record.get("proxy_scene", {}), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "trajectory.json").write_text(json.dumps(plan.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "camera_plan.json").write_text(json.dumps(plan.camera.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
        write_manifest(manifest, run_dir / "run_manifest.json")
        job_path = run_dir / "blender_job.py"
        job_path.write_text(compile_real_proxy_job(plan, manifest, run_dir, proxy_spec=record.get("proxy_scene")), encoding="utf-8")
        jobs.append({"case_id": case_id, "run_dir": str(run_dir), "job_path": str(job_path), "plan_hash": plan_hash})
    index = {"split": split, "harness_version": harness_version, "evaluator_version": evaluator_version, "case_count": len(jobs), "jobs": jobs}
    (output / "job_index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["calibration", "train", "dev", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--harness-version", default="h1-real")
    parser.add_argument("--evaluator-version", default="deterministic-v1")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), default=None)
    parser.add_argument("--samples", type=int, default=None)
    args = parser.parse_args()
    render_settings = {}
    if args.resolution:
        render_settings["resolution"] = args.resolution
    if args.samples is not None:
        render_settings["samples"] = args.samples
    print(json.dumps(prepare_jobs(args.split, args.out_dir, dataset_root=args.dataset_root, harness_version=args.harness_version, evaluator_version=args.evaluator_version, case_ids=args.case_id, render_settings=render_settings), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
