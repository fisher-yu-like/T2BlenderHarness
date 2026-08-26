"""Build a prompt -> proxy video/job -> score index for a dataset run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_index(
    dataset_root: str | Path,
    benchmark_root: str | Path,
    real_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    benchmark = Path(benchmark_root)
    real = Path(real_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = _records(dataset / "manifest.jsonl")
    splits = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    by_id = {record["case_id"]: record for record in manifest}
    benchmark_by_case: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        report_path = benchmark / "benchmarks" / split / "benchmark_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            benchmark_by_case.update({case["case_id"]: {**case, "split": split} for case in report.get("cases", [])})

    rows: list[dict[str, Any]] = []
    for split in ("train", "dev", "test"):
        for case_id in splits.get(split, []):
            record = by_id[case_id]
            run_dir = real / case_id
            video = run_dir / "proxy.mp4"
            job = run_dir / "blender_job.py"
            case_report = benchmark_by_case.get(case_id, {})
            video_exists = video.is_file() and video.stat().st_size > 0
            job_exists = job.is_file() and job.stat().st_size > 0
            rows.append(
                {
                    "case_id": case_id,
                    "split": split,
                    "template_family": record["template_family"],
                    "difficulty": record["difficulty"],
                    "scene_id": record["proxy_scene"]["scene_id"],
                    "prompt": record["prompt"],
                    "proxy_video_address": str(video.resolve()) if video_exists else f"NOT_RENDERED: {video.resolve()}",
                    "proxy_job_address": str(job.resolve()) if job_exists else "NOT_PREPARED",
                    "score": case_report.get("score"),
                    "status": "rendered" if video_exists else "job_prepared" if job_exists else "not_prepared",
                    "failure_ids": ";".join(case_report.get("failure_ids", [])),
                    "plan_hash": case_report.get("plan_hash"),
                }
            )

    fields = list(rows[0]) if rows else []
    csv_path = output / "prompt_proxy_score_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = output / "prompt_proxy_score_index.md"
    lines = [
        "# Prompt → Proxy Video → Score Index",
        "",
        f"Dataset: `{dataset.resolve()}`  ",
        f"Benchmark: `{benchmark.resolve()}`  ",
        "Score source: candidate independent-oracle deterministic benchmark; not VLM score.  ",
        "`NOT_RENDERED` means the immutable Blender job exists but no proxy.mp4 artifact was produced.",
        "",
        "| Case | Split | Family | Prompt | Proxy video | Job | Score | Status | Failures |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    for row in rows:
        prompt = row["prompt"].replace("|", "\\|").replace("\n", " ")
        video = row["proxy_video_address"].replace("|", "\\|")
        job = row["proxy_job_address"].replace("|", "\\|")
        failures = row["failure_ids"].replace("|", "\\|")
        if row["status"] == "rendered":
            video = f"[proxy.mp4]({Path(row['proxy_video_address']).as_posix()})"
        if row["status"] in {"rendered", "job_prepared"}:
            job = f"[blender_job.py]({Path(row['proxy_job_address']).as_posix()})"
        lines.append(
            f"| {row['case_id']} | {row['split']} | {row['template_family']} | {prompt} | {video} | {job} | {row['score']} | {row['status']} | {failures} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "rows": len(rows),
        "rendered_videos": sum(row["status"] == "rendered" for row in rows),
        "prepared_jobs": sum(row["status"] == "job_prepared" for row in rows),
        "scores_available": sum(row["score"] is not None for row in rows),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
    }
    (output / "index_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset/trajectory-v3-hard")
    parser.add_argument("--benchmark-root", default="out/training/t2blendercodeharness-hard-final-v2")
    parser.add_argument("--real-root", default="out/real/trajectory-v3-hard-dev-prepared-final")
    parser.add_argument("--output-dir", default="out/indexes/trajectory-v3-hard")
    args = parser.parse_args()
    print(json.dumps(build_index(args.dataset_root, args.benchmark_root, args.real_root, args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
