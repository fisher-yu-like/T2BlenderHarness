"""Resume only incomplete real Blender jobs in an immutable run root."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.render_proxy_jobs_parallel import _run_one  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--max-render-retries", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.run_root)
    job_dirs = sorted(path.parent for path in root.glob("*/blender_job.py"))
    missing = [path for path in job_dirs if not (path / "render_attempts.json").is_file() or not (path / "proxy.mp4").is_file()]
    results = []
    workers = max(1, args.workers or min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, job, args.blender_bin, args.timeout_s, args.max_render_retries): job.name
            for job in missing
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["case_id"])
    report = {
        "status": "completed",
        "run_root": str(root.resolve()),
        "job_count": len(job_dirs),
        "resumed_count": len(missing),
        "skipped_complete_count": len(job_dirs) - len(missing),
        "workers": workers,
        "max_render_retries": args.max_render_retries,
        "success_count": sum(item.get("status") == "success" for item in results),
        "results": results,
    }
    (root / "resume_render_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
