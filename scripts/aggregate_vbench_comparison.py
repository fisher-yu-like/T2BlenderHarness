"""Join paired baseline/current real-run reports and draw comparison curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont


VARIANTS = ("pretrain", "current")
ACTION_VARIANTS = ("direct_transfer", "reveal_elliptical_return", "subjectless_handoff_return", "parallel_transfer")


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def _result(run_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json", {})
    deterministic = _read_json(run_dir / "deterministic_report.json", {})
    realism = _read_json(run_dir / "realism_report.json", {})
    vlm = _read_json(run_dir / "vlm_report.json", {})
    aggregate = vlm.get("aggregate") or {}
    task = aggregate.get("final_score")
    realism_score = vlm.get("realism_score")
    realism_kind = vlm.get("realism_score_kind")
    if realism_score is None:
        realism_score = realism.get("score")
        realism_kind = realism.get("score_kind")
    findings = deterministic.get("findings") or []
    director_findings = deterministic.get("director_findings") or []
    interaction_findings = deterministic.get("interaction_findings") or []
    all_findings = [*findings, *director_findings, *interaction_findings]
    issue_text = "; ".join(
        str(item.get("message") or item.get("failure_id"))
        for item in all_findings[:4]
        if isinstance(item, dict)
    ) or "no evaluator finding"
    video = run_dir / "proxy.mp4"
    return {
        "case_id": record["case_id"],
        "split": record["split"],
        "category": record["category"],
        "source_dimension": record["source_dimension"],
        "source_prompt": record["source_prompt"],
        "prompt": record["prompt"],
        "proxy_video": str(video.resolve()) if video.is_file() else None,
        "run_dir": str(run_dir.resolve()),
        "harness_version": manifest.get("harness_version"),
        "plan_hash": manifest.get("plan_hash"),
        "director_plan_hash": manifest.get("director_plan_hash"),
        "task_score": task,
        "realism_score": realism_score,
        "realism_score_kind": realism_kind,
        "deterministic_score": deterministic.get("score"),
        "director_plan_score": deterministic.get("director_plan_score"),
        "finding_count": len(all_findings),
        "hard_gate_failed": deterministic.get("hard_gate_failed"),
        "harness_issue": issue_text,
        "vlm_status": vlm.get("status"),
        "review_source": vlm.get("review_source"),
        "review_confidence": vlm.get("review_confidence", vlm.get("confidence")),
    }


def _delta_text(base: dict[str, Any], current: dict[str, Any]) -> str:
    messages: list[str] = []
    base_count = int(base.get("finding_count") or 0)
    current_count = int(current.get("finding_count") or 0)
    if current_count < base_count:
        messages.append(f"current findings decreased {base_count}->{current_count}")
    elif current_count > base_count:
        messages.append(f"current findings increased {base_count}->{current_count}")
    elif current.get("harness_issue") != base.get("harness_issue"):
        messages.append("finding type changed")
    else:
        messages.append("same deterministic finding profile")
    base_task, current_task = base.get("task_score"), current.get("task_score")
    if isinstance(base_task, (int, float)) and isinstance(current_task, (int, float)):
        messages.append(f"task delta {float(current_task) - float(base_task):+.2f}")
    return "; ".join(messages)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if isinstance(row.get(field), (int, float))]
    return round(mean(values), 4) if values else None


def _draw_curve(rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 1800, 980
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    title_font = _font(28)
    axis_font = _font(15)
    draw.text((60, 24), "VBench-derived 100: pretraining vs current Harness", fill=(25, 35, 50), font=title_font)
    panels = [("task_score", "Task score", 105), ("realism_score", "Realism score", 540)]
    colors = {"pretrain": (110, 125, 145), "current": (30, 125, 82)}
    for field, label, top in panels:
        left, right, bottom = 95, width - 80, top + 320
        draw.text((left, top - 30), label, fill=(45, 55, 70), font=_font(20))
        draw.line((left, bottom, right, bottom), fill=(80, 90, 105), width=2)
        draw.line((left, top, left, bottom), fill=(80, 90, 105), width=2)
        for tick in range(0, 101, 20):
            y = bottom - (bottom - top) * tick / 100
            draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
            draw.text((48, y - 8), str(tick), fill=(90, 100, 115), font=axis_font)
        max_case_index = max((int(row.get("case_index", 0)) for row in rows), default=1)
        for variant in VARIANTS:
            values = [row for row in rows if row.get("variant") == variant]
            points = []
            for row in values:
                value = row.get(field)
                if not isinstance(value, (int, float)):
                    continue
                x = left + (right - left) * int(row.get("case_index", 0)) / max(1, max_case_index)
                y = bottom - (bottom - top) * float(value) / 100
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=colors[variant], width=3)
            for point in points[::max(1, len(points) // 25)]:
                x, y = point
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[variant])
        legend_x = right - 260
        for index, variant in enumerate(VARIANTS):
            x = legend_x + index * 120
            draw.line((x, top + 8, x + 25, top + 8), fill=colors[variant], width=4)
            draw.text((x + 32, top), variant, fill=(65, 75, 90), font=axis_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def build_report(benchmark_root: str | Path, dataset_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    benchmark = Path(benchmark_root)
    output = Path(output_root)
    records = [
        json.loads(line)
        for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paired: list[dict[str, Any]] = []
    for case_index, record in enumerate(records):
        action_variant = ACTION_VARIANTS[case_index % len(ACTION_VARIANTS)]
        pair: dict[str, Any] = {"case_id": record["case_id"], "case_index": case_index, "category": record["category"], "split": record["split"], "action_variant": action_variant, "prompt": record["prompt"], "source_prompt": record["source_prompt"]}
        for variant in VARIANTS:
            path = benchmark / variant / "real" / record["split"] / record["case_id"]
            pair[variant] = _result(path, record)
        pair["task_delta"] = (
            float(pair["current"].get("task_score")) - float(pair["pretrain"].get("task_score"))
            if all(isinstance(pair[variant].get("task_score"), (int, float)) for variant in VARIANTS)
            else None
        )
        pair["realism_delta"] = (
            float(pair["current"].get("realism_score")) - float(pair["pretrain"].get("realism_score"))
            if all(isinstance(pair[variant].get("realism_score"), (int, float)) for variant in VARIANTS)
            else None
        )
        pair["paired_interpretation"] = _delta_text(pair["pretrain"], pair["current"])
        paired.append(pair)
    flat = []
    for pair in paired:
        for variant in VARIANTS:
            row = {"case_id": pair["case_id"], "case_index": pair["case_index"], "category": pair["category"], "split": pair["split"], "action_variant": pair["action_variant"], "variant": variant, **pair[variant]}
            flat.append(row)
    summary: dict[str, Any] = {}
    for group_name, selector in (("all", lambda row: True), ("train", lambda row: row["split"] == "train"), ("dev", lambda row: row["split"] == "dev")):
        group: dict[str, Any] = {}
        for variant in VARIANTS:
            rows = [row for row in flat if row["variant"] == variant and selector(row)]
            group[variant] = {"cases": len(rows), "task_mean": _mean(rows, "task_score"), "realism_mean": _mean(rows, "realism_score"), "deterministic_mean": _mean(rows, "deterministic_score")}
        summary[group_name] = group
    category_summary = {}
    for category in sorted({record["category"] for record in records}):
        category_summary[category] = {}
        for variant in VARIANTS:
            rows = [row for row in flat if row["category"] == category and row["variant"] == variant]
            category_summary[category][variant] = {"cases": len(rows), "task_mean": _mean(rows, "task_score"), "realism_mean": _mean(rows, "realism_score")}
    action_summary = {}
    for action_variant in ACTION_VARIANTS:
        action_summary[action_variant] = {}
        for variant in VARIANTS:
            rows = [row for row in flat if row["action_variant"] == action_variant and row["variant"] == variant]
            action_summary[action_variant][variant] = {"cases": len(rows), "task_mean": _mean(rows, "task_score"), "realism_mean": _mean(rows, "realism_score")}
    deltas = [pair["task_delta"] for pair in paired if isinstance(pair["task_delta"], (int, float))]
    realism_deltas = [pair["realism_delta"] for pair in paired if isinstance(pair["realism_delta"], (int, float))]
    current_wins = sum(delta > 1e-9 for delta in deltas)
    current_losses = sum(delta < -1e-9 for delta in deltas)
    payload = {
        "benchmark_id": "vbench-derived-100-current-vs-pretrain-v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "benchmark_root": str(benchmark.resolve()),
        "case_count": len(records),
        "paired_case_count": len(paired),
        "task_scored_pairs": len(deltas),
        "realism_scored_pairs": len(realism_deltas),
        "current_task_wins": current_wins,
        "current_task_losses": current_losses,
        "current_task_ties": len(deltas) - current_wins - current_losses,
        "task_delta_mean": round(mean(deltas), 4) if deltas else None,
        "realism_delta_mean": round(mean(realism_deltas), 4) if realism_deltas else None,
        "summary": summary,
        "category_summary": category_summary,
        "action_summary": action_summary,
        "pairs": paired,
        "flat_rows": flat,
        "policy": {"scores": "task and realism remain separate; no combined score is used", "missing": "missing scores remain null and are excluded from means", "training": "no benchmark result was used to patch either Harness"},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _draw_curve(flat, output / "vbench-100-current-vs-pretrain-curves.png")
    lines = [
        "# VBench-derived 100 Harness paired comparison",
        "",
        f"- Cases: {len(records)}; paired cases: {len(paired)}; task-scored pairs: {len(deltas)}; realism-scored pairs: {len(realism_deltas)}.",
        f"- Current task wins/ties/losses: {current_wins}/{len(deltas) - current_wins - current_losses}/{current_losses}.",
        f"- Mean task delta (current - pretrain): {payload['task_delta_mean']}.",
        f"- Mean realism delta (current - pretrain): {payload['realism_delta_mean']}.",
        "- Scores are kept separate; null means the real video or independent review was not eligible and was not imputed.",
        "",
        f"![paired curves]({(output / 'vbench-100-current-vs-pretrain-curves.png').resolve().as_posix()})",
        "",
        "## Overall and split summary",
        "",
        "| group | variant | cases | task mean | realism mean | deterministic mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group_name, group in summary.items():
        for variant in VARIANTS:
            row = group[variant]
            lines.append(f"| {group_name} | {variant} | {row['cases']} | {row['task_mean']} | {row['realism_mean']} | {row['deterministic_mean']} |")
    lines.extend(["", "## Action-pattern diagnostic", "", "| action pattern | variant | cases | task mean | realism mean |", "|---|---|---:|---:|---:|"])
    for action_variant in ACTION_VARIANTS:
        for variant in VARIANTS:
            row = action_summary[action_variant][variant]
            lines.append(f"| {action_variant} | {variant} | {row['cases']} | {row['task_mean']} | {row['realism_mean']} |")
    lines.extend(["", "## Paired case table", "", "| case | split | category | prompt | pretrain video | current video | pretrain task | current task | task delta | pretrain realism | current realism | realism delta | Harness/evaluator interpretation |", "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|"])
    for pair in paired:
        base, current = pair["pretrain"], pair["current"]
        prompt = pair["prompt"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {pair['case_id']} | {pair['split']} | {pair['category']} | {prompt} | "
            f"{base.get('proxy_video')} | {current.get('proxy_video')} | {base.get('task_score')} | {current.get('task_score')} | "
            f"{pair.get('task_delta')} | {base.get('realism_score')} | {current.get('realism_score')} | {pair.get('realism_delta')} | {pair.get('paired_interpretation')} |"
        )
    report_text = "\n".join(lines).replace("螖", "delta")
    (output / "vbench-100-current-vs-pretrain-report.md").write_text(report_text + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_report(args.benchmark_root, args.dataset_root, args.out)
    print(json.dumps({key: payload[key] for key in ("case_count", "paired_case_count", "task_scored_pairs", "realism_scored_pairs", "task_delta_mean", "realism_delta_mean")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
