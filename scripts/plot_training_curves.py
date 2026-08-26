"""Draw separate Director/task/realism/artifact curves from completed rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_ROOT = ROOT / "out" / "training" / "multi-five-rounds-v1"
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "multi-training-curves-v1.png"


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _report_path(report_root: Path, round_number: int) -> Path:
    candidates = (
        report_root / f"round-{round_number:02d}" / "overall_report.json",
        report_root / f"round-{round_number:02d}" / "overall_evaluation.json",
        report_root / f"round-{round_number:02d}" / "overall" / "real" / "real_unified_score.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no completed overall report for round {round_number} under {report_root}")


def _number(row: dict, *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _summary(report_root: Path, round_number: int) -> dict:
    report = json.loads(_report_path(report_root, round_number).read_text(encoding="utf-8"))
    cases = report.get("cases", [])
    summary = {"round": round_number}
    if not cases and report.get("splits"):
        # Rounds 1-3 were written by the earlier unified-report shape, where
        # the case rows remain under split reports.  Read the same aggregate
        # channels so the final curve does not silently drop those rounds.
        for split in ("train", "dev"):
            split_report = report["splits"].get(split, {})
            aggregate = split_report.get("aggregate", {})
            summary[f"{split}_director"] = None
            summary[f"{split}_task"] = _number(
                aggregate,
                "mean_task_final_score",
                "mean_final_score",
            )
            summary[f"{split}_realism"] = _number(
                aggregate,
                "mean_artifact_only_realism_score",
            )
            summary[f"{split}_deterministic"] = _number(
                aggregate,
                "mean_deterministic_score",
            )
            rows = split_report.get("cases", [])
            summary[f"{split}_pass_rate"] = (
                sum(row.get("deterministic_status") == "pass" for row in rows) / len(rows) * 100.0
                if rows
                else None
            )
            summary[f"{split}_artifact_completion"] = (
                sum(
                    100.0
                    if row.get("artifact_status") == "complete" or row.get("video_exists") is True
                    else 0.0
                    for row in rows
                )
                / len(rows)
                if rows
                else None
            )
        summary["scored_count"] = sum(
            split_report.get("aggregate", {}).get("realism_scored_count", 0)
            for split_report in report.get("splits", {}).values()
        )
        summary["video_count"] = sum(
            bool(row.get("video_exists"))
            for split_report in report.get("splits", {}).values()
            for row in split_report.get("cases", [])
        )
        return summary
    channels = {
        "director": ("director_plan_score",),
        "task": ("task_final_score", "video_score", "task_score"),
        "realism": ("realism_score",),
        "deterministic": ("deterministic_score",),
    }
    for split in ("train", "dev"):
        rows = [row for row in cases if row.get("split") == split]
        for channel, keys in channels.items():
            values = [value for row in rows if (value := _number(row, *keys)) is not None]
            summary[f"{split}_{channel}"] = sum(values) / len(values) if values else None
        summary[f"{split}_pass_rate"] = (
            sum(row.get("deterministic_status") == "pass" for row in rows) / len(rows) * 100.0
            if rows
            else None
        )
        completion = [
            100.0 if row.get("artifact_status") == "complete" or row.get("video_exists") is True else 0.0
            for row in rows
        ]
        summary[f"{split}_artifact_completion"] = sum(completion) / len(completion) if completion else None
    summary["scored_count"] = sum(
        _number(row, "task_final_score", "video_score", "task_score") is not None for row in cases
    )
    summary["video_count"] = sum(
        row.get("video_exists") is True or row.get("artifact_status") == "complete" for row in cases
    )
    return summary


def _draw_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, str, list[float | None]]],
    rounds: list[int],
    fonts: dict,
):
    left, top, right, bottom = box
    draw.rectangle(box, outline=(190, 198, 210), width=2)
    draw.text((left + 20, top + 16), title, fill=(26, 37, 52), font=fonts["subtitle"])
    plot = (left + 72, top + 70, right - 28, bottom - 58)
    px0, py0, px1, py1 = plot
    for tick in (0, 25, 50, 75, 100):
        y = py1 - (py1 - py0) * tick / 100
        draw.line((px0, y, px1, y), fill=(224, 229, 236), width=1)
        draw.text((left + 20, y - 9), str(tick), fill=(90, 103, 120), font=fonts["small"])
    for index, round_number in enumerate(rounds):
        x = px0 if len(rounds) == 1 else px0 + (px1 - px0) * index / (len(rounds) - 1)
        draw.line((x, py0, x, py1), fill=(239, 242, 246), width=1)
        draw.text((x - 12, py1 + 13), f"R{round_number}", fill=(90, 103, 120), font=fonts["small"])
    for name, color, values in series:
        points = []
        for index, value in enumerate(values):
            if value is None:
                continue
            x = px0 if len(rounds) == 1 else px0 + (px1 - px0) * index / (len(rounds) - 1)
            y = py1 - (py1 - py0) * float(value) / 100
            points.append((int(x), int(y)))
        if len(points) >= 2:
            draw.line(points, fill=color, width=4)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline="white", width=2)
    legend_x = px0
    legend_y = bottom - 30
    for name, color, _ in series:
        draw.line((legend_x, legend_y + 8, legend_x + 26, legend_y + 8), fill=color, width=4)
        draw.text((legend_x + 34, legend_y), name, fill=(57, 70, 88), font=fonts["small"])
        legend_x += 190


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--round", dest="rounds", action="append", type=int)
    args = parser.parse_args()
    report_root = Path(args.report_root)
    rounds = args.rounds or sorted(
        int(path.name.removeprefix("round-"))
        for path in report_root.glob("round-*")
        if path.is_dir() and path.name.removeprefix("round-").isdigit()
    )
    if not rounds:
        raise SystemExit("no completed overall rounds found; missing rounds are never synthesized")
    summaries = [_summary(report_root, round_number) for round_number in rounds]
    image = Image.new("RGB", (1500, 920), (248, 250, 253))
    draw = ImageDraw.Draw(image)
    fonts = {"title": _font(34, True), "subtitle": _font(22, True), "small": _font(16)}
    draw.text((56, 28), "T2Blendercodeharness multi-five real training", fill=(20, 31, 46), font=fonts["title"])
    draw.text((58, 72), "Director plan, task, realism, and artifact completion; unavailable review is not zero", fill=(84, 96, 113), font=fonts["small"])
    _draw_chart(
        draw, (48, 120, 730, 510), "Director plan score",
        [("train", (24, 101, 178), [s["train_director"] for s in summaries]), ("dev", (210, 91, 67), [s["dev_director"] for s in summaries])], rounds, fonts,
    )
    _draw_chart(
        draw, (770, 120, 1452, 510), "Task score",
        [("train", (36, 142, 89), [s["train_task"] for s in summaries]), ("dev", (139, 91, 173), [s["dev_task"] for s in summaries])], rounds, fonts,
    )
    _draw_chart(
        draw, (48, 550, 730, 900), "Realism score",
        [("train", (193, 130, 38), [s["train_realism"] for s in summaries]), ("dev", (87, 112, 137), [s["dev_realism"] for s in summaries])], rounds, fonts,
    )
    _draw_chart(
        draw, (770, 550, 1452, 900), "Artifact completion (%)",
        [("train", (24, 101, 178), [s["train_artifact_completion"] for s in summaries]), ("dev", (210, 91, 67), [s["dev_artifact_completion"] for s in summaries])], rounds, fonts,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(json.dumps({"output": str(output), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
