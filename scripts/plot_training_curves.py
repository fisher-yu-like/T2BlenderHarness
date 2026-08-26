"""Draw training-memory curves with Pillow (no plotting dependency required)."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "out" / "training" / "six-rounds-real-v6"
OUTPUT = ROOT / "docs" / "assets" / "t2blendercodeharness-training-curves.png"


def _font(size: int, bold: bool = False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _summary(round_number: int) -> dict:
    report = json.loads((REPORT_ROOT / f"round-{round_number:02d}" / "overall_report.json").read_text(encoding="utf-8"))
    cases = report["cases"]
    summary = {"round": round_number}
    for split in ("train", "dev"):
        rows = [row for row in cases if row.get("split") == split]
        summary[f"{split}_deterministic"] = sum(row["deterministic_score"] for row in rows) / len(rows)
        summary[f"{split}_pass_rate"] = sum(row["deterministic_status"] == "pass" for row in rows) / len(rows) * 100.0
        scored = [row["video_score"] for row in rows if row.get("video_score") is not None]
        summary[f"{split}_visual_scored"] = sum(scored) / len(scored) if scored else None
    summary["scored_count"] = sum(row.get("video_score") is not None for row in cases)
    summary["video_count"] = sum(row.get("video_exists") is True for row in cases)
    return summary


def _draw_chart(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, series: list[tuple[str, str, list[float | None]]], rounds: list[int], fonts: dict):
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
        label = f"R{round_number}"
        draw.text((x - 12, py1 + 13), label, fill=(90, 103, 120), font=fonts["small"])
    for series_name, color, values in series:
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
    rounds = [1, 2]
    summaries = [_summary(round_number) for round_number in rounds]
    image = Image.new("RGB", (1500, 920), (248, 250, 253))
    draw = ImageDraw.Draw(image)
    fonts = {"title": _font(34, True), "subtitle": _font(22, True), "small": _font(16)}
    draw.text((56, 28), "T2Blendercodeharness real training memory", fill=(20, 31, 46), font=fonts["title"])
    draw.text((58, 72), "Rounds 1–2; real Blender videos; visual scores only where the deterministic gate passed", fill=(84, 96, 113), font=fonts["small"])
    _draw_chart(
        draw, (48, 120, 730, 510), "Deterministic score (higher is better)",
        [("train", (24, 101, 178), [s["train_deterministic"] for s in summaries]), ("dev", (210, 91, 67), [s["dev_deterministic"] for s in summaries])], rounds, fonts,
    )
    _draw_chart(
        draw, (770, 120, 1452, 510), "Deterministic pass rate (%)", 
        [("train", (24, 101, 178), [s["train_pass_rate"] for s in summaries]), ("dev", (210, 91, 67), [s["dev_pass_rate"] for s in summaries])], rounds, fonts,
    )
    _draw_chart(
        draw, (48, 550, 730, 900), "Scored visual mean (assistant-local review)",
        [("train", (36, 142, 89), [s["train_visual_scored"] for s in summaries]), ("dev", (139, 91, 173), [s["dev_visual_scored"] for s in summaries])], rounds, fonts,
    )
    _draw_chart(
        draw, (770, 550, 1452, 900), "Real video coverage (%)", 
        [("rendered", (87, 112, 137), [s["video_count"] / 120 * 100 for s in summaries]), ("visual scored", (193, 130, 38), [s["scored_count"] / 120 * 100 for s in summaries])], rounds, fonts,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
