"""Plot paired baseline/candidate independent realism and legacy scores."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "out" / "complex-realism-v1" / "evolution-v1" / "realism_evolution_v2_report.json"
OUTPUT = ROOT / "docs" / "assets" / "realism-evolution-v2-comparison.png"


def font(size: int, bold: bool = False):
    path = "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"
    return ImageFont.truetype(path, size) if Path(path).is_file() else ImageFont.load_default()


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    labels = [case["case_id"] for case in report["cases"]]
    image = Image.new("RGB", (1500, 760), (248, 250, 253))
    draw = ImageDraw.Draw(image)
    title = font(32, True)
    subtitle = font(18)
    small = font(15)
    draw.text((48, 30), "Realism evolution: baseline vs candidate-v2", fill=(22, 35, 52), font=title)
    draw.text((50, 72), "Same prompts and plan hashes; old evaluator is a non-regression gate", fill=(78, 93, 112), font=subtitle)
    panels = [(48, 122, 730, 700, "Old evaluator score"), (770, 122, 1452, 700, "Independent realism score")]
    for left, top, right, bottom, panel_title in panels:
        draw.rectangle((left, top, right, bottom), outline=(190, 200, 214), width=2)
        draw.text((left + 20, top + 18), panel_title, fill=(27, 41, 60), font=font(21, True))
        px0, py0, px1, py1 = left + 66, top + 70, right - 26, bottom - 72
        for tick in (0, 25, 50, 75, 100):
            y = py1 - (py1 - py0) * tick / 100
            draw.line((px0, y, px1, y), fill=(225, 230, 237), width=1)
            draw.text((left + 18, y - 8), str(tick), fill=(90, 103, 120), font=small)
        before = [case["baseline_legacy_score"] if panel_title.startswith("Old") else case["baseline_realism_score"] for case in report["cases"]]
        after = [case["candidate_legacy_score"] if panel_title.startswith("Old") else case["candidate_realism_score"] for case in report["cases"]]
        step = (px1 - px0) / len(labels)
        bar_width = max(8, int(step * 0.28))
        for index, label in enumerate(labels):
            center = px0 + step * (index + 0.5)
            for value, color, offset in ((before[index], (138, 154, 173), -bar_width * 0.60), (after[index], (31, 124, 86), bar_width * 0.60)):
                height = (py1 - py0) * float(value) / 100
                x0 = int(center + offset - bar_width / 2)
                x1 = int(center + offset + bar_width / 2)
                draw.rectangle((x0, py1 - height, x1, py1), fill=color)
            draw.text((center - 25, py1 + 14), label, fill=(78, 93, 112), font=small)
        draw.rectangle((px0, bottom - 39, px0 + 18, bottom - 23), fill=(138, 154, 173))
        draw.text((px0 + 27, bottom - 42), "baseline", fill=(78, 93, 112), font=small)
        draw.rectangle((px0 + 120, bottom - 39, px0 + 138, bottom - 23), fill=(31, 124, 86))
        draw.text((px0 + 147, bottom - 42), "candidate", fill=(78, 93, 112), font=small)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(json.dumps({"output": str(OUTPUT), "case_count": len(labels), "aggregate": report["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
