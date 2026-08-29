"""Create auditable contact sheets from exact evaluator-sampled frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _sheet(requests: list[tuple[str, list[Path]]], output: Path, title: str) -> None:
    thumb_w, thumb_h = 128, 128
    label_h = 36
    row_h = thumb_h + label_h
    margin = 12
    label_w = 108
    width = margin * 2 + label_w + thumb_w * 8 + 7 * 4
    height = margin * 2 + 42 + row_h * len(requests)
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    title_font = _font(18)
    small_font = _font(11)
    draw.text((margin, margin), title, fill=(25, 35, 50), font=title_font)
    for row, (case_id, frames) in enumerate(requests):
        y = margin + 42 + row * row_h
        draw.text((margin, y + 5), case_id, fill=(25, 35, 50), font=small_font)
        x0 = margin + label_w
        for index, frame_path in enumerate(frames[:8]):
            x = x0 + index * (thumb_w + 4)
            try:
                frame = Image.open(frame_path).convert("RGB")
                frame.thumbnail((thumb_w, thumb_h))
                canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
                canvas.paste(frame, ((thumb_w - frame.width) // 2, (thumb_h - frame.height) // 2))
                image.paste(canvas, (x, y))
            except (OSError, ValueError):
                draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(180, 50, 50), width=2)
            draw.text((x + 3, y + thumb_h + 2), f"f{index + 1}", fill=(80, 90, 105), font=small_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def build_sheets(benchmark_root: str | Path, output_root: str | Path, chunk_size: int = 10) -> dict[str, int]:
    benchmark = Path(benchmark_root)
    output = Path(output_root)
    counts: dict[str, int] = {}
    for variant in ("pretrain", "current"):
        for split in ("train", "dev"):
            run_root = benchmark / variant / "real" / split
            entries: list[tuple[str, list[Path]]] = []
            for run_dir in sorted(path for path in run_root.iterdir() if path.is_dir()) if run_root.is_dir() else []:
                request_path = run_dir / "assistant_review_request.json"
                if not request_path.is_file():
                    continue
                payload = json.loads(request_path.read_text(encoding="utf-8"))
                frames = [Path(path) for path in payload.get("sampled_frames", [])]
                if len(frames) >= 8 and all(path.is_file() for path in frames):
                    entries.append((run_dir.name, frames))
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                sheet_name = f"{variant}-{split}-{start + 1:03d}-{start + len(chunk):03d}.png"
                _sheet(chunk, output / sheet_name, f"{variant} / {split} / cases {start + 1}-{start + len(chunk)}")
            counts[f"{variant}/{split}"] = len(entries)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_sheets(args.benchmark_root, args.out), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
