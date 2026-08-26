"""Build chronological contact sheets from real evaluator-sampled PNGs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.real_artifacts import sample_event_aligned_frame_paths


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_sheet(run_dir: Path, output: Path) -> Path:
    contract = json.loads((run_dir / "scene_contract.json").read_text(encoding="utf-8"))
    frames = sample_event_aligned_frame_paths(run_dir, contract, max_frames=8)
    if not frames:
        raise ValueError(f"no sampled frames for {run_dir}")
    tiles = []
    for index, path in enumerate(frames, start=1):
        with Image.open(path) as source:
            image = source.convert("RGB").resize((384, 384))
        tile = Image.new("RGB", (384, 420), "white")
        tile.paste(image, (0, 36))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), f"sample {index} | {path.stem}", fill=(10, 10, 10), font=_font(20))
        tiles.append(tile)
    sheet = Image.new("RGB", (768, 1680), (232, 236, 242))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 2) * 384, (index // 2) * 420))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.run_root)
    output_root = Path(args.output_root)
    outputs = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and (path / "scene_contract.json").is_file()):
        outputs.append(str(make_sheet(run_dir, output_root / f"{run_dir.name}.png").resolve()))
    print(json.dumps({"count": len(outputs), "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
