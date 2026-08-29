"""Build reproducible, arm-blind review requests for the three-arm benchmark."""

from __future__ import annotations

import argparse
import json
import random
import sys
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from videoact.real_artifacts import sample_event_aligned_frame_paths  # noqa: E402


ARMS = ("pretrain", "trained", "direct_code")
BLIND_LABELS = ("sample_a", "sample_b", "sample_c")
BLIND_REVIEW_VERSION = "three-arm-blind-v1"
VISUAL_DIMENSIONS = (
    "prompt_compliance", "physical_plausibility", "object_trajectory", "event_timing",
    "camera_coverage", "camera_innovation", "character_trajectory", "temporal_smoothness",
    "visual_clarity", "appearance_detail", "physical_realism", "spatial_consistency",
    "motion_naturalness", "visual_presentation",
)


def _json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def build_blind_mapping(records: list[dict[str, Any]], *, seed: int = 20260827) -> dict[str, dict[str, str]]:
    rng = random.Random(seed)
    mapping: dict[str, dict[str, str]] = {}
    for record in records:
        labels = list(BLIND_LABELS)
        rng.shuffle(labels)
        mapping[record["case_id"]] = dict(zip(ARMS, labels))
    return mapping


def build_case_review_request(record: dict[str, Any], samples: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "review_version": BLIND_REVIEW_VERSION,
        "review_source": "assistant_local_review",
        "reviewer": "codex-assistant",
        "case_id": record["case_id"],
        "prompt": record["prompt"],
        "samples": {
            label: {
                "sample_label": label,
                "sampled_frames": paths,
            }
            for label, paths in sorted(samples.items())
        },
        "dimensions": list(VISUAL_DIMENSIONS),
        "instructions": {
            "frame_order": "chronological",
            "score_range": "0-100 integer for every dimension; use null only when the complete sample is unavailable",
            "evidence_policy": "score only visible evidence in these exact sampled frames; do not infer from hidden plans",
            "trajectory_focus": "inspect camera coverage/innovation, character and object trajectories, event timing, and temporal smoothness",
            "realism_focus": "inspect appearance detail, physical realism, spatial consistency, motion naturalness, and visual presentation",
            "no_preset_bonus": "do not add score because of sample label, action type, branch, or expected method",
        },
    }


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _build_case_sheet(samples: dict[str, list[str]], output: Path, case_id: str) -> None:
    thumb_w, thumb_h, label_h = 128, 128, 35
    margin, label_w, gap = 12, 105, 4
    width = margin * 2 + label_w + 8 * thumb_w + 7 * gap
    height = margin * 2 + 36 + len(BLIND_LABELS) * (thumb_h + label_h)
    image = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.text((margin, margin), f"{case_id} / blind three-arm samples", fill=(25, 35, 50), font=_font(18))
    for row, label in enumerate(BLIND_LABELS):
        y = margin + 36 + row * (thumb_h + label_h)
        draw.text((margin, y + 5), label, fill=(25, 35, 50), font=_font(12))
        for index, raw_path in enumerate(samples.get(label, [])[:8]):
            x = margin + label_w + index * (thumb_w + gap)
            target = Path(raw_path)
            try:
                with Image.open(target) as opened:
                    frame = opened.convert("RGB")
                    frame.thumbnail((thumb_w, thumb_h))
                    canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
                    canvas.paste(frame, ((thumb_w - frame.width) // 2, (thumb_h - frame.height) // 2))
                image.paste(canvas, (x, y))
            except (OSError, ValueError):
                draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(180, 50, 50), width=2)
            draw.text((x + 3, y + thumb_h + 2), f"f{index + 1}", fill=(80, 90, 105), font=_font(10))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def prepare_requests(
    *,
    dataset_root: str | Path,
    pretrain_root: str | Path,
    trained_root: str | Path,
    direct_root: str | Path,
    output_root: str | Path,
    seed: int = 20260827,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    records = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mapping = build_blind_mapping(records, seed=seed)
    roots = {"pretrain": Path(pretrain_root), "trained": Path(trained_root), "direct_code": Path(direct_root)}
    output = Path(output_root)
    requests_dir = output / "requests"
    sheets_dir = output / "sheets"
    requests_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        split = record["split"]
        samples: dict[str, list[str]] = {}
        hidden_paths: dict[str, str] = {}
        for arm in ARMS:
            run_dir = roots[arm] / "real" / split / record["case_id"]
            contract = _json(run_dir / "scene_contract.json", {})
            frames = sample_event_aligned_frame_paths(run_dir, contract, max_frames=8)
            label = mapping[record["case_id"]][arm]
            anonymous_dir = output / "frames" / record["case_id"] / label
            anonymous_dir.mkdir(parents=True, exist_ok=True)
            paths = []
            for index, source_frame in enumerate(frames, start=1):
                # Never expose the original arm directory in the review
                # request. The evaluator-side blind_manifest retains the
                # private join key; reviewers see only sample_a/b/c paths.
                target = anonymous_dir / f"frame_{index:04d}.png"
                shutil.copy2(source_frame, target)
                paths.append(str(target.resolve()))
            samples[label] = paths
            hidden_paths[arm] = str(run_dir.resolve())
        request = build_case_review_request(record, samples)
        (requests_dir / f"{record['case_id']}.json").write_text(json.dumps(request, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _build_case_sheet(samples, sheets_dir / f"{record['case_id']}.png", record["case_id"])
        mapping[record["case_id"]]["_paths"] = hidden_paths

    # This file is an evaluator-side key and is kept outside each reviewer
    # request.  It is needed to join blind labels back to arms after review.
    (output / "blind_manifest.json").write_text(
        json.dumps({"version": BLIND_REVIEW_VERSION, "seed": seed, "arms": list(ARMS), "mapping": mapping}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"case_count": len(records), "request_count": len(list(requests_dir.glob("*.json"))), "sheet_count": len(list(sheets_dir.glob("*.png"))), "output_root": str(output.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pretrain-root", required=True)
    parser.add_argument("--trained-root", required=True)
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    print(json.dumps(prepare_requests(dataset_root=args.dataset_root, pretrain_root=args.pretrain_root, trained_root=args.trained_root, direct_root=args.direct_root, output_root=args.out, seed=args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
