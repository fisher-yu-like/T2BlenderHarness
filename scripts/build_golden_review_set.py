"""Build an arm-blinded human-review bundle from real video artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from videoact.real_artifacts import probe_mp4, sample_event_aligned_frame_paths


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _read_records(dataset_root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (dataset_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _select_records(records: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not 30 <= count <= 50:
        raise ValueError("golden review case count must be between 30 and 50")
    if len(records) < count:
        raise ValueError(f"dataset has only {len(records)} records; cannot sample {count}")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record.get("category") or "uncategorized")].append(record)
    rng = random.Random(seed)
    for values in by_category.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    categories = sorted(by_category)
    while len(selected) < count:
        progressed = False
        for category in categories:
            if by_category[category] and len(selected) < count:
                selected.append(by_category[category].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _write_contact_sheet(samples: dict[str, list[Path]], destination: Path, case_id: str) -> None:
    labels = sorted(samples)
    thumb_w, thumb_h, label_h = 192, 144, 28
    margin, label_w, gap = 12, 110, 6
    width = margin * 2 + label_w + 8 * thumb_w + 7 * gap
    height = margin * 2 + 30 + len(labels) * (thumb_h + label_h + 8)
    sheet = Image.new("RGB", (width, height), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), f"{case_id} / blinded samples", fill=(20, 30, 40), font=_font(18))
    for row, label in enumerate(labels):
        y = margin + 30 + row * (thumb_h + label_h + 8)
        draw.text((margin, y + 8), label, fill=(20, 30, 40), font=_font(14))
        for index, path in enumerate(samples[label][:8]):
            x = margin + label_w + index * (thumb_w + gap)
            try:
                with Image.open(path) as opened:
                    frame = opened.convert("RGB")
                    frame.thumbnail((thumb_w, thumb_h))
                    tile = Image.new("RGB", (thumb_w, thumb_h), "white")
                    tile.paste(frame, ((thumb_w - frame.width) // 2, (thumb_h - frame.height) // 2))
                sheet.paste(tile, (x, y))
            except (OSError, ValueError):
                draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline=(190, 40, 40), width=2)
            draw.text((x + 4, y + thumb_h + 5), f"f{index + 1}", fill=(70, 80, 90), font=_font(10))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def build_golden_review_set(
    *,
    dataset_root: str | Path,
    arm_roots: dict[str, str | Path],
    output_root: str | Path,
    sample_count: int = 30,
    seed: int = 20260827,
    include_splits: set[str] | None = None,
) -> dict[str, Any]:
    if len(arm_roots) < 3:
        raise ValueError("golden review requires three or more valid video sources")
    dataset = Path(dataset_root)
    records = _read_records(dataset)
    if include_splits is not None:
        allowed = {str(value) for value in include_splits}
        if not allowed or not allowed.issubset({"calibration", "train", "dev", "test"}):
            raise ValueError("include_splits must contain only calibration, train, dev, or test")
        records = [record for record in records if str(record.get("split")) in allowed]
    selected = _select_records(records, sample_count, seed)
    output = Path(output_root)
    manifest_path = output / "manifest.jsonl"
    blind_manifest: dict[str, Any] = {"version": "golden-review-v1", "seed": seed, "mapping": {}}
    manifest_rows: list[dict[str, Any]] = []
    render_prompt_mismatch_count = 0
    for record in selected:
        case_id = str(record["case_id"])
        source_prompt = record.get("source_prompt")
        render_prompt = record.get("prompt")
        review_prompt = source_prompt if isinstance(source_prompt, str) and source_prompt.strip() else record["prompt"]
        labels = [f"sample_{chr(ord('a') + index)}" for index in range(len(arm_roots))]
        shuffled_arms = list(sorted(arm_roots))
        random.Random(f"{seed}:{case_id}").shuffle(shuffled_arms)
        samples: dict[str, list[Path]] = {}
        videos: dict[str, Path] = {}
        private: dict[str, Any] = {}
        case_prompt_mismatch = bool(
            isinstance(source_prompt, str) and source_prompt.strip() and render_prompt != source_prompt
        )
        split = str(record.get("split") or "train")
        for label, arm in zip(labels, shuffled_arms):
            run_dir = Path(arm_roots[arm]) / "real" / split / case_id
            contract_path = run_dir / "scene_contract.json"
            if not contract_path.is_file():
                raise ValueError(f"missing scene contract for {arm}/{case_id}")
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            frames = sample_event_aligned_frame_paths(run_dir, contract, max_frames=8)
            if len(frames) < 3:
                raise ValueError(f"need at least 3 sampled frames for {arm}/{case_id}")
            video_source = run_dir / "proxy.mp4"
            video_probe = probe_mp4(video_source)
            if not video_probe["playable"]:
                raise ValueError(f"proxy video is not playable for {arm}/{case_id}: {video_probe.get('error')}")
            run_manifest_path = run_dir / "run_manifest.json"
            if run_manifest_path.is_file():
                try:
                    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
                    expected_prompt_hash = hashlib.sha256(review_prompt.encode("utf-8")).hexdigest()
                    if run_manifest.get("prompt_hash") != expected_prompt_hash:
                        case_prompt_mismatch = True
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    case_prompt_mismatch = True
            destination = output / "frames" / case_id / label
            destination.mkdir(parents=True, exist_ok=True)
            copied: list[Path] = []
            for index, frame in enumerate(frames, 1):
                target = destination / f"frame_{index:04d}.png"
                shutil.copy2(frame, target)
                copied.append(target.resolve())
            samples[label] = copied
            video_destination = output / "videos" / case_id / f"{label}.mp4"
            video_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_source, video_destination)
            videos[label] = video_destination.resolve()
            private[label] = {
                "arm": arm,
                "run_dir": str(run_dir.resolve()),
                "render_prompt_hash": hashlib.sha256(str(render_prompt).encode("utf-8")).hexdigest(),
            }
        _write_contact_sheet(samples, output / "sheets" / f"{case_id}.png", case_id)
        manifest_row = {
            "case_id": case_id,
            "blind_group": f"{case_id}-blind",
            "prompt": review_prompt,
            "prompt_en": review_prompt,
            "prompt_origin": "benchmark_verbatim" if isinstance(source_prompt, str) and source_prompt.strip() else "dataset_record",
            "split": split,
            "sampled_frames": {label: [str(path) for path in paths] for label, paths in samples.items()},
            "sampled_videos": {label: str(path) for label, path in videos.items()},
        }
        if isinstance(source_prompt, str) and source_prompt.strip():
            manifest_row["source_prompt"] = review_prompt
        for field in ("source_dataset", "source_dimension", "source_index"):
            if field in record:
                manifest_row[field] = record[field]
        manifest_rows.append(manifest_row)
        blind_manifest["mapping"][case_id] = private
        if case_prompt_mismatch:
            render_prompt_mismatch_count += 1
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows), encoding="utf-8")
    (output / "human_scores.jsonl").write_text("", encoding="utf-8")
    fingerprint = hashlib.sha256(
        json.dumps(manifest_rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (output / "blind_manifest.json").write_text(json.dumps(blind_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_id": "golden-review-exact-v2",
                "fingerprint": fingerprint,
                "reproducibility_seed": seed,
                "case_count": len(manifest_rows),
                "arm_count": len(arm_roots),
                "sample_labels": sorted(labels),
                "source_dataset": sorted({str(row["source_dataset"]) for row in manifest_rows if row.get("source_dataset")}),
                "prompt_policy": "use source_prompt verbatim when present; never rewrite benchmark prompt for human review",
                "review_splits": sorted({str(row.get("split")) for row in manifest_rows}),
                "run_manifest_prompt_hash_policy": "every available run_manifest.prompt_hash must equal sha256(displayed prompt)",
                "benchmark_prompt_count": sum(1 for row in manifest_rows if row.get("prompt_origin") == "benchmark_verbatim"),
                "render_prompt_mismatch_count": render_prompt_mismatch_count,
                "comparison_only": render_prompt_mismatch_count > 0,
                "comparison_only_reason": (
                    "historical videos were rendered with an augmented executable prompt"
                    if render_prompt_mismatch_count
                    else None
                ),
                "annotation_granularity": "case_sample",
                "video_count": len(manifest_rows) * len(arm_roots),
                "arms_hidden": True,
                "patch_selection_allowed": False,
                "status": "awaiting_human_annotations",
                "annotators": [],
                "inter_rater_agreement": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "status": "awaiting_human_annotations",
        "case_count": len(manifest_rows),
        "video_count": len(manifest_rows) * len(arm_roots),
        "comparison_only": render_prompt_mismatch_count > 0,
        "render_prompt_mismatch_count": render_prompt_mismatch_count,
        "output_root": str(output.resolve()),
        "fingerprint": fingerprint,
    }


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--arm must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("--arm must be NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=ROOT; repeat for each blind video source")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--include-split",
        action="append",
        choices=["calibration", "train", "dev", "test"],
        default=None,
        help="restrict calibration cases to the named dataset splits; repeat for multiple splits",
    )
    args = parser.parse_args()
    arm_roots = dict(_parse_arm(value) for value in args.arm)
    print(json.dumps(build_golden_review_set(dataset_root=args.dataset_root, arm_roots=arm_roots, output_root=args.out, sample_count=args.sample_count, seed=args.seed, include_splits=set(args.include_split) if args.include_split else None), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
