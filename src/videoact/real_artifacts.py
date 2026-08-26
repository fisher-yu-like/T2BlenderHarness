"""Host-side validation and fingerprinting for real Blender proxy runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class RealRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    split: Literal["calibration", "train", "dev", "test"]
    prompt_hash: str
    plan_hash: str
    harness_version: str
    evaluator_version: str
    blender_version: str
    fps: int = Field(gt=0)
    frame_start: int = Field(ge=1)
    frame_end: int = Field(ge=1)
    render_settings: dict[str, Any]
    fingerprint: str
    state: Literal["prepared", "executing", "rendered", "artifact_valid", "evaluated", "failed"]


class RealArtifactReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_status: Literal["complete", "incomplete"]
    hard_failures: list[str] = Field(default_factory=list)
    readable_frame_count: int = Field(ge=0)
    video_frame_count: int = Field(default=0, ge=0)
    video_fps: float = Field(default=0.0, ge=0)
    video_duration_s: float = Field(default=0.0, ge=0)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    manifest: RealRunManifest | None = None


REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "scene_contract.json",
    "trajectory.json",
    "camera_plan.json",
    "blender_job.py",
    "proxy.blend",
    "proxy.mp4",
    "telemetry.json",
    "frames/index.json",
)


def sample_frame_paths(run_dir: str | Path, *, max_frames: int | None = None) -> list[Path]:
    frames_dir = Path(run_dir) / "frames"
    index_path = frames_dir / "index.json"
    indexed: list[Path] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8")).get("frames", [])
            indexed = [frames_dir / str(entry["path"]) for entry in entries]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            indexed = []
    if max_frames is None:
        return indexed or sorted(frames_dir.glob("*.png"))
    animation = sorted((frames_dir / "animation").glob("frame_*.png"))
    by_name: dict[str, Path] = {}
    direct = sorted(frames_dir.glob("frame_*.png"))
    for path in indexed + animation + direct:
        by_name.setdefault(path.name, path)
    candidates = sorted(
        by_name.values(),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]) if "_" in path.stem else path.name,
    )
    if len(candidates) <= max_frames:
        return candidates
    indices = {
        round(index * (len(candidates) - 1) / max(1, max_frames - 1))
        for index in range(max_frames)
    }
    return [candidates[index] for index in sorted(indices)]


def sample_event_aligned_frame_paths(
    run_dir: str | Path,
    scene_contract: Any,
    *,
    max_frames: int = 8,
) -> list[Path]:
    """Prefer event midpoints, then add endpoints and uniform timeline coverage."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    root = Path(run_dir)
    frames_dir = root / "frames"
    animation = sorted((frames_dir / "animation").glob("frame_*.png"))
    candidates = animation or sample_frame_paths(root, max_frames=None)
    if not candidates:
        return []

    def frame_number(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[-1])

    by_frame = {frame_number(path): path for path in candidates}
    frame_numbers = sorted(by_frame)
    if hasattr(scene_contract, "model_dump"):
        scene_contract = scene_contract.model_dump(mode="json")
    contract = scene_contract if isinstance(scene_contract, dict) else {}
    fps = int(contract.get("fps") or 24)
    events = contract.get("events") or []
    must_show = set(contract.get("must_show") or [event.get("id") for event in events])

    event_targets: list[int] = []
    for event in events:
        if event.get("id") not in must_show:
            continue
        midpoint_s = (float(event.get("start", 0.0)) + float(event.get("end", 0.0))) / 2.0
        event_targets.append(max(1, round(midpoint_s * fps) + 1))

    selected: list[int] = []
    for target in event_targets:
        nearest = min(frame_numbers, key=lambda value: abs(value - target))
        if nearest not in selected:
            selected.append(nearest)
        if len(selected) >= max(0, max_frames - 2):
            break
    for endpoint in (frame_numbers[0], frame_numbers[-1]):
        if endpoint not in selected:
            selected.append(endpoint)
    if len(selected) < max_frames:
        uniform_indices = {
            round(index * (len(frame_numbers) - 1) / max(1, max_frames - 1))
            for index in range(max_frames)
        }
        for index in sorted(uniform_indices):
            number = frame_numbers[index]
            if number not in selected:
                selected.append(number)
            if len(selected) >= max_frames:
                break
    return [by_frame[number] for number in sorted(selected[:max_frames])]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_mp4(path: str | Path, *, minimum_frames: int = 3) -> dict[str, Any]:
    """Decode enough of an MP4 to prove it is playable, not merely non-empty."""
    video = Path(path)
    if not video.is_file() or video.stat().st_size == 0:
        return {"playable": False, "frame_count": 0, "fps": 0.0, "duration_s": 0.0, "error": "missing_or_empty"}
    try:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
        metadata = next(reader)
        frame_count = 0
        try:
            for _frame in reader:
                frame_count += 1
                if frame_count >= minimum_frames:
                    break
        finally:
            close = getattr(reader, "close", None)
            if close:
                close()
        return {
            "playable": frame_count >= minimum_frames,
            "frame_count": frame_count,
            "fps": float(metadata.get("fps") or 0.0),
            "duration_s": float(metadata.get("duration") or 0.0),
            "error": None if frame_count >= minimum_frames else "insufficient_decoded_frames",
        }
    except (ImportError, OSError, RuntimeError, ValueError, StopIteration) as exc:
        return {
            "playable": False,
            "frame_count": 0,
            "fps": 0.0,
            "duration_s": 0.0,
            "error": f"{type(exc).__name__}:{exc}",
        }


class RealArtifactGate:
    def __init__(self, *, minimum_readable_frames: int = 3):
        self.minimum_readable_frames = minimum_readable_frames

    def validate(self, run_dir: str | Path) -> RealArtifactReport:
        root = Path(run_dir)
        failures: list[str] = []
        hashes: dict[str, str] = {}
        manifest = None
        for relative in REQUIRED_ARTIFACTS:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing_artifact:{relative}")
            else:
                hashes[relative] = _sha256(path)

        manifest_path = root / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = RealRunManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"invalid_manifest:{type(exc).__name__}")

        readable = 0
        for frame_path in sample_frame_paths(root):
            try:
                with Image.open(frame_path) as image:
                    image.verify()
                readable += 1
                relative = str(frame_path.relative_to(root)).replace("\\", "/")
                hashes[relative] = _sha256(frame_path)
            except (OSError, ValueError):
                failures.append(f"unreadable_frame:{frame_path.name}")
        if readable < self.minimum_readable_frames:
            failures.append(f"insufficient_readable_frames:{readable}")

        video_probe = probe_mp4(root / "proxy.mp4", minimum_frames=self.minimum_readable_frames)
        if not video_probe["playable"]:
            failures.append("unplayable_video:proxy.mp4")

        return RealArtifactReport(
            artifact_status="complete" if not failures else "incomplete",
            hard_failures=sorted(set(failures)),
            readable_frame_count=readable,
            video_frame_count=int(video_probe["frame_count"]),
            video_fps=float(video_probe["fps"]),
            video_duration_s=float(video_probe["duration_s"]),
            artifact_hashes=hashes,
            manifest=manifest,
        )


def fingerprint_real_run(
    *,
    prompt_hash: str,
    plan_hash: str,
    harness_version: str,
    evaluator_version: str,
    blender_version: str,
    render_settings: dict[str, Any],
) -> str:
    payload = {
        "prompt_hash": prompt_hash,
        "plan_hash": plan_hash,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "blender_version": blender_version,
        "render_settings": render_settings,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resume_matches(manifest: RealRunManifest, expected: dict[str, Any]) -> bool:
    return manifest.fingerprint == expected.get("fingerprint")
