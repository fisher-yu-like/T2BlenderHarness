"""Host-side video assembly for Blender PNG animation output."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field


class VideoAssemblyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str
    frame_count: int = Field(ge=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)


def assemble_mp4_from_pngs(
    frame_paths: list[str | Path],
    output_path: str | Path,
    *,
    fps: int,
) -> VideoAssemblyResult:
    if not frame_paths:
        raise ValueError("at least one PNG frame is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio-ffmpeg is required to assemble real proxy videos") from exc

    paths = [Path(path) for path in frame_paths]
    with Image.open(paths[0]) as first:
        rgb = first.convert("RGB")
        width, height = rgb.size
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size=(width, height),
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        fps=fps,
        codec="libx264",
        macro_block_size=1,
    )
    writer.send(None)
    try:
        for path in paths:
            with Image.open(path) as image:
                frame = image.convert("RGB")
                if frame.size != (width, height):
                    raise ValueError(f"frame size mismatch: {path}")
                writer.send(frame.tobytes())
    finally:
        writer.close()
    return VideoAssemblyResult(
        output_path=str(output), frame_count=len(paths), width=width, height=height, fps=fps
    )
