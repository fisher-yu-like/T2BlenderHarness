"""Local blind-video review app for the evaluator golden set.

The server exposes only anonymized sample labels and writes sample-level human
scores atomically. It deliberately never serves ``blind_manifest.json`` or any
source-arm path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

try:  # Works both as ``python scripts/golden_review_app.py`` and as a package import.
    from scripts.prompt_translation import translate_prompt
except ImportError:  # pragma: no cover - exercised by the CLI entry point.
    from prompt_translation import translate_prompt


GOLDEN_DIMENSIONS = (
    "prompt_compliance",
    "physical_plausibility",
    "camera_coverage",
    "camera_innovation",
    "character_trajectory",
    "object_trajectory",
    "event_timing",
    "temporal_smoothness",
    "visual_clarity",
    "appearance_detail",
    "physical_realism",
    "spatial_consistency",
    "motion_naturalness",
    "visual_presentation",
)

ALLOWED_FAILURE_OWNERS = (
    "none",
    "director_prompt_interpreter",
    "director_event_scheduler",
    "director_trajectory",
    "director_camera",
    "blender_code_agent",
    "blender_executor",
    "proxy_renderer",
    "evaluator",
)
PASS_FAIL_VALUES = ("pass", "borderline", "fail")
LEAK_KEYS = {"arm", "variant", "branch", "commit", "score", "source_arm", "harness_version"}


def _resolve_inside(root: Path, value: Any) -> Path:
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("artifact path must be a non-empty string")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("artifact path escapes the review bundle") from exc
    return resolved


def _split_lines(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item.strip()]
    raise ValueError(f"{field} must be a string or list of strings")


class GoldenReviewStore:
    """Load a blind bundle and persist validated sample-level annotations."""

    def __init__(self, bundle: str | Path):
        self.bundle = Path(bundle).resolve()
        self.manifest_path = self.bundle / "manifest.jsonl"
        self.scores_path = self.bundle / "human_scores.jsonl"
        self._lock = threading.RLock()
        self._samples: dict[str, dict[str, dict[str, Any]]] = {}
        self._frames: dict[tuple[str, str], list[Path]] = {}
        self._sheets: dict[str, Path] = {}
        self._metadata = self._load_metadata()
        self._public_rows = self._load_manifest()

    def _load_metadata(self) -> dict[str, Any]:
        path = self.bundle / "metadata.json"
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _load_manifest(self) -> list[dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise ValueError(f"missing manifest: {self.manifest_path}")
        public_rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid manifest JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"manifest line {line_number} must be an object")
            case_id = str(row.get("case_id") or "")
            prompt = row.get("prompt")
            prompt_en = row.get("prompt_en") or prompt
            frames = row.get("sampled_frames")
            videos = row.get("sampled_videos")
            if not case_id or not isinstance(prompt, str) or not isinstance(prompt_en, str):
                raise ValueError(f"manifest line {line_number} has invalid case_id/prompt")
            if not isinstance(frames, dict) or not isinstance(videos, dict):
                raise ValueError(f"manifest case {case_id} must have sampled_frames and sampled_videos objects")
            labels = sorted({str(label) for label in frames} & {str(label) for label in videos})
            if len(labels) < 3 or set(frames) != set(videos):
                raise ValueError(f"manifest case {case_id} must contain matching three or more sample labels")
            samples: dict[str, dict[str, Any]] = {}
            for label in labels:
                video_path = _resolve_inside(self.bundle, videos[label])
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise ValueError(f"missing sampled video for {case_id}/{label}")
                frame_paths = [_resolve_inside(self.bundle, value) for value in (frames[label] or [])]
                self._frames[(case_id, label)] = frame_paths
                samples[label] = {
                    "video_url": f"/media/{quote(case_id, safe='')}/{quote(label, safe='')}.mp4",
                    "frame_urls": [
                        f"/frames/{quote(case_id, safe='')}/{quote(label, safe='')}/{index}"
                        for index, _path in enumerate(frame_paths)
                    ],
                }
            sheet = self.bundle / "sheets" / f"{case_id}.png"
            if sheet.is_file() and sheet.stat().st_size:
                self._sheets[case_id] = sheet.resolve()
            self._samples[case_id] = samples
            public_rows.append(
                {
                    "case_id": case_id,
                    "prompt": prompt,
                    "prompt_en": prompt_en,
                    "prompt_zh": translate_prompt(case_id, prompt_en),
                    "samples": samples,
                    "sheet_url": f"/sheets/{quote(case_id, safe='')}.png" if case_id in self._sheets else None,
                }
            )
        if not public_rows:
            raise ValueError("review manifest is empty")
        return public_rows

    def public_manifest(self) -> list[dict[str, Any]]:
        """Return prompt and anonymized media URLs, never source paths."""

        return json.loads(json.dumps(self._public_rows))

    def public_metadata(self) -> dict[str, Any]:
        """Return only safe bundle metadata; never disclose arm mappings or paths."""

        with self._lock:
            sample_count = sum(len(row["samples"]) for row in self._public_rows)
            return {
                "dataset_id": self._metadata.get("dataset_id"),
                "case_count": len(self._public_rows),
                "sample_count": sample_count,
                "required_annotators": 2,
                "dimension_count": len(GOLDEN_DIMENSIONS),
                "annotation_granularity": self._metadata.get("annotation_granularity", "case_sample"),
                "arms_hidden": self._metadata.get("arms_hidden") is True,
                "patch_selection_allowed": self._metadata.get("patch_selection_allowed"),
                "comparison_only": self._metadata.get("comparison_only") is True,
                "render_prompt_mismatch_count": self._metadata.get("render_prompt_mismatch_count", 0),
                "status": self._metadata.get("status"),
            }

    def _read_score_rows(self) -> list[dict[str, Any]]:
        if not self.scores_path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(self.scores_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid score JSON at line {line_number}") from exc
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def public_progress(self, annotator_id: str) -> dict[str, dict[str, bool]]:
        """Return completed sample keys for one annotator."""

        progress: dict[str, dict[str, bool]] = {}
        with self._lock:
            for row in self._read_score_rows():
                if str(row.get("annotator_id")) != annotator_id:
                    continue
                case_id = str(row.get("case_id") or "")
                sample_id = str(row.get("sample_id") or "")
                if case_id in self._samples and sample_id in self._samples[case_id]:
                    progress[f"{case_id}/{sample_id}"] = {"saved": True}
        return progress

    def get_score(self, case_id: str, sample_id: str, annotator_id: str) -> dict[str, Any] | None:
        with self._lock:
            found = None
            for row in self._read_score_rows():
                if (
                    str(row.get("case_id")) == case_id
                    and str(row.get("sample_id")) == sample_id
                    and str(row.get("annotator_id")) == annotator_id
                ):
                    found = row
            return json.loads(json.dumps(found)) if found is not None else None

    def _validate_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("score payload must be an object")
        leaked = {str(key).lower() for key in payload} & LEAK_KEYS
        if leaked:
            raise ValueError(f"score payload contains forbidden metadata: {sorted(leaked)}")
        case_id = str(payload.get("case_id") or "")
        sample_id = str(payload.get("sample_id") or "")
        annotator_id = str(payload.get("annotator_id") or "").strip()
        if case_id not in self._samples:
            raise ValueError("unknown case_id")
        if sample_id not in self._samples[case_id]:
            raise ValueError("unknown sample_id")
        if not annotator_id or len(annotator_id) > 100:
            raise ValueError("annotator_id must be 1-100 characters")
        scores = payload.get("scores")
        if not isinstance(scores, dict) or set(scores) != set(GOLDEN_DIMENSIONS):
            raise ValueError("scores must contain exactly the 14 golden dimensions")
        clean_scores: dict[str, float | int] = {}
        for dimension in GOLDEN_DIMENSIONS:
            value = scores[dimension]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{dimension} must be a finite numeric score")
            if not 0 <= float(value) <= 100:
                raise ValueError(f"{dimension} must be in the 0-100 range")
            clean_scores[dimension] = value
        pass_fail = str(payload.get("pass_fail") or "borderline")
        if pass_fail not in PASS_FAIL_VALUES:
            raise ValueError("pass_fail must be pass, borderline, or fail")
        owner = str(payload.get("primary_failure_owner") or "none")
        if owner not in ALLOWED_FAILURE_OWNERS:
            raise ValueError("unknown primary_failure_owner")
        confidence = payload.get("confidence", 0.0)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError("confidence must be in the 0-1 range")
        return {
            "review_version": "golden-review-ui-v1",
            "case_id": case_id,
            "sample_id": sample_id,
            "annotator_id": annotator_id,
            "scores": clean_scores,
            "pass_fail": pass_fail,
            "primary_failure_owner": owner,
            "visible_evidence": _split_lines(payload.get("visible_evidence"), "visible_evidence"),
            "weaknesses": _split_lines(payload.get("weaknesses"), "weaknesses"),
            "confidence": round(float(confidence), 4),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

    def save_score(self, payload: Any) -> dict[str, Any]:
        """Validate and atomically upsert one case/sample/annotator record."""

        clean = self._validate_payload(payload)
        key = (clean["case_id"], clean["sample_id"], clean["annotator_id"])
        with self._lock:
            rows = self._read_score_rows()
            replaced = False
            output: list[dict[str, Any]] = []
            for row in rows:
                row_key = (str(row.get("case_id")), str(row.get("sample_id")), str(row.get("annotator_id")))
                if row_key == key:
                    if not replaced:
                        output.append(clean)
                        replaced = True
                    continue
                output.append(row)
            if not replaced:
                output.append(clean)
            self.scores_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(
                prefix="human_scores.", suffix=".tmp", dir=str(self.scores_path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    for row in output:
                        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self.scores_path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        return clean

    def media_path(self, case_id: str, sample_id: str) -> Path:
        if case_id not in self._samples or sample_id not in self._samples[case_id]:
            raise ValueError("unknown media sample")
        path = self.bundle / "videos" / case_id / f"{sample_id}.mp4"
        if not path.is_file():
            raise ValueError("sampled video is missing")
        return _resolve_inside(self.bundle, path)

    def frame_path(self, case_id: str, sample_id: str, index: int) -> Path:
        paths = self._frames.get((case_id, sample_id), [])
        if index < 0 or index >= len(paths):
            raise ValueError("unknown frame")
        path = paths[index]
        if not path.is_file():
            raise ValueError("sampled frame is missing")
        return _resolve_inside(self.bundle, path)

    def sheet_path(self, case_id: str) -> Path:
        path = self._sheets.get(case_id)
        if path is None or not path.is_file():
            raise ValueError("unknown contact sheet")
        return _resolve_inside(self.bundle, path)


class _ReviewHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def _content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {".mp4": "video/mp4", ".png": "image/png", ".html": "text/html; charset=utf-8"}.get(
        suffix, "application/octet-stream"
    )


def _handler_for(store: GoldenReviewStore):
    class ReviewHandler(BaseHTTPRequestHandler):
        server_version = "GoldenReviewApp/1.0"

        def log_message(self, _format: str, *_args: Any) -> None:
            return

        def _json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, *, allow_range: bool = False) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            status = 200
            range_header = self.headers.get("Range") if allow_range else None
            if range_header:
                if not range_header.startswith("bytes=") or "," in range_header:
                    self._json(416, {"error": "unsupported range"})
                    return
                start_text, end_text = range_header[6:].split("-", 1)
                try:
                    if start_text:
                        start = int(start_text)
                    else:
                        length = int(end_text)
                        start = max(0, size - length)
                    if end_text and start_text:
                        end = int(end_text)
                    end = min(end, size - 1)
                except ValueError:
                    self._json(416, {"error": "invalid range"})
                    return
                if start < 0 or start >= size or end < start:
                    self._json(416, {"error": "range not satisfiable"})
                    return
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", _content_type(path))
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            parsed = urlparse(self.path)
            route = parsed.path
            try:
                if route == "/":
                    self._file(Path(__file__).parent / "golden_review_ui" / "index.html")
                    return
                if route == "/api/manifest":
                    self._json(200, store.public_manifest())
                    return
                if route == "/api/metadata":
                    self._json(200, store.public_metadata())
                    return
                if route == "/api/progress":
                    annotator_id = parse_qs(parsed.query).get("annotator_id", [""])[0]
                    self._json(200, store.public_progress(annotator_id))
                    return
                if route == "/api/score":
                    query = parse_qs(parsed.query)
                    row = store.get_score(
                        query.get("case_id", [""])[0],
                        query.get("sample_id", [""])[0],
                        query.get("annotator_id", [""])[0],
                    )
                    self._json(200, row)
                    return
                parts = [unquote(part) for part in route.strip("/").split("/")]
                if len(parts) == 3 and parts[0] == "media" and parts[2].endswith(".mp4"):
                    self._file(store.media_path(parts[1], parts[2][:-4]), allow_range=True)
                    return
                if len(parts) == 4 and parts[0] == "frames":
                    self._file(store.frame_path(parts[1], parts[2], int(parts[3])))
                    return
                if len(parts) == 2 and parts[0] == "sheets" and parts[1].endswith(".png"):
                    self._file(store.sheet_path(parts[1][:-4]))
                    return
                self._json(404, {"error": "not found"})
            except (OSError, ValueError) as exc:
                self._json(404, {"error": str(exc)})

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            if urlparse(self.path).path != "/api/score":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("request body must be between 1 byte and 1 MB")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(200, store.save_score(payload))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

    return ReviewHandler


def create_server(bundle: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    """Create a local review server without starting its event loop."""

    store = GoldenReviewStore(bundle)
    return _ReviewHTTPServer((host, port), _handler_for(store))


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the blind golden-review video scoring UI")
    parser.add_argument("--bundle", default="dataset/golden-review-exact-v2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = parser.parse_args()
    server = create_server(args.bundle, host=args.host, port=args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Golden review UI: {url}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
