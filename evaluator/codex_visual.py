"""Codex-provider visual review for real Blender proxy frames.

This provider keeps visual review on the local Codex boundary. It asks a
read-only ``codex exec`` subprocess to inspect the exact sampled frame files
and return the existing VLMJudgeResponse contract. No generated scene data,
deterministic findings, or template score is supplied to the reviewer.
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from videoact.codex_exec_provider import CodexExecProvider
from evaluator.openai_vlm import VLMUnavailable

from .schemas import VLMJudgeResponse


CODEX_VISUAL_REVIEW_SOURCE = "codex_local_visual_review"
CODEX_VISUAL_REVIEW_VERSION = "codex-exec-visual-review-v1"
_REVIEW_DIMENSIONS = (
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


class _VisualReviewFailureCircuit:
    """Share a terminal provider outage across per-case provider clones."""

    def __init__(self) -> None:
        self.reason: str | None = None

    def open(self, reason: str) -> None:
        if self.reason is None:
            self.reason = str(reason)

    @property
    def is_open(self) -> bool:
        return self.reason is not None


def visual_review_lock_path(frame_paths: list[str | Path]) -> Path | None:
    """Return the per-stream lock path used to serialize Codex visual calls."""

    configured = os.getenv("T2BLENDER_CODEX_VISUAL_LOCK_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    for path in frame_paths:
        for parent in Path(path).expanduser().resolve().parents:
            if parent.name.lower() == "jobs":
                return parent.parent / "codex_visual_review.lock"
    return None


@contextmanager
def _acquire_visual_review_slot(lock_path: Path | None, *, timeout_s: float = 1800.0) -> Iterator[None]:
    """Serialize local Codex image inspection across case subprocesses."""

    if lock_path is None:
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        acquired = False
        while not acquired:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - production target is Windows
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise VLMUnavailable(
                        f"codex_visual_review_concurrency_timeout:{lock_path}"
                    )
                time.sleep(0.25)
        try:
            yield
        finally:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - production target is Windows
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _contract_payload(scene_contract: Any) -> dict[str, Any]:
    if hasattr(scene_contract, "model_dump"):
        scene_contract = scene_contract.model_dump(mode="json")
    return scene_contract if isinstance(scene_contract, dict) else {}


def _frame_number(path: Path) -> int | None:
    match = re.search(r"frame_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _build_review_context(
    scene_contract: Any | None,
    frame_paths: list[str | Path],
    frame_metadata: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build frame/timecode metadata without passing plan semantics to Judge."""

    # ``scene_contract`` is accepted only for legacy direct callers. Formal
    # provider calls pass ``None`` and supply already-materialized timecodes.
    contract = _contract_payload(scene_contract)
    fps = max(1.0, float(contract.get("fps") or 24.0))
    frame_start = max(1, int(contract.get("frame_start") or 1))
    timeline = []
    supplied_by_frame = {
        int(item["frame"]): item
        for item in (frame_metadata or [])
        if isinstance(item, dict) and str(item.get("frame") or "").isdigit()
    }
    for raw_path in frame_paths:
        path = Path(raw_path).resolve()
        frame = _frame_number(path)
        item = dict(supplied_by_frame.get(frame, {})) if frame is not None else {}
        item.update({"path": str(path), "filename": path.name, "frame": frame})
        if "time_s" not in item:
            item["time_s"] = round((frame - frame_start) / fps, 4) if frame is not None else None
        timeline.append(item)
    must_show = {str(item) for item in contract.get("must_show") or []}
    events = []
    for raw_event in contract.get("events") or []:
        if not isinstance(raw_event, dict):
            continue
        event_id = str(raw_event.get("id") or "").strip()
        if not event_id or (must_show and event_id not in must_show):
            continue
        start_s = float(raw_event.get("start") or 0.0)
        end_s = float(raw_event.get("end") or start_s)
        events.append(
            {
                "id": event_id,
                "description": str(raw_event.get("description") or ""),
                "start_s": round(start_s, 4),
                "end_s": round(end_s, 4),
                "start_frame": max(frame_start, round(start_s * fps) + frame_start),
                "end_frame": max(frame_start, round(end_s * fps) + frame_start),
            }
        )
    return {"frame_timeline": timeline, "required_events": events}


def _codex_response_schema() -> dict[str, Any]:
    """Make per-dimension evidence a required, finite structured field."""

    schema = VLMJudgeResponse.model_json_schema()
    evidence_schema = dict(schema.get("$defs", {}).get("DimensionEvidence") or {})
    schema.setdefault("properties", {})["dimension_evidence"] = {
        "type": "object",
        "properties": {
            name: dict(evidence_schema)
            for name in _REVIEW_DIMENSIONS
        },
        "required": list(_REVIEW_DIMENSIONS),
        "additionalProperties": False,
    }
    return schema


class CodexVisualReviewProvider:
    """Use a local Codex structured call as the real-frame visual judge."""

    def __init__(
        self,
        *,
        command: str = "codex",
        timeout_s: int = 1800,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "low",
        fallback_model: str | None = None,
        fallback_timeout_s: int | None = None,
        visual_frame_budget: int = 8,
        _failure_circuit: _VisualReviewFailureCircuit | None = None,
    ) -> None:
        if isinstance(visual_frame_budget, bool) or int(visual_frame_budget) < 1:
            raise ValueError("visual_frame_budget must be a positive integer")
        self.command = command
        self.timeout_s = int(timeout_s)
        self._codex_model = model
        self._reasoning_effort = reasoning_effort
        self.visual_frame_budget = int(visual_frame_budget)
        default_fallback_model = "gpt-5.6-luna" if model == "gpt-5.6-terra" else "gpt-5.6-terra"
        self.fallback_model = (
            fallback_model
            if fallback_model is not None
            else os.getenv("CODEX_VLM_FALLBACK_MODEL") or default_fallback_model
        )
        self.fallback_timeout_s = int(
            fallback_timeout_s
            if fallback_timeout_s is not None
            else os.getenv("CODEX_VLM_FALLBACK_TIMEOUT_S") or min(self.timeout_s, 300)
        )
        self._failure_circuit = _failure_circuit or _VisualReviewFailureCircuit()
        self.review_source = CODEX_VISUAL_REVIEW_SOURCE
        # Keep the concrete requested model visible in reports while the
        # separate review_source field identifies the judging boundary.
        self.model_alias = str(model).strip().lower()
        self.model = "codex-local-visual-review"
        self.model_id = self.model
        self.model_version = CODEX_VISUAL_REVIEW_VERSION
        self.provider_kind = "codex_exec_visual_review"
        self.template_backed = False
        self.llm_generated = True
        # The desktop Codex host serializes its local image-inspection
        # sessions.  Keep this explicit so evaluate_split can still
        # parallelize providers that advertise independent process safety,
        # while this provider falls back to one review at a time.
        self.parallel_safe = False
        try:
            self._provider = CodexExecProvider(
                command=command,
                timeout_s=timeout_s,
                response_schema=_codex_response_schema(),
                prompt_builder=self._build_prompt,
                stage="visual_review",
                provider_kind=self.provider_kind,
                model_id=self.model_id,
                model_version=self.model_version,
                template_backed=False,
                llm_generated=True,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except TypeError as exc:
            # Keep this provider importable against older Codex host bridges;
            # the default bridge in this branch supports model selection.
            if "unexpected keyword argument 'model'" not in str(exc):
                raise
            self._provider = CodexExecProvider(
                command=command,
                timeout_s=timeout_s,
                response_schema=_codex_response_schema(),
                prompt_builder=self._build_prompt,
                stage="visual_review",
                provider_kind=self.provider_kind,
                model_id=self.model_id,
                model_version=self.model_version,
                template_backed=False,
                llm_generated=True,
            )

    def clone(self) -> "CodexVisualReviewProvider":
        """Create an isolated local Codex session for one case review."""

        return type(self)(
            command=self.command,
            timeout_s=self.timeout_s,
            model=self._codex_model,
            reasoning_effort=self._reasoning_effort,
            fallback_model=self.fallback_model,
            fallback_timeout_s=self.fallback_timeout_s,
            visual_frame_budget=self.visual_frame_budget,
            _failure_circuit=self._failure_circuit,
        )

    @property
    def call_records(self) -> list[dict[str, Any]]:
        return self._provider.call_records

    def last_call(self, stage: str | None = None) -> dict[str, Any] | None:
        return self._provider.last_call(stage or "visual_review")

    @staticmethod
    def _build_prompt(payload: Any) -> str:
        request = payload if isinstance(payload, dict) else {}
        frames = [str(Path(path).resolve()) for path in request.get("frame_paths", [])]
        frame_timeline = request.get("frame_timeline") or []
        required_events = request.get("required_events") or []
        return (
            "You are the local Codex visual evaluator for a real Blender proxy. "
            "Use your available read-only image inspection capability to open every supplied PNG "
            "in chronological order before scoring. Judge only visible evidence in those frames against "
            "the exact prompt; do not infer success from an unseen frame, plan, telemetry, or source. "
            "Use the supplied frame_timeline only to map an inspected filename to its frame and time, "
            "and use required_events only to know which event interval needs visual evidence; never "
            "assume that an event occurred because it is listed. For event_timing, inspect the frames "
            "around the listed interval and cite the concrete frame filenames when the timing is visible. "
            "For an observe/full-scene event with no discrete motion, visibility at the interval's "
            "start and end frames is valid timing evidence; cite those endpoints when the subject stays "
            "visible throughout the sampled interval. "
            "Return only one JSON object matching the supplied VLMJudgeResponse schema. Score every "
            "dimension from 0 to 100, provide visible_evidence and weaknesses, and set conservative "
            "confidence. For every required dimension, also return dimension_evidence with an entry "
            "containing confidence, evidence_completeness, and evidence_refs; evidence_refs must name "
            "the inspected frame files (for example frame_0001.png) and completeness must be 1 only "
            "when the frames visibly support that dimension. Return event_scores for every distinct "
            "required event when one is identifiable from the prompt, using null when it is not visibly "
            "decidable. Missing visual evidence must lower the relevant score or make event evidence "
            "unavailable; do not use a fixed template or a default score.\n"
            + json.dumps(
                {
                    "exact_prompt": str(request.get("prompt") or ""),
                    "chronological_frame_paths": frames,
                    "frame_timeline": frame_timeline,
                    "required_events": required_events,
                    "required_dimensions": [
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
                    ],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def _evaluate_once(
        self,
        *,
        prompt: str,
        frame_paths: list[str | Path],
        frame_metadata: list[dict[str, Any]] | None = None,
    ) -> tuple[VLMJudgeResponse, dict[str, Any]]:
        normalized_frames = [str(Path(path).resolve()) for path in frame_paths]
        review_context = _build_review_context(None, normalized_frames, frame_metadata)
        try:
            lock_path = visual_review_lock_path(normalized_frames)
            lock_timeout = os.getenv("T2BLENDER_CODEX_VISUAL_LOCK_TIMEOUT_S")
            try:
                wait_timeout_s = float(lock_timeout) if lock_timeout else 1800.0
            except ValueError:
                wait_timeout_s = 1800.0
            with _acquire_visual_review_slot(lock_path, timeout_s=wait_timeout_s):
                raw = self._provider(
                    {
                        "prompt": prompt,
                        "frame_paths": normalized_frames,
                        "_codex_image_paths": normalized_frames,
                        **review_context,
                    }
                )
            response = VLMJudgeResponse.model_validate(raw)
        except VLMUnavailable:
            raise
        except Exception as exc:
            raise VLMUnavailable(
                f"codex_visual_review_unavailable:{type(exc).__name__}:{exc}"
            ) from exc
        return response, self.last_call() or {}

    def evaluate(
        self,
        *,
        prompt: str,
        frame_paths: list[str | Path],
        video_path: str | Path | None = None,
        frame_metadata: list[dict[str, Any]] | None = None,
        scene_contract: Any | None = None,
        deterministic_findings: list[Any] | None = None,
        harness_version: str | None = None,
    ) -> tuple[VLMJudgeResponse, dict[str, Any]]:
        del video_path, scene_contract, deterministic_findings, harness_version
        if self._failure_circuit.is_open:
            raise VLMUnavailable(
                "codex_visual_review_circuit_open:"
                + str(self._failure_circuit.reason)
            )
        try:
            return self._evaluate_once(
                prompt=prompt,
                frame_paths=frame_paths,
                frame_metadata=frame_metadata,
            )
        except VLMUnavailable as primary_error:
            fallback_model = str(self.fallback_model or "").strip()
            if not fallback_model or fallback_model == self._codex_model:
                self._failure_circuit.open(str(primary_error))
                raise
            fallback = type(self)(
                command=self.command,
                timeout_s=self.fallback_timeout_s,
                model=fallback_model,
                reasoning_effort=self._reasoning_effort,
                fallback_model=None,
                fallback_timeout_s=self.fallback_timeout_s,
                visual_frame_budget=self.visual_frame_budget,
            )
            try:
                response, raw = fallback._evaluate_once(
                    prompt=prompt,
                    frame_paths=frame_paths,
                    frame_metadata=frame_metadata,
                )
            except VLMUnavailable as fallback_error:
                self._failure_circuit.open(str(fallback_error))
                raise VLMUnavailable(
                    f"{primary_error}; fallback_model={fallback_model}; fallback_error={fallback_error}"
                ) from fallback_error
            raw = dict(raw or {})
            raw["fallback_from_model"] = self._codex_model
            raw["fallback_model"] = fallback_model
            return response, raw


__all__ = [
    "CODEX_VISUAL_REVIEW_SOURCE",
    "CODEX_VISUAL_REVIEW_VERSION",
    "CodexVisualReviewProvider",
    "visual_review_lock_path",
]
