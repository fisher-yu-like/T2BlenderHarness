"""Optional VLM judge adapter with identity-blind input preparation."""

from __future__ import annotations

from typing import Any, Callable

from .schemas import VLMJudgeResponse


class VLMJudge:
    def __init__(self, provider: Callable[[dict[str, Any]], Any] | None = None):
        self.provider = provider

    @staticmethod
    def prepare_input(
        *,
        prompt: str,
        selected_frames: list[str],
        video_path: str | None = None,
        frame_metadata: list[dict[str, Any]] | None = None,
        scene_contract: Any | None = None,
        deterministic_findings: list[Any] | None = None,
        harness_version: str | None = None,
    ) -> dict[str, Any]:
        del harness_version  # Harness identity must not become a VLM cue.
        del scene_contract, deterministic_findings
        payload = {
            "prompt": prompt,
            "selected_frames": list(selected_frames),
        }
        if video_path is not None:
            payload["video_path"] = str(video_path)
        if frame_metadata is not None:
            payload["frame_metadata"] = list(frame_metadata)
        return payload

    def judge(self, **kwargs: Any) -> VLMJudgeResponse:
        if self.provider is None:
            raise RuntimeError("VLM provider is not configured")
        prepared = self.prepare_input(**kwargs)
        response = self.provider(prepared)
        return self.parse_response(response)

    @staticmethod
    def parse_response(response: Any) -> VLMJudgeResponse:
        if isinstance(response, VLMJudgeResponse):
            return response
        return VLMJudgeResponse.model_validate(response)
