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
        scene_contract: Any,
        selected_frames: list[str],
        deterministic_findings: list[Any],
        harness_version: str | None = None,
    ) -> dict[str, Any]:
        del harness_version  # Harness identity must not become a VLM cue.
        if hasattr(scene_contract, "model_dump"):
            scene_contract = scene_contract.model_dump(mode="json")
        deterministic_findings = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in deterministic_findings
        ]
        return {
            "prompt": prompt,
            "scene_contract": scene_contract,
            "selected_frames": list(selected_frames),
            "deterministic_findings": deterministic_findings,
        }

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
