"""Schema-constrained prompt interpretation boundary.

The provider is injectable so a real structured-output model can be used in a
configured environment while tests and offline runs remain deterministic. All
provider claims are validated against the existing PromptInterpretation
contract, including exact prompt evidence spans.
"""

from __future__ import annotations

from typing import Any, Callable

from .director_contracts import DirectorRequest
from .director_prompt import PromptInterpretation


class StructuredPromptInterpreter:
    def __init__(self, provider: Callable[[DirectorRequest], dict[str, Any]]) -> None:
        self.provider = provider

    @staticmethod
    def build_request(request: DirectorRequest) -> dict[str, Any]:
        return {
            "prompt": request.prompt,
            "obligations": request.obligations,
            "response_schema": PromptInterpretation.model_json_schema(),
            "evidence_rule": "prompt_span and quoted_text must be an exact substring of prompt",
        }

    def interpret(self, request: DirectorRequest) -> PromptInterpretation:
        payload = self.provider(request)
        if not isinstance(payload, dict):
            raise ValueError("structured prompt provider must return an object")
        if "request" in payload:
            raise ValueError("structured prompt provider must not override request")
        return PromptInterpretation.model_validate(
            {"request": request.model_dump(mode="json"), **payload}
        )
