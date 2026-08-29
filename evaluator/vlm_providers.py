"""Provider adapters for evidence-based multi-frame VLM review.

The adapters are deliberately small: they build an OpenAI-compatible
chat-completions request and validate the returned JSON against the shared
``VLMJudgeResponse`` contract.  Transport errors are represented as
``unavailable`` and never become a numeric quality score.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .schemas import VLMJudgeResponse
from .openai_vlm import VLMUnavailable


REVIEW_FIELDS = (
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


def _data_url(path: str | Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _as_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("VLM response did not contain a JSON object")


@dataclass(frozen=True)
class OpenAICompatibleVLMAdapter:
    """One named model adapter over the shared OpenAI-compatible protocol."""

    model: str
    endpoint_path: str = "/chat/completions"

    def build_payload(
        self,
        *,
        prompt: str,
        frame_paths: list[str | Path],
        scene_contract: Any,
        deterministic_findings: list[Any],
    ) -> dict[str, Any]:
        # Keep the primary visual judge blind to all generator-side artifacts.
        # The arguments remain in the public API for compatibility, but are
        # intentionally not serialized into the request.
        del scene_contract, deterministic_findings
        context = {
            "blind_review_version": "primary-blind-v1",
            "prompt": prompt,
            "required_dimensions": list(REVIEW_FIELDS),
        }
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Review the chronological Blender-rendered proxy frames against the exact prompt. "
                    "Do not infer an event from the plan unless visible in a frame. Return only the "
                    "strict JSON schema. Score each requested dimension as an integer from 0 to 100, "
                    "include visible_evidence, weaknesses, and an event_scores object keyed by each "
                    "distinct required event that can be identified from the exact prompt; use null when "
                    "the event is not independently visible. Set confidence from 0 to 1. "
                    "Distinguish low-poly proxy appearance limitations from absent action, contact, "
                    "timing, trajectory, camera, or physical evidence.\n"
                    + json.dumps(context, ensure_ascii=False, sort_keys=True)
                ),
            }
        ]
        for index, path in enumerate(frame_paths, start=1):
            content.append({"type": "text", "text": f"Chronological frame {index}/{len(frame_paths)}."})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(path), "detail": "low"},
                }
            )
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1200,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a conservative visual evaluator. Missing visible evidence is not a pass.",
                },
                {"role": "user", "content": content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "proxy_evaluator",
                    "strict": True,
                    "schema": VLMJudgeResponse.model_json_schema(),
                },
            },
        }

    def parse_response(self, response: dict[str, Any]) -> VLMJudgeResponse:
        choices = response.get("choices") or []
        if not choices:
            raise ValueError("chat-completions response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(str(item.get("text") or "") for item in content if isinstance(item, dict))
        payload = _as_json(content)
        try:
            return VLMJudgeResponse.model_validate(payload)
        except Exception as original_error:
            # Some OpenAI-compatible upstreams ignore response_format and
            # return one object per dimension.  Expand that representation
            # without inventing missing values or confidence.
            normalized: dict[str, Any] = {}
            evidence: list[str] = []
            weaknesses: list[str] = []
            score_payload = payload.get("scores")
            if not isinstance(score_payload, dict):
                score_payload = payload
            for field in REVIEW_FIELDS:
                value = score_payload.get(field)
                if isinstance(value, dict) and "score" in value:
                    normalized[field] = value["score"]
                    visible = value.get("visible_evidence")
                    weak = value.get("weaknesses")
                    if visible:
                        evidence.extend(visible if isinstance(visible, list) else [str(visible)])
                    if weak:
                        weaknesses.extend(weak if isinstance(weak, list) else [str(weak)])
                elif value is not None:
                    normalized[field] = value
            if "visible_evidence" in payload:
                visible = payload["visible_evidence"]
                evidence.extend(visible if isinstance(visible, list) else [str(visible)])
            if "weaknesses" in payload:
                weak = payload["weaknesses"]
                weaknesses.extend(weak if isinstance(weak, list) else [str(weak)])
            normalized["visible_evidence"] = list(dict.fromkeys(str(item) for item in evidence if str(item).strip()))
            normalized["weaknesses"] = list(dict.fromkeys(str(item) for item in weaknesses if str(item).strip()))
            if "confidence" in payload:
                normalized["confidence"] = payload["confidence"]
            try:
                return VLMJudgeResponse.model_validate(normalized)
            except Exception:
                raise original_error


PROVIDERS: dict[str, OpenAICompatibleVLMAdapter] = {
    "gpt-5.6-luna": OpenAICompatibleVLMAdapter(model="gpt-5.6-luna"),
    "gpt-5.6-terra": OpenAICompatibleVLMAdapter(model="gpt-5.6-terra"),
}


def _default_transport(payload: dict[str, Any], *, base_url: str, api_key: str, timeout_s: float) -> dict[str, Any]:
    normalized_base = base_url.rstrip("/")
    if not normalized_base.endswith("/v1"):
        normalized_base += "/v1"
    request = urllib.request.Request(
        f"{normalized_base}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(str(exc)) from exc


class OpenAICompatibleVLMProvider:
    """Evaluator-facing provider using the documented chat-completions path."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        selected_model = model or os.getenv("OPENAI_VLM_MODEL", "gpt-5.6-luna")
        if selected_model.lower() not in PROVIDERS:
            raise ValueError(f"unsupported VLM model: {selected_model}")
        self.model = selected_model.lower()
        self.model_alias = self.model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://ai-pixel.online/v1")
        self.timeout_s = timeout_s
        self.transport = transport

    def evaluate(
        self,
        *,
        prompt: str,
        scene_contract: Any,
        frame_paths: list[str | Path],
        deterministic_findings: list[Any],
        harness_version: str | None = None,
    ) -> tuple[VLMJudgeResponse, dict[str, Any]]:
        del harness_version
        result = dispatch_vlm(
            model=self.model,
            prompt=prompt,
            frame_paths=frame_paths,
            scene_contract=scene_contract,
            deterministic_findings=deterministic_findings,
            transport=self.transport,
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_s=self.timeout_s,
        )
        if result.get("status") != "complete":
            raise VLMUnavailable(str(result.get("reason") or "vlm_unavailable"))
        return VLMJudgeResponse.model_validate(result["response"]), result.get("raw_response") or {}


def dispatch_vlm(
    *,
    model: str,
    prompt: str,
    frame_paths: list[str | Path],
    scene_contract: Any,
    deterministic_findings: list[Any],
    transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    """Call a named adapter and return a fail-closed serializable result."""
    adapter = PROVIDERS.get(model.lower())
    if adapter is None:
        return {"status": "unavailable", "score": None, "reason": "unknown_model", "model": model.lower()}
    try:
        payload = adapter.build_payload(
            prompt=prompt,
            frame_paths=frame_paths,
            scene_contract=scene_contract,
            deterministic_findings=deterministic_findings,
        )
        if transport is None:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError("api_key_not_configured")
            raw = _default_transport(
                payload,
                base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=key,
                timeout_s=timeout_s,
            )
        else:
            raw = transport(payload)
        parsed = adapter.parse_response(raw)
    except ValueError:
        return {"status": "unavailable", "score": None, "reason": "schema_error", "model": adapter.model}
    except Exception:
        return {"status": "unavailable", "score": None, "reason": "transport_error", "model": adapter.model}
    return {
        "status": "complete",
        "score": None,
        "model": adapter.model,
        "response": parsed.model_dump(mode="json"),
        "raw_response": raw,
    }
