"""OpenAI-compatible Responses API VLM provider for real proxy frames."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .schemas import VLMJudgeResponse


VLM_MODEL_ALIASES = {
    "gpt-5.6-Luna": "gpt-5.6-luna",
    "gpt-5.6-Terra": "gpt-5.6-terra",
    "gpt-5.6-luna": "gpt-5.6-luna",
    "gpt-5.6-terra": "gpt-5.6-terra",
}


def normalize_vlm_model(model: str) -> str:
    return VLM_MODEL_ALIASES.get(model, model)


class VLMUnavailable(RuntimeError):
    """Raised when the configured VLM cannot be called."""


def _image_data_url(path: str | Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_responses_payload(
    *,
    prompt: str,
    scene_contract: Any,
    frame_paths: list[str | Path],
    deterministic_findings: list[Any],
    model: str,
    harness_version: str | None = None,
) -> dict[str, Any]:
    del harness_version
    if hasattr(scene_contract, "model_dump"):
        scene_contract = scene_contract.model_dump(mode="json")
    normalized_findings = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in deterministic_findings
    ]
    context = {
        "prompt": prompt,
        "scene_contract": scene_contract,
        "deterministic_findings": normalized_findings,
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": json.dumps(context, sort_keys=True)
            + "\nFrame order is chronological. Judge only visible evidence and do not infer unseen frames.",
        }
    ]
    for index, path in enumerate(frame_paths, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"Chronological sample {index}/{len(frame_paths)}; source frame {Path(path).stem}.",
            }
        )
        content.append({"type": "input_image", "image_url": _image_data_url(path), "detail": "low"})
    return {
        "model": model,
        "store": False,
        "temperature": 0,
        "max_output_tokens": 800,
        "instructions": (
            "Evaluate the real Blender proxy video frames against the prompt and contract. "
            "Return only the JSON schema. Score 0-100 with evidence for every dimension. "
            "Prompt compliance checks event order and required scene semantics. Physical plausibility checks "
            "support-before-grasp, attachment, no penetration, and release. Camera coverage checks whether "
            "follow/orbit/dolly/hold shots make required events observable. Camera innovation specifically judges "
            "whether choreography uses meaningful orbit/dolly/reframing rather than a static camera. "
            "Character trajectory checks phase progression and continuity; object trajectory checks grasp-carry-place "
            "path and attachment visibility; event timing checks temporal alignment. Temporal smoothness checks "
            "jumps and abrupt cuts. Visual clarity checks visibility and composition. Use conservative scores when "
            "evidence is missing; if a task dimension is explicitly not required by the contract, score it 100 and state "
            "that it is not applicable in visible evidence. Also return appearance_detail, physical_realism, "
            "spatial_consistency, motion_naturalness, and visual_presentation for the separate realism channel. "
            "Do not infer geometry from the plan: score visible character/object detail, material/light response, "
            "contact and penetration, identity persistence, motion naturalness, and composition from the frames. "
            "Confidence is 0 to 1."
        ),
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "proxy_evaluator",
                "strict": True,
                "schema": VLMJudgeResponse.model_json_schema(),
            }
        },
    }


def parse_responses_json(response: dict[str, Any]) -> dict[str, Any]:
    text = response.get("output_text")
    if not text:
        for item in response.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    text = content["text"]
                    break
            if text:
                break
    if not text:
        raise VLMUnavailable("Responses API returned no output text")
    text = str(text).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise VLMUnavailable("Responses API output was not valid JSON") from exc


class OpenAIVLMProvider:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 120,
        opener: Callable[..., Any] | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        requested_model = model or os.getenv("OPENAI_VLM_MODEL") or "gpt-5.6-luna"
        self.model_alias = requested_model
        self.model = normalize_vlm_model(requested_model)
        self.timeout_s = timeout_s
        self.opener = opener or urllib.request.urlopen

    def evaluate(
        self,
        *,
        prompt: str,
        scene_contract: Any,
        frame_paths: list[str | Path],
        deterministic_findings: list[Any],
        harness_version: str | None = None,
    ) -> tuple[VLMJudgeResponse, dict[str, Any]]:
        if not self.api_key:
            raise VLMUnavailable("OPENAI_API_KEY is not configured")
        payload = build_responses_payload(
            prompt=prompt,
            scene_contract=scene_contract,
            frame_paths=frame_paths,
            deterministic_findings=deterministic_findings,
            model=self.model,
            harness_version=harness_version,
        )
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self.opener(request, timeout=self.timeout_s) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:400]
            except OSError:
                detail = ""
            raise VLMUnavailable(f"VLM request failed: HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise VLMUnavailable(f"VLM request failed: {type(exc).__name__}") from exc
        parsed = VLMJudgeResponse.model_validate(parse_responses_json(raw))
        return parsed, raw
