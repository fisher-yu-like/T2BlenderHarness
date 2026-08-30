"""Respond to a pending Director request with an authored interpretation.

The driving session reads the request (original prompt + obligations), decides
the semantics NOW, and writes a small authored spec; this tool performs only
mechanical work - evidence span location, id computation, contract validation
against the PromptInterpretation schema with the request's exact prompt, and
the response file write.  Authored specs failing validation are rejected
before the provider consumes anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from director_prompt import PromptInterpretation  # noqa: E402
from director_contracts import DirectorRequest  # noqa: E402

UNC_STYLE = {
    "id": "unc_visual_style",
    "description": "The benchmark prompt does not specify material, wardrobe, or set dressing; a neutral visual prior is applied.",
    "severity": "soft",
    "resolved": False,
}


def build_interpretation(prompt: str, spec: dict) -> dict:
    entities: list[dict] = []
    evidence: list[dict] = []
    for item in spec["entities"]:
        entity_id, kind, role, label, quote, hint = item
        ev_id = f"ev_entity_{entity_id}"
        entities.append({
            "id": entity_id, "kind": kind, "role": role, "label": label,
            "attributes": {"source_evidence_id": ev_id, "visual_hint": hint or label},
        })
        if quote is None:
            evidence.append({
                "id": ev_id, "source": "policy", "prompt_span": None,
                "quoted_text": None, "claim": f"{label} is staged as the {role}",
            })
        else:
            start = prompt.find(quote)
            if start < 0:
                raise SystemExit(f"quote {quote!r} not in prompt")
            evidence.append({
                "id": ev_id, "source": "prompt", "prompt_span": [start, start + len(quote)],
                "quoted_text": quote, "claim": f"{label} appears in the prompt",
            })
    for ev_id, quote, claim in spec.get("action_evidence", []):
        start = prompt.find(quote)
        if start < 0:
            raise SystemExit(f"action quote {quote!r} not in prompt")
        evidence.append({
            "id": ev_id, "source": "prompt", "prompt_span": [start, start + len(quote)],
            "quoted_text": quote, "claim": claim,
        })
    directives = [
        {
            "id": d["id"], "action": d["action"],
            **{k: v for k, v in d.items() if k in {"actor_id", "prop_id", "receiver_id"} and v},
            "evidence_id": d["evidence_id"],
        }
        for d in spec["directives"]
    ]
    camera_cues = [
        {"id": f"cue_{i + 1:02d}", "action": c["action"], "direction": c.get("direction"),
         "evidence_id": c["evidence_id"]}
        for i, c in enumerate(spec.get("camera", []))
    ]
    return {
        "entities": entities,
        "directives": directives,
        "camera_cues": camera_cues,
        "evidence": evidence,
        "assumptions": spec.get("assumptions", ["The action is staged so it stays framed."]),
        "uncertainties": [UNC_STYLE],
        **({"color_transition": spec["color_transition"]} if spec.get("color_transition") else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, help="pending director request JSON")
    parser.add_argument("--spec", required=True, help="authored spec JSON for this prompt")
    parser.add_argument("--session-root", default="out/assistant-session")
    args = parser.parse_args()

    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    payload = request.get("payload") or {}
    request_obj = payload.get("request") or {}
    prompt = str(request_obj.get("prompt") or "")
    if not prompt:
        raise SystemExit("request payload carries no prompt")
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    interpretation = build_interpretation(prompt, spec)

    # Validate against the exact contract the DirectorAgent will apply.
    director_request = DirectorRequest(
        prompt=prompt,
        scene_id=str(request_obj.get("scene_id") or request.get("scene_id") or "unknown"),
        duration_s=float(request_obj.get("duration_s") or 10.0),
        fps=int(request_obj.get("fps") or 12),
        provider=str(request_obj.get("provider") or "assistant-session-glm-flash"),
        policy=str(request_obj.get("policy") or "director-v5-glm-structured"),
        obligations=request_obj.get("obligations") or {},
    )
    PromptInterpretation.model_validate({"request": director_request.model_dump(mode="json"), **interpretation})

    response_path = Path(request["respond_to"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(interpretation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "responded": str(response_path),
        "scene_id": request_obj.get("scene_id"),
        "prompt": prompt[:80],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
