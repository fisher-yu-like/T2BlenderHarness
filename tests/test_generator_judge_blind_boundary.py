from __future__ import annotations

import json

from PIL import Image


def test_primary_chat_payload_does_not_include_generator_artifacts(tmp_path) -> None:
    from evaluator.vlm_providers import PROVIDERS

    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (120, 80, 40)).save(frame)
    payload = PROVIDERS["gpt-5.6-luna"].build_payload(
        prompt="A person walks to a red cup, lifts it, and carries it to a tray.",
        frame_paths=[frame],
        scene_contract={"director_plan": "must not leak"},
        deterministic_findings=[{"owner": "director_camera", "failure_id": "must not leak"}],
    )
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    for forbidden in (
        '"scene_contract"',
        '"director_plan"',
        '"deterministic_findings"',
        '"owner"',
        '"harness_version"',
    ):
        assert forbidden not in serialized
    assert payload["messages"][1]["content"][0]["type"] == "text"
    assert payload["messages"][1]["content"][0]["text"].count("A person walks") == 1


def test_primary_responses_payload_is_blind_to_contract_and_findings(tmp_path) -> None:
    from evaluator.openai_vlm import build_responses_payload

    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (20, 30, 40)).save(frame)
    payload = build_responses_payload(
        prompt="A ceramic bowl rotates while the camera orbits around it.",
        scene_contract={"director_plan": "hidden"},
        frame_paths=[frame],
        deterministic_findings=[{"failure_id": "hidden", "owner": "hidden"}],
        model="gpt-5.6-terra",
        harness_version="hidden-harness",
    )
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert '"scene_contract"' not in serialized
    assert '"deterministic_findings"' not in serialized
    assert "hidden-harness" not in serialized
    assert "primary-blind-v1" in serialized
