import base64
import json

from PIL import Image


def test_responses_payload_contains_prompt_contract_and_image_data_url_without_harness_identity(tmp_path):
    from evaluator.openai_vlm import build_responses_payload

    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(frame)
    payload = build_responses_payload(
        prompt="Show a grasp.",
        frame_paths=[frame],
        model="vision-model",
        video_path=tmp_path / "proxy.mp4",
        frame_metadata=[{"frame": 1, "timecode": "00:00:00.000"}],
    )

    assert payload["model"] == "vision-model"
    assert payload["store"] is False
    content = payload["input"][0]["content"]
    image_item = next(item for item in content if item["type"] == "input_image")
    assert image_item["image_url"].startswith("data:image/png;base64,")
    assert any("Chronological sample 1/1" in item.get("text", "") for item in content)
    assert "secret-harness" not in json.dumps(payload)


def test_response_text_parser_extracts_strict_json_from_responses_output():
    from evaluator.openai_vlm import parse_responses_json

    expected = {"prompt_compliance": 90, "physical_plausibility": 90}
    raw = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(expected)}]}]}

    assert parse_responses_json(raw) == expected
