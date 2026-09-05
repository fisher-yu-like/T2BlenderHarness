from __future__ import annotations

import json

from PIL import Image


def test_payload_normalizes_canonical_model_to_endpoint_id(tmp_path):
    from evaluator.openai_vlm import build_responses_payload

    frame = tmp_path / "frame.png"
    Image.new("RGB", (4, 4), (255, 255, 255)).save(frame)
    payload = build_responses_payload(
        prompt="Observe.",
        frame_paths=[frame],
        model="gpt-5.6-Terra",
        frame_metadata=[{"frame": 1, "timecode": "00:00:00.000"}],
    )

    assert payload["model"] == "gpt-5.6-terra"
    assert "gpt-5.6-Terra" not in json.dumps(payload)


def test_shared_review_uses_canonical_report_name(tmp_path):
    from evaluator.deterministic import DeterministicReport
    from evaluator.schemas import VLMJudgeResponse
    from evaluator.shared_review import score_shared_visual_review

    response = VLMJudgeResponse(
        prompt_compliance=80,
        physical_plausibility=80,
        camera_coverage=80,
        camera_innovation=80,
        character_trajectory=80,
        object_trajectory=80,
        event_timing=80,
        temporal_smoothness=80,
        visual_clarity=80,
        appearance_detail=80,
        physical_realism=80,
        spatial_consistency=80,
        motion_naturalness=80,
        visual_presentation=80,
        visible_evidence=["frame"],
        weaknesses=[],
        confidence=0.9,
    )
    deterministic = DeterministicReport(terminal_status="pass", hard_gate_failed=False, score=90)
    result = score_shared_visual_review(
        tmp_path,
        deterministic=deterministic,
        response=response,
        source="gpt-5.6-terra",
        frame_paths=[],
        video_probe={"playable": True},
        model="gpt-5.6-terra",
    )

    assert result["vlm_model"] == "gpt-5.6-terra"
    assert result["review_source"] == "gpt-5.6-terra"
