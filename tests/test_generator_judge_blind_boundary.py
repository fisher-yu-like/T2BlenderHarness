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
        video_path=tmp_path / "proxy.mp4",
        frame_metadata=[{"frame": 1, "timecode": "00:00:00.000"}],
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
        frame_paths=[frame],
        model="gpt-5.6-terra",
        video_path=tmp_path / "proxy.mp4",
        frame_metadata=[{"frame": 1, "timecode": "00:00:00.000"}],
    )
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert '"scene_contract"' not in serialized
    assert '"deterministic_findings"' not in serialized
    assert "hidden-harness" not in serialized
    assert "primary-blind-v1" in serialized


def test_formal_judge_caller_passes_only_blind_evidence_inputs(tmp_path, monkeypatch) -> None:
    from evaluator.schemas import VLMJudgeResponse
    from scripts.evaluate_real_videos import evaluate_vlm_run

    frames = tmp_path / "frames" / "animation"
    frames.mkdir(parents=True)
    for number in (1, 2, 3):
        Image.new("RGB", (4, 4), (number, 20, 30)).save(frames / f"frame_{number:06d}.png")
    (tmp_path / "deterministic_report.json").write_text(
        json.dumps({"terminal_status": "pass", "hard_gate_failed": False, "score": 90, "findings": []}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.evaluate_real_videos.probe_mp4",
        lambda *_args, **_kwargs: {"playable": True, "frame_count": 3, "fps": 24.0, "duration_s": 0.125},
    )
    seen: dict[str, object] = {}

    class InspectingProvider:
        model_alias = "gpt-5.6-luna"

        def evaluate(self, **kwargs):
            seen.update(kwargs)
            return VLMJudgeResponse(
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
                visible_evidence=["frame_000001.png"],
                weaknesses=[],
                confidence=0.9,
            ), {"id": "inspected"}

    result = evaluate_vlm_run(
        tmp_path,
        prompt="A hidden plan must not be supplied to the judge.",
        scene_contract={"director_plan": "secret", "events": [], "fps": 24},
        provider=InspectingProvider(),
        scoring_policy="legacy-aggregate",
    )

    assert result["status"] == "scored"
    assert set(seen) == {"prompt", "video_path", "frame_paths", "frame_metadata"}
    assert "secret" not in json.dumps(seen, default=str)
