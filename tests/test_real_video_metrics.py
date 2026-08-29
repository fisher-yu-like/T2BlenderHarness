from __future__ import annotations

import json

from PIL import Image

from videoact.real_video import assemble_mp4_from_pngs


def _write_video_run(tmp_path, *, with_observations: bool = True):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True)
    frames = []
    for frame_number in range(1, 9):
        path = frames_dir / f"frame_{frame_number:06d}.png"
        image = Image.new("RGB", (64, 64), (30 + frame_number * 4, 60, 90))
        image.save(path)
        frames.append(path)
    assemble_mp4_from_pngs(frames, tmp_path / "proxy.mp4", fps=4)

    observations = []
    if with_observations:
        for frame_number in range(1, 9):
            x = (frame_number - 1) * 0.1
            observations.append(
                {
                    "frame": frame_number,
                    "camera": {"location": [4.0 + x, -8.0, 4.0], "rotation": [0.0, 0.0, 0.0]},
                    "entities": {
                        "actor_a": {
                            "root_location": [x, 0.0, 0.0],
                            "world_bbox": [[x - 0.4, -0.4, 0.0], [x + 0.4, 0.4, 2.0]],
                            "screen_bbox": [0.25 + x * 0.02, 0.2, 0.55 + x * 0.02, 0.9],
                            "visible_fraction": 1.0,
                        },
                        "prop_01_cup": {
                            "root_location": [x + 0.7, 0.0, 0.8],
                            "world_bbox": [[x + 0.5, -0.2, 0.6], [x + 0.9, 0.2, 1.2]],
                            "screen_bbox": [0.55 + x * 0.02, 0.4, 0.75 + x * 0.02, 0.65],
                            "visible_fraction": 1.0,
                        },
                    },
                }
            )
    (tmp_path / "telemetry.json").write_text(
        json.dumps({"runtime_observations": observations}),
        encoding="utf-8",
    )
    trajectory = {
        "timebase": {"fps": 4, "frame_start": 1, "frame_end": 8},
        "entities": {
            "actor_a": {
                "states": [
                    {"frame": 1, "position": [0.0, 0.0, 0.0]},
                    {"frame": 8, "position": [0.7, 0.0, 0.0]},
                ],
                "motion_primitives": [{"type": "walk", "parameters": {"phase": "event_01"}}],
            },
            "prop_01_cup": {
                "states": [
                    {"frame": 1, "position": [0.7, 0.0, 0.8]},
                    {"frame": 8, "position": [1.4, 0.0, 0.8]},
                ],
                "motion_primitives": [{"type": "carry", "parameters": {"phase": "event_01"}}],
            },
        },
        "camera": {
            "shots": [
                {
                    "shot_id": "shot_01",
                    "start_frame": 1,
                    "end_frame": 8,
                    "trajectory_type": "follow",
                    "camera_cue": "follow",
                    "required_event_ids": ["event_01"],
                    "target_ids": ["actor_a", "prop_01_cup"],
                }
            ]
        },
    }
    contract = {
        "fps": 4,
        "events": [{"id": "event_01", "start": 0.0, "end": 2.0, "target_ids": ["actor_a", "prop_01_cup"]}],
        "must_show": ["event_01"],
        "entities": [
            {"id": "actor_a", "kind": "character"},
            {"id": "prop_01_cup", "kind": "prop"},
        ],
    }
    return tmp_path, contract, trajectory


def test_real_video_metrics_require_mp4_and_runtime_observations(tmp_path):
    from evaluator.real_video_metrics import evaluate_real_video

    root, contract, trajectory = _write_video_run(tmp_path / "complete")
    scored = evaluate_real_video(
        root,
        prompt="A person walks while carrying a cup.",
        scene_contract=contract,
        trajectory_plan=trajectory,
        telemetry=json.loads((root / "telemetry.json").read_text(encoding="utf-8")),
    )

    assert scored["status"] == "scored"
    assert scored["source"] == "actual_proxy_mp4_and_runtime_observations"
    assert scored["channels"]["visual_score"] is not None
    assert scored["channels"]["physical_score"] is not None
    assert scored["channels"]["trajectory_score"] is not None
    assert scored["channels"]["camera_score"] is not None
    assert scored["dimensions"]["object_trajectory"] < 100 or scored["dimensions"]["character_trajectory"] < 100
    assert scored["decoded_video"]["sampled_frame_count"] == 8


def test_real_video_metrics_fail_closed_without_runtime_observations(tmp_path):
    from evaluator.real_video_metrics import evaluate_real_video

    root, contract, trajectory = _write_video_run(tmp_path / "missing-observations", with_observations=False)
    result = evaluate_real_video(
        root,
        prompt="A person walks while carrying a cup.",
        scene_contract=contract,
        trajectory_plan=trajectory,
        telemetry={"runtime_observations": []},
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "runtime_observations_missing"
    assert result["channels"] == {
        "visual_score": None,
        "physical_score": None,
        "trajectory_score": None,
        "camera_score": None,
    }
