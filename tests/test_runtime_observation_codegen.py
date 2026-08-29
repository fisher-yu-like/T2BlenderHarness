from __future__ import annotations


def test_local_codegen_emits_runtime_observation_capture_for_real_video_scoring():
    from videoact.codex_self_provider import CodexLocalProvider

    plan = {
        "request": {"scene_id": "case-observation", "prompt": "A person walks beside a cup.", "duration_s": 2.0, "fps": 4},
        "entities": [
            {"id": "actor_a", "kind": "actor", "label": "person", "role": "participant"},
            {"id": "prop_01_cup", "kind": "prop", "label": "cup", "role": "target_object"},
            {"id": "support_surface", "kind": "support", "label": "support", "role": "environment"},
        ],
        "events": [{"id": "event_01_walk", "action": "walk", "start": 0.0, "end": 2.0, "participant_ids": ["actor_a"], "target_ids": ["prop_01_cup"]}],
        "camera_plan": {"shots": [{"shot_id": "shot_01", "start_frame": 1, "end_frame": 8, "trajectory_type": "follow", "camera_cue": "follow", "required_event_ids": ["event_01_walk"], "target_ids": ["actor_a", "prop_01_cup"]}]},
        "trajectory_summary": {
            "timebase": {"fps": 4, "frame_start": 1, "frame_end": 8},
            "entities": {
                "actor_a": {"states": [{"frame": 1, "position": [0, 0, 0]}, {"frame": 8, "position": [1, 0, 0]}], "motion_primitives": []},
                "prop_01_cup": {"states": [{"frame": 1, "position": [0.8, 0, 0.8]}, {"frame": 8, "position": [1.8, 0, 0.8]}], "motion_primitives": []},
                "support_surface": {"states": [{"frame": 1, "position": [0, 0, 0]}], "motion_primitives": []},
            },
        },
    }

    response = CodexLocalProvider().codegen({"director_plan": plan, "model": "codex-local"})
    source = response["generated_code"]

    assert "runtime_observations" in source
    assert "capture_runtime_observations" in source
    assert "screen_bbox" in source
    assert "world_bbox" in source
