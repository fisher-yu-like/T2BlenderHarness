import json

from PIL import Image


def complex_prompt():
    return (
        "Establish the table and red cup. The character walks to the support, reaches for the object, grasps it, "
        "lifts it clear, carries it to the drop zone, places it, and releases it. Follow the approach, orbit during "
        "carry, and dolly into the release close-up."
    )


def test_camera_evaluator_checks_required_motion_intents_not_only_event_coverage():
    from evaluator.camera_metrics import check_camera_motion_intent
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build(complex_prompt(), duration_s=16.0, fps=24)
    plan = TrajectoryPlanner().plan(contract)
    broken_shots = [shot.model_copy(update={"trajectory_type": "hold"}) for shot in plan.camera.shots]
    broken_plan = plan.model_copy(update={"camera": plan.camera.model_copy(update={"shots": broken_shots})})

    findings = check_camera_motion_intent(contract, broken_plan)

    assert {finding.failure_id for finding in findings} >= {"camera_orbit_intent_missing", "camera_dolly_intent_missing"}


def test_trajectory_evaluator_checks_camera_coupled_and_phase_primitives():
    from evaluator.trajectory_metrics import check_trajectory_phase_alignment
    from videoact.scene_contract import SceneContractBuilder
    from videoact.trajectory import TrajectoryPlanner

    contract = SceneContractBuilder().build(complex_prompt(), duration_s=16.0, fps=24)
    plan = TrajectoryPlanner().plan(contract)
    character = plan.entities["character"].model_copy(update={"motion_primitives": []})
    broken_plan = plan.model_copy(update={"entities": {**plan.entities, "character": character}})

    findings = check_trajectory_phase_alignment(contract, broken_plan)

    assert any(finding.failure_id == "trajectory_phase_primitive_missing" for finding in findings)


def test_vlm_schema_explicitly_scores_camera_and_character_object_trajectories():
    from evaluator.schemas import VLMJudgeResponse

    response = VLMJudgeResponse.model_validate(
        {
            "prompt_compliance": 80,
            "physical_plausibility": 75,
            "camera_coverage": 70,
            "camera_innovation": 65,
            "character_trajectory": 85,
            "object_trajectory": 80,
            "event_timing": 75,
            "temporal_smoothness": 70,
            "visual_clarity": 90,
            "visible_evidence": ["orbit during carry", "release close-up"],
            "weaknesses": ["brief occlusion"],
            "confidence": 0.8,
        }
    )

    assert response.camera_innovation == 65
    assert response.character_trajectory == 85
    assert response.object_trajectory == 80
    assert response.event_timing == 75


def test_evaluation_frame_sampling_supplements_three_index_frames_with_timeline_frames(tmp_path):
    from videoact.real_artifacts import sample_frame_paths

    frames = tmp_path / "frames"
    animation = frames / "animation"
    animation.mkdir(parents=True)
    indexed = []
    for frame in range(1, 25):
        path = animation / f"frame_{frame:06d}.png"
        Image.new("RGB", (4, 4), (frame, 0, 0)).save(path)
        if frame in (1, 12, 24):
            sample = frames / f"frame_{frame:06d}.png"
            Image.new("RGB", (4, 4), (frame, 0, 0)).save(sample)
            indexed.append({"frame": frame, "path": sample.name})
    (frames / "index.json").write_text(json.dumps({"frames": indexed}), encoding="utf-8")

    sampled = sample_frame_paths(tmp_path, max_frames=8)

    assert len(sampled) == 8
    assert len({path.name for path in sampled}) == 8


def test_event_aligned_sampling_prioritizes_event_midpoints_and_endpoints(tmp_path):
    from videoact.real_artifacts import sample_event_aligned_frame_paths

    animation = tmp_path / "frames" / "animation"
    animation.mkdir(parents=True)
    for frame in range(1, 101):
        Image.new("RGB", (4, 4), (frame % 255, 0, 0)).save(animation / f"frame_{frame:04d}.png")
    contract = {
        "fps": 10,
        "must_show": ["grasp", "release"],
        "events": [
            {"id": "grasp", "start": 1.8, "end": 2.2},
            {"id": "release", "start": 7.8, "end": 8.2},
        ],
    }

    sampled = sample_event_aligned_frame_paths(tmp_path, contract, max_frames=4)
    numbers = [int(path.stem.rsplit("_", 1)[-1]) for path in sampled]

    assert numbers == [1, 21, 81, 100]
