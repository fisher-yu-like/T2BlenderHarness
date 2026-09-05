from __future__ import annotations

import pytest


def test_vlm_rsi_protocol_enables_vlm_and_keeps_only_baseline_final_test() -> None:
    from scripts.train_real_harness import prepare_vlm_rsi_protocol

    protocol = {
        "rounds": [
            {"round": round_number, "test_evaluation": {"scheduled": True}}
            for round_number in range(1, 7)
        ]
    }

    result = prepare_vlm_rsi_protocol(
        protocol,
        visual_provider=object(),
        test_schedule="baseline_final_only",
    )

    assert result["execution_mode"] == "ai_only_vlm_rsi"
    assert result["formal_training_allowed"] is False
    assert result["visual_scores_permitted"] is True
    assert result["vlm_required"] is True
    assert result["patch_controller"] == "outer_transition_controller"
    assert result["patch_executor"] == "patch_executor"
    assert [
        item["test_evaluation"]["scheduled"] for item in result["rounds"]
    ] == [False, False, False, False, False, True]


def test_vlm_rsi_protocol_rejects_a_missing_visual_provider() -> None:
    from scripts.train_real_harness import prepare_vlm_rsi_protocol

    with pytest.raises(ValueError, match="visual_provider"):
        prepare_vlm_rsi_protocol(
            {"rounds": []},
            visual_provider=None,
            test_schedule="baseline_final_only",
        )


def test_vlm_rsi_transition_rejects_test_evidence_before_controller() -> None:
    from videoact.vlm_rsi import VlmRsiTransitionController

    controller = VlmRsiTransitionController(output_dir=".")

    with pytest.raises(ValueError, match="train-only"):
        controller._validate_train_reports(
            [{"split": "test", "case_id": "test-01"}]
        )


def test_vlm_rsi_patch_executor_can_rollback_after_dev_regression(tmp_path) -> None:
    from videoact.patch_executor import PatchExecutor

    source = tmp_path / "src" / "videoact" / "director_camera.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'before'\n", encoding="utf-8")
    executor = PatchExecutor(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
        owner_challenge_runner=lambda: True,
        unit_test_runner=lambda: True,
    )

    result = executor.execute(
        {
            "owner": "director_camera",
            "root_cause_id": "camera_visibility",
            "affected_files": ["src/videoact/director_camera.py"],
            "source_split": "train",
            "source_case_ids": ["train-01", "train-02"],
        },
        {
            "file_contents": {
                "src/videoact/director_camera.py": "VALUE = 'after'\n"
            }
        },
    )

    assert result["status"] == "accepted"
    rollback = executor.rollback_last()

    assert rollback["status"] == "rolled_back"
    assert source.read_text(encoding="utf-8") == "VALUE = 'before'\n"


def test_vlm_rsi_runner_is_explicitly_model_backed_and_has_no_visual_fallback() -> None:
    text = (
        __import__("pathlib").Path("scripts/run_vlm_rsi_six_rounds.py")
        .read_text(encoding="utf-8")
    )

    assert "ai_only_vlm_rsi=True" in text
    assert "CodexVisualReviewProvider" in text
    assert 'fallback_model=""' in text
    assert "CodexHarnessPatchAgent" in text
    assert "VlmRsiTransitionController" in text


def test_vlm_provider_records_requested_model_separately_from_review_source() -> None:
    from evaluator.codex_visual import CodexVisualReviewProvider

    provider = CodexVisualReviewProvider(
        model="gpt-5.6-luna",
        fallback_model="",
    )

    assert provider.model_alias == "gpt-5.6-luna"
    assert provider.review_source == "codex_local_visual_review"


def test_ai_only_vlm_rsi_rounds_pass_real_visual_provider_and_disable_local_proxy(
    monkeypatch, tmp_path
) -> None:
    import scripts.train_real_harness as training

    sentinel = object()
    seen: list[dict] = []

    def fake_attempt(*_args, **kwargs):
        seen.append(kwargs)
        empty = {"aggregate": {}, "real_video_count": 0, "vlm_scored_count": 0}
        return {
            "round": kwargs["round_number"],
            "attempt": kwargs["attempt_number"],
            "splits": {"train": empty, "dev": empty},
        }

    def fake_test(*_args, **kwargs):
        seen.append(kwargs)
        return {"round": kwargs["round_number"], "split": "test", "case_ids": []}

    monkeypatch.setattr(training, "run_outer_attempt", fake_attempt)
    monkeypatch.setattr(training, "run_round_test", fake_test)
    monkeypatch.setattr(training, "update_training_memory_table", lambda *args, **kwargs: None)

    result = training.run_six_round_protocol(
        tmp_path / "vlm-rsi",
        dataset_root="dataset/vbench2-agent-training-index-v1",
        test_dataset_root="dataset/vbench2-agent-test-100-v1",
        harness_version="test-vlm-rsi",
        evaluator_version="visual-primary-v7",
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        provider_timeout_s=1,
        vlm_model="gpt-5.6-luna",
        markdown_path=tmp_path / "memory.md",
        provider_mode="model",
        outer_transition=lambda *_args: {"action": "accept", "status": "accepted", "reason": "test"},
        diagnostic_only=False,
        visual_provider=sentinel,
        test_schedule="baseline_final_only",
        ai_only_vlm_rsi=True,
    )

    assert result["execution_mode"] == "ai_only_vlm_rsi"
    assert result["visual_scores_permitted"] is True
    assert result["vlm_required"] is True
    outer_rounds = [
        item["round_number"]
        for item in seen
        if "attempt_number" in item
    ]
    test_rounds = [
        item["round_number"]
        for item in seen
        if "test_case_ids" in item
    ]
    assert outer_rounds == [1, 2, 3, 4, 5, 6]
    assert test_rounds == [0, 6]
    assert all(item.get("visual_provider") is sentinel for item in seen)
    assert all(item.get("assistant_local") is False for item in seen)


def test_vlm_motion_and_appearance_dimensions_can_create_harness_findings() -> None:
    from videoact.failure_extractor import FailureExtractor

    record = {
        "case_id": "train-motion-01",
        "split": "train",
        "vlm_report": {
            "status": "scored",
            "review_source": "codex_local_visual_review",
            "visual_primary": {
                "status": "scored",
                "confidence": 0.9,
                "motion_naturalness": 20.0,
                "appearance_detail": 25.0,
                "dimension_evidence": {
                    "motion_naturalness": {
                        "evidence_completeness": 1.0,
                        "evidence_refs": ["frames/frame_0001.png"],
                    },
                    "appearance_detail": {
                        "evidence_completeness": 1.0,
                        "evidence_refs": ["frames/frame_0002.png"],
                    },
                },
            },
        },
    }

    evidence = FailureExtractor(visual_failure_threshold=60.0).extract(record)

    assert {item.owner_candidate for item in evidence if item.actionable} == {
        "director_trajectory",
        "blender_code_agent",
    }
