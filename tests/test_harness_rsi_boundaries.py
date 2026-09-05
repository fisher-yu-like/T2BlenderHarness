from __future__ import annotations

from pathlib import Path

import pytest


def _finding_record(
    case_id: str,
    *,
    split: str | None = "train",
    status: str = "pass",
    evidence: list[str] | None = None,
    root_cause_id: str = "camera:coverage",
) -> dict:
    return {
        "case_id": case_id,
        "split": split,
        "status": status,
        "findings": [
            {
                "failure_id": "camera_event_uncovered",
                "owner": "camera_planner",
                "category": "camera_coverage",
                "severity": "warning",
                "message": "the required event is not sufficiently visible",
                "root_cause_id": root_cause_id,
                "evidence": evidence if evidence is not None else [f"{case_id}/report.json"],
                "repair_route": "camera_repair",
            }
        ],
    }


def test_pass_train_cases_with_repeated_evidence_can_trigger_a_patch(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    proposal = MetaHarnessOptimizer(output_dir=tmp_path).propose(
        [
            _finding_record("train-01", status="pass"),
            _finding_record("train-02", status="pass"),
        ]
    )

    assert proposal.owner == "camera_planner"
    assert proposal.source_split == "train"
    assert proposal.source_case_ids == ["train-01", "train-02"]


def test_patch_proposal_requires_explicit_train_records_even_when_cases_pass(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    with pytest.raises(ValueError, match="train-only"):
        MetaHarnessOptimizer(output_dir=tmp_path).propose(
            [
                _finding_record("case-01", split=None),
                _finding_record("case-02", split=None),
            ]
        )


def test_findings_without_case_evidence_cannot_trigger_a_patch(tmp_path):
    from videoact.meta_harness import MetaHarnessOptimizer

    with pytest.raises(ValueError, match="evidence"):
        MetaHarnessOptimizer(output_dir=tmp_path).propose(
            [
                _finding_record("train-01", evidence=[]),
                _finding_record("train-02", evidence=[]),
            ]
        )


def test_agent_provenance_failure_isolated_to_the_offending_case():
    from scripts.train_real_harness import _provenance_failed_case_ids

    assert _provenance_failed_case_ids(
        {
            "status": "fail",
            "failures": ["case-b:provider_manifest_not_complete"],
        },
        ["case-a", "case-b"],
    ) == {"case-b"}

    assert _provenance_failed_case_ids(
        {"status": "fail", "failures": ["probable_template_reuse"]},
        ["case-a", "case-b"],
    ) == {"case-a", "case-b"}


def test_retryable_generation_failure_is_not_relabelled_as_provenance_failure():
    from scripts.train_real_harness import _provenance_failed_case_ids

    provenance = {
        "status": "fail",
        "failures": [
            "case-a:job_status=coverage_failed",
            "case-a:missing_codegen_call_id",
            "case-a:provider_manifest_not_complete",
        ],
    }

    assert _provenance_failed_case_ids(
        provenance,
        ["case-a"],
        retryable_case_ids={"case-a"},
    ) == set()


def test_batch_keeps_verified_cases_when_one_case_fails_provenance(monkeypatch, tmp_path):
    import json

    import scripts.train_real_harness as module

    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.jsonl").write_text(
        "\n".join(
            json.dumps({"case_id": case_id, "split": "test", "prompt": case_id})
            for case_id in ("case-a", "case-b")
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "metadata.json").write_text('{"fingerprint": "fixture"}\n', encoding="utf-8")
    prepared_calls: list[list[str]] = []
    render_calls: list[list[str]] = []

    def fake_prepare(split, root, **kwargs):
        pending = list(kwargs["case_ids"])
        prepared_calls.append(pending)
        return {
            "generation_mode": "agent",
            "provider_mode": "glm",
            "jobs": [
                {"case_id": case_id, "status": "prepared"}
                for case_id in pending
            ],
        }

    def fake_audit(index, *, run_root, expected_case_ids):
        return {
            "status": "fail",
            "failures": ["case-b:provider_manifest_not_complete"],
        }

    def fake_render(root, *, case_ids, **kwargs):
        render_calls.append(list(case_ids))
        for case_id in case_ids:
            case_root = Path(root) / case_id
            case_root.mkdir(parents=True, exist_ok=True)
            (case_root / "run_manifest.json").write_text(
                json.dumps({"case_id": case_id}), encoding="utf-8"
            )
            (case_root / "deterministic_report.json").write_text(
                json.dumps({"terminal_status": "pass", "hard_gate_failed": False}),
                encoding="utf-8",
            )
            (case_root / "proxy.mp4").write_bytes(b"video")
        return {
            "results": [
                {"case_id": case_id, "status": "success"}
                for case_id in case_ids
            ]
        }

    def fake_evaluate(run_dir, *, record, **kwargs):
        return {
            "case_id": record["case_id"],
            "status": "pass",
            "score": 1.0,
            "artifact_status": "complete",
            "proxy_video": str(Path(run_dir) / "proxy.mp4"),
            "findings": [],
        }

    def fake_merge(*, run_root, deterministic_results, vlm_results):
        return {
            "scoring_mode": "test",
            "case_count": len(deterministic_results),
            "real_video_count": sum(item.get("proxy_video") is not None for item in deterministic_results),
            "vlm_scored_count": 0,
            "cases": deterministic_results,
            "aggregate": {},
        }

    monkeypatch.setattr(module, "prepare_jobs", fake_prepare)
    monkeypatch.setattr(module, "audit_dynamic_agent_index", fake_audit)
    monkeypatch.setattr(module, "render_jobs", fake_render)
    monkeypatch.setattr(module, "evaluate_real_run", fake_evaluate)
    monkeypatch.setattr(module, "merge_real_scores", fake_merge)
    visual_case_ids: list[list[str] | None] = []

    def fake_visual_review(*args, **kwargs):
        requested = kwargs.get("case_ids")
        visual_case_ids.append(list(requested) if requested is not None else None)
        if requested is not None and "case-b" in requested:
            raise AssertionError("non-rendered cases must not be sent to visual review")
        return []

    monkeypatch.setattr(module, "evaluate_split", fake_visual_review)
    monkeypatch.setattr(module, "write_unified_outputs", lambda *args, **kwargs: None)

    result = module.run_real_batch_with_inner_loop(
        tmp_path / "run",
        split="test",
        case_ids=["case-a", "case-b"],
        dataset_root=dataset_root,
        harness_version="test",
        evaluator_version="test",
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        vlm_model="codex",
        director_agent=object(),
        code_agent=object(),
        provider_mode="glm",
        max_inner_attempts=1,
    )

    assert prepared_calls == [["case-a", "case-b"]]
    assert render_calls == [["case-a"]]
    assert result["inner_loop"]["completed_count"] == 1
    assert result["inner_loop"]["pending_case_ids"] == ["case-b"]
    assert visual_case_ids == [["case-a"]]


def test_outer_transition_rejects_a_patch_declared_from_dev():
    from scripts.train_real_harness import run_bounded_outer_attempts

    with pytest.raises(ValueError, match="train"):
        run_bounded_outer_attempts(
            run_attempt=lambda _attempt: {
                "train": {"findings": []},
                "dev": {"findings": []},
            },
            transition=lambda _attempt, _reports: {
                "action": "patch",
                "proposal": {
                    "owner": "camera_planner",
                    "source_split": "dev",
                    "affected_files": ["src/videoact/camera.py"],
                },
            },
            max_attempts=1,
        )


def test_optimize_mode_cannot_be_invoked_for_test_split(tmp_path):
    from scripts.run_real_pipeline import run_pipeline

    with pytest.raises(ValueError, match="only consume the train split"):
        run_pipeline(
            "optimize",
            split="test",
            out_dir=tmp_path,
            train_records=[
                _finding_record("test-01", split="train"),
                _finding_record("test-02", split="train"),
            ],
        )


def test_six_round_runner_stops_after_first_round_without_outer_transition(monkeypatch, tmp_path):
    import scripts.train_real_harness as module

    monkeypatch.setattr(module, "update_training_memory_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        module,
        "run_outer_attempt",
        lambda *args, **kwargs: {"splits": {"train": {}, "dev": {}}},
    )
    monkeypatch.setattr(
        module,
        "run_outer_overall",
        lambda *args, **kwargs: {"splits": {"train": {}, "dev": {}}},
    )
    monkeypatch.setattr(
        module,
        "run_round_test",
        lambda *args, **kwargs: {
            "round": kwargs["round_number"],
            "scope": "frozen_test_after_every_round",
            "split": "test",
            "case_ids": list(kwargs["test_case_ids"]),
        },
    )

    result = module.run_six_round_protocol(
        tmp_path / "rounds",
        dataset_root="dataset/vbench2-agent-training-index-v1",
        harness_version="harness-rsi-test",
        evaluator_version="scoring-v7-independent-channels",
        blender_bin="D:/blender/blender.exe",
        workers=1,
        timeout_s=1,
        vlm_model="codex_local_visual_review",
        director_agent=object(),
        code_agent=object(),
        provider_mode="glm",
        outer_transition=None,
        diagnostic_only=True,
    )

    assert result["status"] == "awaiting_harness_patch"
    assert len(result["rounds"]) == 1


def test_camera_only_environment_event_reaches_a_hold_trajectory():
    from videoact.director_contracts import (
        DirectorDecisionEvidence,
        DirectorEntity,
        DirectorRequest,
    )
    from videoact.director_prompt import DirectorCameraCue, PromptInterpretation
    from videoact.director_schedule import EventScheduler
    from videoact.director_trajectory import MultiEntityTrajectoryComposer

    request = DirectorRequest(
        prompt="Garden, zoom in.",
        scene_id="camera-only-boundary",
        duration_s=10.0,
        fps=12,
        provider="external-glm",
        policy="director-v5-glm-structured",
    )
    interpretation = PromptInterpretation(
        request=request,
        entities=[
            DirectorEntity(id="garden", kind="environment", role="setting", label="Garden")
        ],
        directives=[],
        camera_cues=[
            DirectorCameraCue(
                id="cam_zoom_in",
                action="zoom",
                direction="in",
                evidence_id="ev_zoom",
            )
        ],
        evidence=[
            DirectorDecisionEvidence(
                id="ev_zoom",
                source="prompt",
                prompt_span=(8, 15),
                quoted_text="zoom in",
                claim="camera zooms in",
            )
        ],
    )
    schedule = EventScheduler().schedule(request, interpretation)
    trajectories = MultiEntityTrajectoryComposer().compose(request, interpretation, schedule)

    garden = trajectories.entities["garden"]
    assert any(
        primitive.parameters.get("event_id") == "camera_observe_cam_zoom_in"
        for primitive in garden.motion_primitives
    )
