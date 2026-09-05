from __future__ import annotations

import json


def test_streaming_case_command_forces_glm_generation_and_codex_review():
    from scripts.run_streaming_eval import build_case_command

    command = build_case_command(
        python_executable="python",
        dataset_root="dataset/vbench2-agent-test-100-v1",
        case_id="vbench2-test100-01-01",
        run_root="out/test-eval/harness-rsi-100-stream/job-01",
        blender_bin="D:/blender/blender.exe",
        timeout_s=600,
    )

    assert command[0] == "python"
    assert "--provider-mode" in command
    assert command[command.index("--provider-mode") + 1] == "glm"
    assert "--visual-review-provider" in command
    assert command[command.index("--visual-review-provider") + 1] == "codex"
    assert "--case-ids" in command
    assert command[command.index("--case-ids") + 1] == "vbench2-test100-01-01"
    assert "template_baseline" not in command


def test_streaming_summary_merges_per_case_reports(tmp_path):
    from scripts.run_streaming_eval import summarize_case_reports

    for case_id, score, status in (
        ("case-a", 80.0, "complete"),
        ("case-b", None, "incomplete_local_visual_review"),
    ):
        root = tmp_path / case_id
        root.mkdir()
        report = {
            "status": status,
            "case_count": 1,
            "real_video_count": 1 if score is not None else 0,
            "vlm_scored_count": 1 if score is not None else 0,
            "preparation_failed_count": 0 if score is not None else 1,
            "artifact_failed_count": 0 if score is not None else 1,
            "cases": [
                {
                    "case_id": case_id,
                    "video_exists": score is not None,
                    "overall_vlm_score": score,
                    "task_final_score": score,
                    "realism_score": score,
                    "vlm_status": "scored" if score is not None else "unavailable",
                }
            ],
        }
        (root / "real_unified_score.json").write_text(
            json.dumps(report), encoding="utf-8"
        )

    summary = summarize_case_reports(tmp_path)

    assert summary["case_count"] == 2
    assert summary["real_video_count"] == 1
    assert summary["vlm_scored_count"] == 1
    assert summary["preparation_failed_count"] == 1
    assert summary["aggregate"]["mean_task_final_score"] == 80.0
    assert summary["vlm_status_counts"] == {"scored": 1, "unavailable": 1}


def test_streaming_runner_records_worker_exception_instead_of_aborting_all_cases(tmp_path):
    from scripts.run_streaming_eval import safe_case_future_result

    class FailedFuture:
        def result(self):
            raise RuntimeError("worker crashed")

    result = safe_case_future_result(
        FailedFuture(),
        case_id="case-crashed",
        job_root=tmp_path / "case-crashed",
    )

    assert result["case_id"] == "case-crashed"
    assert result["return_code"] == -1
    assert result["report_exists"] is False
    assert result["error"] == "worker_exception:RuntimeError:worker crashed"
