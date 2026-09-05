from __future__ import annotations

import json
from pathlib import Path


def test_resumable_six_round_protocol_skips_completed_rounds_and_retests_only_baseline_final(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.train_real_harness as training

    dataset_root = Path("dataset/vbench2-agent-training-index-v1")
    test_dataset_root = Path("dataset/vbench2-agent-test-100-v1")
    root = tmp_path / "diagnostic-six-rounds"
    for round_number in (1, 2):
        round_dir = root / f"round-{round_number:02d}"
        round_dir.mkdir(parents=True)
        (round_dir / "round_report.json").write_text(
            json.dumps({"round": round_number, "attempt": {"round": round_number}}),
            encoding="utf-8",
        )

    outer_calls: list[int] = []
    test_calls: list[int] = []

    monkeypatch.setattr(training, "build_dynamic_codex_agents", lambda **_kwargs: (object(), object()))
    monkeypatch.setattr(training, "update_training_memory_table", lambda *args, **kwargs: None)

    def fake_outer(*args, **kwargs):
        round_number = kwargs["round_number"]
        outer_calls.append(round_number)
        return {
            "round": round_number,
            "attempt": 1,
            "batch": {"train": [], "dev": []},
            "splits": {"train": {}, "dev": {}},
        }

    def fake_test(*args, **kwargs):
        round_number = kwargs["round_number"]
        test_calls.append(round_number)
        return {"round": round_number, "split": "test", "case_ids": []}

    monkeypatch.setattr(training, "run_resumable_outer_attempt", fake_outer)
    monkeypatch.setattr(training, "run_resumable_round_test", fake_test)

    result = training.run_resumable_six_round_protocol(
        root,
        dataset_root=dataset_root,
        test_dataset_root=test_dataset_root,
        blender_bin="blender",
        workers=1,
        timeout_s=1,
        provider_timeout_s=1,
        vlm_model="gpt-5.6-luna",
        codex_command="codex",
        markdown_path=tmp_path / "memory.md",
        provider_mode="model",
        test_schedule="baseline_final_only",
        diagnostic_only=True,
    )

    assert outer_calls == [3, 4, 5, 6]
    assert test_calls == [0, 6]
    assert result["status"] == "complete"
    assert result["test_schedule"] == "baseline_final_only"
    assert result["completed_round_count"] == 6
