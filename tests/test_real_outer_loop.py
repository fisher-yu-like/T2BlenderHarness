import json


def test_real_outer_loop_records_no_patch_when_train_and_dev_have_no_failures(tmp_path):
    from scripts.run_real_outer_loop import run_outer_loop

    for split in ("train", "dev"):
        root = tmp_path / split
        case = root / "case-01"
        case.mkdir(parents=True)
        (case / "run_manifest.json").write_text(
            json.dumps({"case_id": "case-01"}), encoding="utf-8"
        )
        (case / "deterministic_report.json").write_text(
            json.dumps(
                {
                    "terminal_status": "pass",
                    "hard_gate_failed": False,
                    "score": 100.0,
                    "findings": [],
                }
            ),
            encoding="utf-8",
        )

    result = run_outer_loop(tmp_path / "train", tmp_path / "dev", tmp_path / "outer.json")

    assert result["status"] == "no_patch"
    assert result["train"]["mean_score"] == 100.0
    assert result["dev"]["mean_score"] == 100.0
