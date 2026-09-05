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


def test_real_outer_loop_extracts_visual_bottleneck_even_when_deterministic_passes(tmp_path):
    from scripts.run_real_outer_loop import run_outer_loop

    for split, case_ids in (("train", ("train-01", "train-02")), ("dev", ("dev-01",))):
        root = tmp_path / split
        for case_id in case_ids:
            case = root / case_id
            case.mkdir(parents=True)
            (case / "run_manifest.json").write_text(
                json.dumps({"case_id": case_id, "split": split}), encoding="utf-8"
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
            if split == "train":
                (case / "vlm_report.json").write_text(
                    json.dumps(
                        {
                            "status": "scored",
                            "review_source": "gpt-5.6-luna",
                            "visual_primary": {
                                "status": "scored",
                                "camera_coverage": 40.0,
                                "confidence": 0.9,
                                "dimension_evidence": {
                                    "camera_coverage": {
                                        "evidence_completeness": 1.0,
                                        "evidence_refs": ["frames/frame_0010.png"],
                                    }
                                },
                            },
                            "sampled_frames": ["frames/frame_0010.png"],
                        }
                    ),
                    encoding="utf-8",
                )

    result = run_outer_loop(tmp_path / "train", tmp_path / "dev", tmp_path / "outer.json")

    assert result["status"] == "proposal_ready"
    assert result["proposal"]["owner"] == "director_camera"
    assert result["proposal"]["root_cause_id"] == "camera_visibility"
    assert all(record["findings"][0]["root_cause_id"] == "camera_visibility" for record in result["train_records"])
