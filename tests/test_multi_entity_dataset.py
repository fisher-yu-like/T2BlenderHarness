from __future__ import annotations

import json


def test_multi_entity_builder_creates_frozen_50_60_30_dataset(tmp_path):
    from scripts.build_multi_entity_dataset import build_dataset
    from scripts.validate_multi_entity_dataset import validate_dataset

    metadata = build_dataset(tmp_path / "multi")
    report = validate_dataset(tmp_path / "multi")

    assert metadata["splits"] == {"train": 50, "dev": 60, "test": 30}
    assert report["status"] == "pass"
    assert report["case_count"] == 140
    assert report["family_overlap"] == []
    assert report["composition_overlap"] == []


def test_multi_entity_records_author_event_interaction_camera_and_negative_evidence(tmp_path):
    from scripts.build_multi_entity_dataset import build_dataset

    root = tmp_path / "multi"
    build_dataset(root)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len({record["prompt"] for record in records}) == 140
    for record in records:
        assert record["event_graph"]
        assert record["interactions"]
        assert record["camera_evidence"]
        assert record["negative_constraints"]
        assert record["prompt_hash"]
        assert record["composition_signature"]
        assert len([entity for entity in record["entities"] if entity["kind"] == "actor"]) >= 2
        assert len([entity for entity in record["entities"] if entity["kind"] == "prop"]) >= 2


def test_multi_entity_build_is_reproducible(tmp_path):
    from scripts.build_multi_entity_dataset import build_dataset

    first = build_dataset(tmp_path / "first")
    second = build_dataset(tmp_path / "second")

    assert first["fingerprint"] == second["fingerprint"]
