from __future__ import annotations

import json

import pytest


def test_default_owner_challenge_set_has_five_positive_negative_pairs_per_owner() -> None:
    from videoact.owner_challenges import build_default_challenge_set, validate_owner_challenges

    fixtures = build_default_challenge_set()
    report = validate_owner_challenges(fixtures)

    assert report.status == "pass"
    assert report.owner_count == 7
    assert report.fixture_count == 70
    assert report.pair_count == 35
    assert all(value == 5 for value in report.pairs_per_owner.values())
    assert all(fixture.split == "train" for fixture in fixtures)


def test_owner_challenge_validation_rejects_missing_pair_and_cross_split() -> None:
    from videoact.owner_challenges import build_default_challenge_set, validate_owner_challenges

    fixtures = build_default_challenge_set()
    with pytest.raises(ValueError, match="positive/negative pair"):
        validate_owner_challenges(fixtures[:-1])

    tampered = [fixture.model_copy(update={"split": "dev"}) if index == 0 else fixture for index, fixture in enumerate(fixtures)]
    with pytest.raises(ValueError, match="train-only"):
        validate_owner_challenges(tampered)


def test_owner_challenge_validation_rejects_near_duplicate_prompt_with_different_owner() -> None:
    from videoact.owner_challenges import build_default_challenge_set, validate_owner_challenges

    fixtures = build_default_challenge_set()
    duplicate = fixtures[0].model_copy(
        update={
            "fixture_id": "tampered-duplicate",
            "pair_id": "tampered-pair",
            "owner": "director_camera",
        }
    )

    with pytest.raises(ValueError, match="prompt collision"):
        validate_owner_challenges([*fixtures, duplicate])


def test_owner_challenge_manifest_round_trip(tmp_path) -> None:
    from videoact.owner_challenges import (
        build_default_challenge_set,
        load_owner_challenges,
        write_owner_challenges,
    )

    destination = tmp_path / "manifest.jsonl"
    write_owner_challenges(destination, build_default_challenge_set())
    loaded = load_owner_challenges(destination)

    assert [item.fixture_id for item in loaded] == [item.fixture_id for item in build_default_challenge_set()]
    assert json.loads(destination.read_text(encoding="utf-8").splitlines()[0])["split"] == "train"
