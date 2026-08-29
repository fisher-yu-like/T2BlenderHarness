from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blind_mapping_is_reproducible_and_hides_arm_labels():
    module = _load("prepare_three_arm_reviews", ROOT / "scripts" / "prepare_three_arm_reviews.py")
    mapping = module.build_blind_mapping(
        [{"case_id": "case-1"}, {"case_id": "case-2"}],
        seed=17,
    )
    assert set(mapping) == {"case-1", "case-2"}
    assert set(mapping["case-1"]) == {"pretrain", "trained", "direct_code"}
    assert set(mapping["case-1"].values()) == {"sample_a", "sample_b", "sample_c"}
    assert mapping == module.build_blind_mapping([{"case_id": "case-1"}, {"case_id": "case-2"}], seed=17)


def test_review_request_contains_only_blind_sample_labels():
    module = _load("prepare_three_arm_reviews", ROOT / "scripts" / "prepare_three_arm_reviews.py")
    request = module.build_case_review_request(
        {"case_id": "case-1", "prompt": "Alice carries a red cube to Bob."},
        {
            "sample_a": ["C:/frames/a1.png"],
            "sample_b": ["C:/frames/b1.png"],
            "sample_c": ["C:/frames/c1.png"],
        },
    )
    assert set(request["samples"]) == {"sample_a", "sample_b", "sample_c"}
    serialized = str(request)
    assert "pretrain" not in serialized
    assert "trained" not in serialized
    assert "direct_code" not in serialized
    assert "action_variant" not in serialized

