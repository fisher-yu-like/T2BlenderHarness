from __future__ import annotations

import json

from videoact.code_cache import CodeCache


def test_code_cache_stores_and_reuses_frozen_source(tmp_path) -> None:
    cache = CodeCache(tmp_path / "code-cache")
    source = "from blender.lib.geometry import box\n"

    frozen_path = cache.store("plan-hash", "agent-codegen-v1", source, llm_call_id="call-1")

    assert frozen_path.is_file()
    assert cache.lookup("plan-hash", "agent-codegen-v1") == source
    assert cache.lookup("plan-hash", "agent-codegen-v2") is None
    entries = [json.loads(line) for line in cache.manifest_path.read_text(encoding="utf-8").splitlines()]
    assert entries[0]["plan_hash"] == "plan-hash"
    assert entries[0]["harness_version"] == "agent-codegen-v1"
    assert entries[0]["code_hash"]
    assert entries[0]["llm_call_id"] == "call-1"


def test_code_cache_does_not_allow_unsafe_version_as_a_path(tmp_path) -> None:
    cache = CodeCache(tmp_path / "code-cache")

    path = cache.store("plan-hash", "../outside", "source", llm_call_id="call-2")

    assert path.parent == cache.cache_dir
    assert cache.lookup("plan-hash", "../outside") == "source"


def test_code_cache_exposes_previous_hashes_for_cross_batch_duplicate_gate(tmp_path) -> None:
    cache = CodeCache(tmp_path / "code-cache")
    cache.store("plan-a", "h1", "source-a", llm_call_id="call-a")
    cache.store("plan-b", "h2", "source-b", llm_call_id="call-b")

    hashes = cache.frozen_source_hashes("h1")

    assert set(hashes) == {"plan-a"}
    assert len(hashes["plan-a"]) == 64
