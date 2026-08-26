import json


def test_benchmark_cache_key_binds_prompt_harness_and_evaluator():
    from scripts.run_benchmark import cache_key

    first = cache_key("prompt", "h1", "e1", "fake", "seed-1")
    same = cache_key("prompt", "h1", "e1", "fake", "seed-1")
    changed = cache_key("prompt", "h2", "e1", "fake", "seed-1")

    assert first == same
    assert first != changed
    assert len(first) == 64


def test_fake_benchmark_is_reproducible(tmp_path):
    from scripts.run_benchmark import run_benchmark

    first = run_benchmark("train", "fake", tmp_path / "first")
    second = run_benchmark("train", "fake", tmp_path / "second")

    assert first == second
    assert first["case_count"] == 20
    assert first["aggregate"]["pass_rate"] == 1.0
    assert first["aggregate"]["mean_score"] == 96.4
    assert (tmp_path / "first" / "benchmark_report.json").exists()


def test_benchmark_protocol_is_checked_in():
    from pathlib import Path

    protocol = Path("docs/benchmark-protocol.md").read_text(encoding="utf-8")

    assert "train" in protocol
    assert "dev" in protocol
    assert "test" in protocol
    assert "test split" in protocol.lower()
