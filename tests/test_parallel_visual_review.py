from __future__ import annotations

import json
from pathlib import Path


def test_evaluate_split_uses_independent_provider_sessions_in_parallel(tmp_path: Path, monkeypatch) -> None:
    import scripts.evaluate_real_videos as module

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    records = []
    run_dirs = []
    for index in range(3):
        case_id = f"case-{index + 1}"
        records.append({"case_id": case_id, "prompt": f"prompt {index + 1}"})
        run_dir = tmp_path / "runs" / case_id
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text(json.dumps({"case_id": case_id}), encoding="utf-8")
        (run_dir / "scene_contract.json").write_text(json.dumps({"case_id": case_id}), encoding="utf-8")
        run_dirs.append(run_dir)
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    class FakeProvider:
        def __init__(self, clones: list[FakeProvider] | None = None) -> None:
            self.clones = clones if clones is not None else []

        def clone(self) -> FakeProvider:
            clone = FakeProvider(self.clones)
            self.clones.append(clone)
            return clone

    base = FakeProvider()
    used: list[FakeProvider] = []

    def fake_evaluate(*args, provider, **kwargs):
        used.append(provider)
        return {"status": "scored", "task_score": 1.0}

    monkeypatch.setattr(module, "discover_run_dirs", lambda _root: run_dirs)
    monkeypatch.setattr(module, "evaluate_vlm_run", fake_evaluate)

    results = module.evaluate_split(
        tmp_path / "runs",
        dataset,
        provider=base,
        max_workers=3,
    )

    assert [item["case_id"] for item in results] == ["case-1", "case-2", "case-3"]
    assert len(used) == 3
    assert len({id(provider) for provider in used}) == 3
    assert all(provider is not base for provider in used)
