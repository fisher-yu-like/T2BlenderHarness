from __future__ import annotations

import json

import pytest


def _proposal(**updates):
    payload = {
        "owner": "director_camera",
        "root_cause_id": "camera_visibility",
        "affected_files": ["src/videoact/director_camera.py"],
        "source_split": "train",
        "source_case_ids": ["train-one", "train-two"],
    }
    payload.update(updates)
    return payload


def _make_repo(tmp_path):
    source = tmp_path / "src" / "videoact" / "director_camera.py"
    source.parent.mkdir(parents=True)
    source.write_text("def compose_camera_plan():\n    return 'old'\n", encoding="utf-8")
    return source


def _patch():
    return {
        "diff": "diff --git a/src/videoact/director_camera.py b/src/videoact/director_camera.py\n+new\n",
        "changed_files": ["src/videoact/director_camera.py"],
    }


def test_executor_applies_one_owner_patch_after_ordered_gates(tmp_path):
    from videoact.patch_executor import PatchExecutor

    source = _make_repo(tmp_path)
    order: list[str] = []

    def apply(context):
        order.append("apply")
        (tmp_path / "src" / "videoact" / "director_camera.py").write_text(
            "def compose_camera_plan():\n    return 'new'\n", encoding="utf-8"
        )

    result = PatchExecutor(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
        owner_challenge_runner=lambda _context: order.append("challenge") or {"status": "pass"},
        unit_test_runner=lambda _context: order.append("unit") or {"status": "pass"},
        production_test_runner=lambda _context: order.append("production") or {"status": "pass"},
        blender_rerun_runner=lambda _context: order.append("blender") or {"status": "pass"},
    ).execute(
        _proposal(),
        _patch(),
        train_evidence=[{"case_id": "train-one", "split": "train", "finding": "camera"}],
        apply_callback=apply,
    )

    assert result["status"] == "accepted"
    assert result["owner"] == "director_camera"
    assert result["parent_hashes"]["src/videoact/director_camera.py"]
    assert result["child_hashes"]["src/videoact/director_camera.py"]
    assert result["diff_sha256"]
    assert order == ["apply", "challenge", "unit", "production", "blender"]
    assert "return 'new'" in source.read_text(encoding="utf-8")
    assert (tmp_path / "audit" / "patch_executions.jsonl").is_file()


def test_rejected_patch_restores_parent_and_keeps_failure_evidence(tmp_path):
    from videoact.patch_executor import PatchExecutor

    source = _make_repo(tmp_path)
    before = source.read_bytes()

    def apply(_context):
        source.write_text("def broken(:\n", encoding="utf-8")

    result = PatchExecutor(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
        owner_challenge_runner=lambda _context: {"status": "pass"},
        unit_test_runner=lambda _context: {"status": "fail"},
    ).execute(_proposal(), _patch(), apply_callback=apply)

    assert result["status"] == "rejected"
    assert result["restored"] is True
    assert source.read_bytes() == before
    evidence = tmp_path / "audit" / "camera_visibility.failure_evidence.json"
    assert evidence.is_file()
    assert "owner/unit test gate failed" in evidence.read_text(encoding="utf-8")
    events = [json.loads(line) for line in (tmp_path / "audit" / "patch_executions.jsonl").read_text().splitlines()]
    assert events[-1]["append_only"] is True


def test_undeclared_files_are_rejected_and_restored(tmp_path):
    from videoact.patch_executor import PatchExecutor

    source = _make_repo(tmp_path)
    extra = tmp_path / "src" / "videoact" / "other.py"

    def apply(_context):
        source.write_text("changed\n", encoding="utf-8")
        extra.write_text("changed\n", encoding="utf-8")

    result = PatchExecutor(repo_root=tmp_path, output_dir=tmp_path / "audit").execute(
        _proposal(), _patch(), apply_callback=apply
    )

    assert result["status"] == "rejected"
    assert result["restored"] is True
    assert source.read_text(encoding="utf-8").startswith("def compose_camera_plan")
    assert not extra.exists()
    assert any("undeclared" in item for item in result["failure_evidence"])


@pytest.mark.parametrize(
    "proposal",
    [
        _proposal(owners=["director_camera", "director_trajectory"], cross_owner_exception=True),
        _proposal(affected_files=["dataset/train.jsonl"]),
        _proposal(affected_files=["src/videoact/observer_contract.py"]),
        _proposal(affected_files=["tests/test_camera.py"]),
    ],
)
def test_executor_rejects_multiple_owners_and_frozen_paths(tmp_path, proposal):
    from videoact.patch_executor import PatchExecutor

    _make_repo(tmp_path)
    result = PatchExecutor(repo_root=tmp_path, output_dir=tmp_path / "audit").execute(
        proposal, _patch()
    )

    assert result["status"] == "blocked"
    assert result["action"] == "blocked"


def test_executor_does_not_expose_dev_or_test_context_to_coding_contract(tmp_path):
    from videoact.patch_executor import PatchExecutor

    _make_repo(tmp_path)
    executor = PatchExecutor(repo_root=tmp_path)
    with pytest.raises(ValueError, match="train evidence"):
        executor.build_coding_context(
            _proposal(),
            [{"case_id": "dev-one", "split": "dev"}],
        )


def test_file_contents_patch_is_compiled_and_accepted(tmp_path):
    from videoact.patch_executor import PatchExecutor

    source = _make_repo(tmp_path)
    result = PatchExecutor(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
        owner_challenge_runner=lambda: True,
        unit_test_runner=lambda: True,
    ).execute(
        _proposal(),
        {
            "file_contents": {"src/videoact/director_camera.py": "def compose_camera_plan():\n    return 'new'\n"},
        },
    )

    assert result["status"] == "accepted"
    assert "return 'new'" in source.read_text(encoding="utf-8")
