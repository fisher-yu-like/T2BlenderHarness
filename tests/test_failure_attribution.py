from __future__ import annotations

import pytest


def _failure(**updates):
    payload = {
        "case_id": "train-attribution-01",
        "split": "train",
        "failure_id": "handoff_failure",
        "root_cause_id": "ownership_transition",
        "first_divergence_stage": "runtime_execution",
        "owner_candidate": "blender_executor",
        "owner_confidence": 0.8,
        "severity": "hard",
        "category": "ownership",
        "message": "final owner is wrong",
        "evidence_complete": True,
        "evidence_refs": ["observer_report.json"],
        "actionable": True,
        "abstain": False,
    }
    payload.update(updates)
    return payload


def test_same_source_blender_counterfactual_selects_executor_and_excludes_upstream_owners() -> None:
    from videoact.failure_attribution import CounterfactualAttributor

    result = CounterfactualAttributor(max_runs=4).attribute(
        _failure(),
        counterfactuals=[
            {"family": "same_prompt_director", "owner": "director_prompt_interpreter", "status": "pass"},
            {"family": "same_plan_codegen", "owner": "blender_code_agent", "status": "pass"},
            {"family": "same_source_blender", "owner": "blender_executor", "status": "fail", "evidence_refs": ["run-2"]},
        ],
    )

    assert result.abstain is False
    assert result.owner_candidate == "blender_executor"
    assert result.first_divergence_stage == "runtime_execution"
    assert "director_prompt_interpreter" in result.excluded_owners
    assert "blender_code_agent" in result.excluded_owners
    assert result.counterfactual_count == 3
    assert len(result.parent_hash) == 64
    assert all(len(value) == 64 for value in result.child_hashes)


def test_multiple_indistinguishable_owners_abstain() -> None:
    from videoact.failure_attribution import CounterfactualAttributor

    result = CounterfactualAttributor().attribute(
        _failure(owner_candidate=None, owner_confidence=0.0, actionable=False, abstain=True),
        counterfactuals=[
            {"family": "same_source_blender", "owner": "blender_executor", "status": "fail"},
            {"family": "same_source_blender", "owner": "interaction_library", "status": "fail"},
        ],
    )

    assert result.abstain is True
    assert result.owner_candidate is None
    assert result.reason == "multiple_indistinguishable_owner_candidates"


def test_executor_timeout_is_not_attributed_to_director_or_codegen() -> None:
    from videoact.failure_attribution import CounterfactualAttributor

    result = CounterfactualAttributor().attribute(
        _failure(first_divergence_stage="runtime_execution"),
        counterfactuals=[
            {"family": "same_plan_codegen", "owner": "blender_code_agent", "status": "pass"},
            {"family": "same_source_blender", "owner": "blender_executor", "status": "timeout"},
        ],
    )

    assert result.owner_candidate in {"blender_executor", None}
    assert "director_prompt_interpreter" in result.excluded_owners
    assert "blender_code_agent" in result.excluded_owners


def test_attribution_rejects_non_train_and_enforces_fixed_budget() -> None:
    from videoact.failure_attribution import CounterfactualAttributor

    with pytest.raises(ValueError, match="train-only"):
        CounterfactualAttributor().attribute(_failure(split="dev"))
    with pytest.raises(ValueError, match="budget"):
        CounterfactualAttributor(max_runs=1).attribute(
            _failure(),
            counterfactuals=[
                {"family": "same_prompt_director", "owner": "director_prompt_interpreter", "status": "pass"},
                {"family": "same_source_blender", "owner": "blender_executor", "status": "fail"},
            ],
        )


def test_counterfactual_runner_hashes_parent_and_children_deterministically() -> None:
    from videoact.failure_attribution import CounterfactualRunner

    runner = CounterfactualRunner(max_runs=2)
    first = runner.run(
        case_id="train-runner-01",
        prompt="a prompt",
        split="train",
        variants=[{"family": "same_video_judge", "output": {"score": 10}}],
    )
    second = runner.run(
        case_id="train-runner-01",
        prompt="a prompt",
        split="train",
        variants=[{"family": "same_video_judge", "output": {"score": 10}}],
    )

    assert first.parent_hash == second.parent_hash
    assert first.child_hashes == second.child_hashes
    assert first.split == "train"


def test_attribution_can_be_serialized_by_the_cli_runner(tmp_path) -> None:
    import json

    from scripts.run_counterfactual_attribution import run

    input_path = tmp_path / "attribution.json"
    output_path = tmp_path / "out.json"
    input_path.write_text(
        json.dumps(
            {
                "failure": _failure(),
                "counterfactuals": [
                    {"family": "same_source_blender", "owner": "blender_executor", "status": "fail"}
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run(input_path, output_path)

    assert result[0]["owner_candidate"] == "blender_executor"
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["schema_version"] == "failure-attribution-v1"
