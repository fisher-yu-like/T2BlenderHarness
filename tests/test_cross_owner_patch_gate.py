from __future__ import annotations

import pytest


def _valid_proposal() -> dict:
    return {
        "owner": "director_camera",
        "owners": ["director_camera", "blender_code_agent"],
        "cross_owner_exception": True,
        "affected_files": [
            "src/videoact/director_camera.py",
            "src/videoact/blender_code_agent.py",
        ],
        "dependency_manifest": {
            "version": "dependency-manifest-v1",
            "owners": ["director_camera", "blender_code_agent"],
            "joint_required": True,
            "required_contract_edges": [
                {
                    "from_owner": "director_camera",
                    "to_owner": "blender_code_agent",
                    "interface": "director_plan_v1",
                    "reason": "the code agent consumes the newly required camera cue contract",
                }
            ],
        },
        "ablation_report": {
            "version": "cross-owner-ablation-v1",
            "a_only": {"train_delta": 0.2, "dev_delta": -0.1, "contract_satisfied": False, "accepted": False},
            "b_only": {"train_delta": 0.3, "dev_delta": -0.2, "contract_satisfied": False, "accepted": False},
            "a_plus_b": {"train_delta": 2.4, "dev_delta": 0.1, "contract_satisfied": True, "accepted": True},
        },
    }


def test_cross_owner_without_dependency_manifest_is_rejected() -> None:
    from videoact.cross_owner import validate_cross_owner_proposal

    proposal = _valid_proposal()
    proposal.pop("dependency_manifest")

    with pytest.raises(ValueError, match="dependency manifest"):
        validate_cross_owner_proposal(proposal)


def test_cross_owner_without_all_ablation_arms_is_rejected() -> None:
    from videoact.cross_owner import validate_cross_owner_proposal

    proposal = _valid_proposal()
    proposal["ablation_report"]["b_only"] = None

    with pytest.raises(ValueError, match="A-only, B-only, and A\+B"):
        validate_cross_owner_proposal(proposal)


def test_valid_cross_owner_report_contains_train_dev_and_interaction_effect() -> None:
    from videoact.cross_owner import validate_cross_owner_proposal

    report = validate_cross_owner_proposal(_valid_proposal())

    assert report["status"] == "pass"
    assert report["train_dev_deltas"]["a_plus_b"] == {"train": 2.4, "dev": 0.1}
    assert report["interaction_effect"]["train"] == pytest.approx(1.9)
    assert report["interaction_effect"]["dev"] == pytest.approx(0.4)


def test_cross_owner_cannot_change_acceptance_or_evaluator_files() -> None:
    from videoact.cross_owner import validate_cross_owner_proposal

    proposal = _valid_proposal()
    proposal["affected_files"].append("src/videoact/outer_loop.py")

    with pytest.raises(ValueError, match="acceptance"):
        validate_cross_owner_proposal(proposal)


def test_single_owner_remains_the_default() -> None:
    from videoact.cross_owner import validate_cross_owner_proposal

    report = validate_cross_owner_proposal(
        {"owner": "director_camera", "affected_files": ["src/videoact/director_camera.py"]}
    )

    assert report["status"] == "pass"
    assert report["cross_owner_exception"] is False
