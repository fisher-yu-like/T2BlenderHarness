from videoact.contracts import Finding, TrajectoryRequirement
from videoact.scene_contract import SceneContractBuilder
from videoact.trajectory import TrajectoryPlanner


def test_static_observe_contract_declares_no_character_phase_template():
    contract = SceneContractBuilder().build("Observe a still red cup on a table.")
    plan = TrajectoryPlanner().plan(contract)

    assert contract.trajectory_requirements == []
    from evaluator.deterministic import DeterministicEvaluator

    report = DeterministicEvaluator().evaluate(contract, plan)
    assert not any(f.failure_id.startswith("trajectory_") for f in report.findings)


def test_trajectory_evaluator_consumes_contract_declared_custom_requirement():
    contract = SceneContractBuilder().build("Observe a still red cup on a table.")
    plan = TrajectoryPlanner().plan(contract)
    contract = contract.model_copy(
        update={
            "trajectory_requirements": [
                TrajectoryRequirement(
                    entity_id="red_cup",
                    required_event_ids=["observe"],
                    minimum_states=3,
                )
            ]
        }
    )

    from evaluator.deterministic import DeterministicEvaluator

    report = DeterministicEvaluator().evaluate(contract, plan)
    finding = next(f for f in report.findings if f.failure_id == "trajectory_state_evidence_insufficient")
    assert finding.severity == "error"
    assert finding.root_cause_id == "trajectory_requirement:red_cup"


def test_duplicate_findings_charge_only_the_strongest_root_cause_once():
    from evaluator.findings import deduplicate_findings, score_findings

    findings = [
        Finding(
            failure_id="camera_event_uncovered",
            root_cause_id="camera_required_event_coverage",
            owner="camera_planner",
            category="camera_coverage",
            severity="error",
            message="plan check",
            evidence=["grasp"],
            repair_route="camera_repair",
        ),
        Finding(
            failure_id="missing_required_event",
            root_cause_id="camera_required_event_coverage",
            owner="camera_planner",
            category="camera_coverage",
            severity="hard",
            message="oracle check",
            evidence=["release"],
            repair_route="camera_repair",
        ),
    ]

    unique = deduplicate_findings(findings)
    assert len(unique) == 1
    assert unique[0].severity == "hard"
    assert unique[0].evidence == ["grasp", "release"]
    assert score_findings(findings) == 70
