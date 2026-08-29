from __future__ import annotations


def test_formal_config_freezes_paired_statistics_thresholds() -> None:
    from evaluator.formal_config import FormalEvaluatorConfig

    config = FormalEvaluatorConfig.from_mapping(
        {
            "generator_model_id": "codex-cli",
            "primary_judge_model_id": "gpt-5.6-luna",
            "audit_judge_model_id": "gpt-5.6-terra",
            "paired_statistics_version": "paired-statistics-v1",
            "bootstrap_seed": 20260829,
            "bootstrap_iterations": 2000,
            "alpha": 0.05,
            "train_min_gain": 1.0,
            "dev_noninferiority_margin": -1.0,
            "secondary_noninferiority_margin": -1.0,
        }
    )

    assert config.as_dict()["paired_statistics_version"] == "paired-statistics-v1"
    assert config.bootstrap_seed == 20260829
    assert config.bootstrap_iterations == 2000
    assert config.train_min_gain == 1.0
    assert config.dev_noninferiority_margin == -1.0
    assert config.confidence_threshold == 0.6
    assert config.evidence_completeness_threshold == 1.0


def test_formal_config_freezes_separate_director_and_codegen_identities() -> None:
    from evaluator.formal_config import FormalEvaluatorConfig

    config = FormalEvaluatorConfig.from_mapping(
        {
            "generator_model_id": "external_openai_compatible:gpt-5.6-luna|codex_exec_local:codex-cli",
            "director_model_id": "gpt-5.6-luna",
            "codegen_model_id": "codex-cli",
            "director_provider_kind": "external_openai_compatible",
            "codegen_provider_kind": "codex_exec_local",
            "primary_judge_model_id": "gpt-5.6-luna",
            "audit_judge_model_id": "gpt-5.6-terra",
        }
    )

    assert config.director_model_id == "gpt-5.6-luna"
    assert config.codegen_model_id == "codex-cli"
    assert config.director_provider_kind == "external_openai_compatible"
    assert config.codegen_provider_kind == "codex_exec_local"
    assert config.as_dict()["generator_model_id"].startswith("external_openai_compatible:")


def test_formal_config_rejects_invalid_statistics_thresholds() -> None:
    from evaluator.formal_config import FormalEvaluatorConfig

    try:
        FormalEvaluatorConfig.from_mapping(
            {
                "generator_model_id": "codex-cli",
                "primary_judge_model_id": "gpt-5.6-luna",
                "audit_judge_model_id": "gpt-5.6-terra",
                "alpha": 1.0,
            }
        )
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:  # pragma: no cover - documents the fail-closed contract
        raise AssertionError("invalid alpha unexpectedly accepted")


def test_formal_config_rejects_invalid_evidence_threshold() -> None:
    from evaluator.formal_config import FormalEvaluatorConfig

    try:
        FormalEvaluatorConfig.from_mapping(
            {
                "generator_model_id": "codex-cli",
                "primary_judge_model_id": "gpt-5.6-luna",
                "audit_judge_model_id": "gpt-5.6-terra",
                "confidence_threshold": 1.5,
            }
        )
    except ValueError as exc:
        assert "confidence_threshold" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid confidence threshold unexpectedly accepted")
