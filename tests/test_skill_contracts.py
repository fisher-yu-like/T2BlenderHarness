from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_harness_skills_document_evaluator_v5_boundaries():
    english = (ROOT / "skills" / "t2blendercodeharness" / "SKILL.md").read_text(encoding="utf-8")
    chinese = (ROOT / "skills" / "t2blendercodeharness-zh" / "SKILL.md").read_text(encoding="utf-8")
    director = (ROOT / "skills" / "director-agent" / "SKILL.md").read_text(encoding="utf-8")
    evolution = (ROOT / "skills" / "harness-evolution" / "SKILL.md").read_text(encoding="utf-8")
    training = (ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md").read_text(encoding="utf-8")

    for text in (english, chinese, director, evolution, training):
        assert "frame_statistics" in text
        assert "artifact_health" in text
        assert "predicted_fixes" in text
        assert "predicted_regressions" in text
        assert "function_library" in text
    assert "gpt-5.6-luna" in english
    assert "gpt-5.6-terra" in english
    assert "gpt-5.6-Luna" not in english
    assert "prose_guidance" in evolution
    assert "rollback" in evolution
    assert "gpt-5.6-luna" in training
    assert "gpt-5.6-terra" in training
    assert "canonical report names" not in training


def test_director_skill_turns_real_video_feedback_into_executable_checks():
    text = (ROOT / "skills" / "director-agent" / "SKILL.md").read_text(encoding="utf-8")

    for marker in ("penetration", "occlusion", "continuity_group", "orbit", "handoff"):
        assert marker in text


def test_director_skill_separates_production_plan_from_explicit_baseline_projection():
    text = (ROOT / "skills" / "director-agent" / "SKILL.md").read_text(encoding="utf-8")
    assert "plan_explicit_baseline" in text
    assert "not a production fallback" in text


def test_blender_code_agent_skill_documents_layers_and_fail_closed_boundary():
    path = ROOT / "skills" / "blender-code-agent" / "SKILL.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for marker in ("L2", "L3", "L4", "generate-once-freeze", "fail-closed", "template_baseline"):
        assert marker in text


def test_training_skill_declares_active_six_round_60_60_20_protocol():
    path = ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    assert "trajectory-v5-agent-codegen" in text
    assert "exactly 60 train, 60 dev, and 20 frozen test" in text
    assert "exactly 6" in text
    assert "1320" in text
    assert "template_baseline" in text
    assert "gpt-5.6-luna" in text and "gpt-5.6-terra" in text


def test_training_skill_requires_verbatim_benchmark_prompts():
    text = (ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md").read_text(encoding="utf-8")

    assert "dataset/vbench2-agent-training-index-v1" in text
    assert "validate_benchmark_prompt_index.py" in text
    assert "self-built" in text and "ineligible" in text


def test_training_skill_matches_local_codex_inner_loop_policy():
    text = (ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md").read_text(encoding="utf-8")

    for marker in (
        "codex-local",
        "no external endpoint",
        "max_inner_attempts=3",
        "3960",
        "local Codex visual review",
    ):
        assert marker in text


def test_main_harness_skill_points_to_dynamic_agent_protocol():
    path = ROOT / "skills" / "t2blendercodeharness" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    for marker in (
        "BlenderCodeAgent",
        "CodexLocalProvider",
        "CodexExecProvider",
        "generate-once-freeze",
        "trajectory-v5-agent-codegen",
        "template_baseline",
        "source mutation",
    ):
        assert marker in text
    assert "no external endpoint" in text
    assert "max_inner_attempts=3" in text


def test_real_pipeline_reference_documents_bounded_case_regeneration():
    text = (ROOT / "skills" / "t2blendercodeharness" / "references" / "real-pipeline.md").read_text(encoding="utf-8")
    for marker in ("CodexLocalProvider", "inner loop", "at most three", "no external endpoint"):
        assert marker in text


def test_convergence_plan_records_local_provider_and_inner_retry_policy():
    text = (ROOT / "docs" / "superpowers" / "plans" / "2026-08-28-two-plan-convergence-completion.md").read_text(
        encoding="utf-8"
    )
    for marker in ("codex-local", "at most three", "no external endpoint", "人工视觉校准"):
        assert marker in text


def test_chinese_harness_skill_matches_active_training_contract():
    path = ROOT / "skills" / "t2blendercodeharness-zh" / "SKILL.md"
    text = path.read_text(encoding="utf-8")
    for marker in ("BlenderCodeAgent", "trajectory-v5-agent-codegen", "六轮", "1320", "template_baseline"):
        assert marker in text


def test_architecture_document_covers_real_production_handoff_chain():
    path = ROOT / "docs" / "harness-architecture-v2.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for marker in (
        "DirectorAgent",
        "BlenderCodeAgent",
        "CodexExecProvider",
        "case coverage gate",
        "template_baseline",
        "plan_hash → code_hash → artifact_hash",
        "fail-closed",
    ):
        assert marker in text


def test_canonical_training_memory_document_is_ready_for_append_only_updates():
    path = ROOT / "docs" / "t2blendercodeharness-agent-training-memory-v1.md"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    for marker in (
        "append-only",
        "Round",
        "Prompt",
        "Proxy video address",
        "Director plan score",
        "Task score",
        "Realism score",
        "Harness problem",
        "Harness fix location/method",
        "Natural-language handling",
        "NOT_RENDERED",
    ):
        assert marker in text


def test_real_pipeline_reference_matches_active_agent_outer_loop_boundary():
    path = ROOT / "skills" / "t2blendercodeharness" / "references" / "real-pipeline.md"
    text = path.read_text(encoding="utf-8")
    for marker in ("DirectorAgent", "BlenderCodeAgent", "CodexExecProvider", "fail-closed", "at most 12 workers"):
        assert marker in text
    assert "local repair" not in text
    assert "inner loop" in text.lower()
    assert "max_inner_attempts=3" in text


def test_training_skill_documents_exact_prompt_human_gate_and_bounded_outer_state_machine():
    text = (ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md").read_text(encoding="utf-8")
    for marker in (
        "golden-review-exact-v2",
        "include-split train",
        "include-split dev",
        "prompt_hash",
        "awaiting_harness_patch",
        "run_bounded_outer_attempts",
        "at most five",
    ):
        assert marker in text


def test_training_skill_documents_explicit_precalibration_diagnostic_mode():
    text = (ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md").read_text(encoding="utf-8")
    for marker in (
        "diagnostic-six-rounds",
        "diagnostic_precalibration",
        "visual_scores_permitted=false",
        "formal_training_allowed=false",
        "audit_dynamic_agent_index",
        "all_cases_reuse_one_generated_source",
    ):
        assert marker in text


def test_training_cli_exposes_diagnostic_modes_without_relaxing_formal_modes():
    text = (ROOT / "scripts" / "train_real_harness.py").read_text(encoding="utf-8")
    assert "diagnostic-six-rounds" in text
    assert "require_diagnostic_training_readiness" in text
    assert "require_training_readiness(args.readiness_report)" in text


def test_training_cli_uses_safe_four_worker_default():
    text = (ROOT / "scripts" / "train_real_harness.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--workers", type=int, default=4)' in text


def test_review_and_readiness_defaults_use_latest_exact_prompt_bundle():
    readiness = (ROOT / "scripts" / "check_training_readiness.py").read_text(encoding="utf-8")
    app = (ROOT / "scripts" / "golden_review_app.py").read_text(encoding="utf-8")
    finalizer = (ROOT / "scripts" / "finalize_golden_review.py").read_text(encoding="utf-8")
    validator = (ROOT / "scripts" / "validate_golden_review_set.py").read_text(encoding="utf-8")
    for text in (readiness, app, finalizer, validator):
        assert "dataset/golden-review-exact-v2" in text


def test_active_harness_skills_do_not_direct_reviewers_to_historical_bundle():
    training = (ROOT / "skills" / "t2blendercodeharness-training" / "SKILL.md").read_text(encoding="utf-8")
    harness = (ROOT / "skills" / "t2blendercodeharness" / "SKILL.md").read_text(encoding="utf-8")
    assert "`dataset/golden-review-exact-v2` as the active" in training
    assert "production bundle is `dataset/golden-review-exact-v2`" in harness


def test_active_evaluator_and_chinese_docs_point_to_exact_review_bundle():
    paths = (
        ROOT / "docs" / "evaluator-calibration.md",
        ROOT / "docs" / "evaluator-v5-calibration.md",
        ROOT / "skills" / "t2blendercodeharness-zh" / "SKILL.md",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "dataset/golden-review-exact-v2" in text
