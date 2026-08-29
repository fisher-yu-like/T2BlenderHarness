"""Run the real Blender/VLM Harness protocols without synthetic scores.

The round roots produced by this protocol are immutable job
batches.  This runner renders those jobs with Blender CLI, evaluates the real
artifacts, calls the configured VLM on sampled frames, and writes one unified
report per batch.  It can also prepare and evaluate the complete train split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_real_runs import evaluate_real_run, evaluate_real_split  # noqa: E402
from scripts.evaluate_real_videos import evaluate_split  # noqa: E402
from scripts.prepare_real_jobs import prepare_jobs  # noqa: E402
from scripts.render_evaluate_groups import group_case_ids  # noqa: E402
from scripts.render_proxy_jobs_parallel import render_jobs  # noqa: E402
from scripts.validate_benchmark_prompt_index import validate_benchmark_prompt_index  # noqa: E402
from evaluator.openai_vlm import canonical_vlm_name  # noqa: E402
from evaluator.visual_primary import VISUAL_PRIMARY_VERSION  # noqa: E402
from videoact.blender_code_agent import BlenderCodeAgent  # noqa: E402
from videoact.codex_exec_provider import CodexExecProvider  # noqa: E402
from videoact.director import DirectorAgent  # noqa: E402
from videoact.provider_provenance import provider_identity  # noqa: E402
from videoact.real_inner_loop import run_real_inner_loop  # noqa: E402
from videoact.source_fingerprints import audit_source_reuse  # noqa: E402
from videoact.paired_statistics import evaluate_paired_acceptance  # noqa: E402
from videoact.cross_owner import validate_cross_owner_proposal  # noqa: E402
from videoact.release_gates import validate_formal_release_report  # noqa: E402

_MISSING = object()


def require_benchmark_training_dataset(dataset_root: str | Path) -> dict[str, Any]:
    """Fail closed unless the active training index is verbatim benchmark data."""

    report = validate_benchmark_prompt_index(dataset_root)
    if report.get("status") != "pass":
        raise ValueError(
            "training requires a validated benchmark prompt index; "
            f"self-built or mutated datasets are ineligible: {json.dumps(report, ensure_ascii=False, sort_keys=True)}"
        )
    return report


def require_training_readiness(report_path: str | Path) -> dict[str, Any]:
    """Prevent real training until the independent readiness matrix passes."""

    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"training readiness report is missing or unreadable: {path}: {exc}") from exc
    if not isinstance(report, dict) or report.get("training_allowed") is not True:
        raise ValueError(
            "real Harness training is blocked until training_allowed=true: "
            f"status={report.get('status') if isinstance(report, dict) else None}, "
            f"blocking_gates={report.get('blocking_gates') if isinstance(report, dict) else None}, "
            f"report={path.resolve()}"
        )
    return report


def require_formal_training_release(
    readiness_report_path: str | Path,
    release_report_path: str | Path,
) -> dict[str, Any]:
    """Require both the prerequisite matrix and sealed G0--G3 release."""

    readiness = require_training_readiness(readiness_report_path)
    path = Path(release_report_path)
    try:
        release = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"formal release report is missing or unreadable: {path}: {exc}") from exc
    verification = validate_formal_release_report(release)
    if verification.get("training_allowed") is not True:
        raise ValueError(
            "formal release is blocked until sealed G0-G3 reports pass: "
            f"verification={json.dumps(verification, ensure_ascii=False, sort_keys=True)}, "
            f"report={path.resolve()}"
        )
    return {
        "status": "pass",
        "training_allowed": True,
        "readiness": readiness,
        "release": release,
        "release_verification": verification,
    }


def require_model_provider_mode(provider_mode: str) -> str:
    """Reject the rule/template baseline for formal Harness evolution."""

    normalized = str(provider_mode or "").strip().lower()
    if normalized not in {"model", "glm"}:
        raise ValueError(
            "formal Harness training requires --provider-mode model or --provider-mode glm; "
            "rule_template_baseline is diagnostic-only and cannot produce a training claim"
        )
    return normalized


DIAGNOSTIC_DEFERRED_GATES = frozenset({"golden_review", "paired_gate"})
DIAGNOSTIC_REQUIRED_GATES = frozenset(
    {
        "full_test",
        "capability",
        "dataset",
        "frozen_eval",
        "real_blender_smoke",
        "dynamic_agent_provider",
        "golden_review",
        "paired_gate",
    }
)


def require_diagnostic_training_readiness(report_path: str | Path) -> dict[str, Any]:
    """Allow a transparent pre-calibration run without unlocking formal training.

    The production gate remains fail-closed until human visual calibration and
    the paired gate pass.  This separate entry point is intentionally limited
    to a readiness report whose only pending gates are those two human-facing
    gates; it never authorizes a numeric visual score or a formal patch
    acceptance decision.
    """

    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"diagnostic training readiness is missing or unreadable: {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError("diagnostic training is blocked: readiness report must be an object")
    if report.get("numeric_substitutions"):
        raise ValueError("diagnostic training is blocked: numeric gate substitutions are present")
    gates = report.get("gates")
    if not isinstance(gates, dict):
        raise ValueError("diagnostic training is blocked: readiness gates are missing")
    missing = sorted(DIAGNOSTIC_REQUIRED_GATES - set(gates))
    if missing:
        raise ValueError(f"diagnostic training is blocked: readiness gates are missing: {missing}")
    deferred: list[str] = []
    invalid: list[str] = []
    for name, gate in gates.items():
        status = gate.get("status") if isinstance(gate, dict) else None
        if status == "pass":
            continue
        if name in DIAGNOSTIC_DEFERRED_GATES and status == "pending":
            deferred.append(name)
            continue
        invalid.append(f"{name}={status}")
    if invalid:
        raise ValueError(
            "diagnostic training is blocked: only pending golden_review/paired_gate may be deferred; "
            + ", ".join(invalid)
        )
    return {
        "mode": "diagnostic_precalibration",
        "formal_training_allowed": report.get("training_allowed") is True and not deferred,
        "visual_scores_permitted": False,
        "deferred_gates": sorted(deferred),
        "readiness_report": str(path.resolve()),
        "readiness_status": report.get("status"),
    }


def audit_dynamic_agent_index(
    index: dict[str, Any], *, run_root: str | Path, expected_case_ids: list[str]
) -> dict[str, Any]:
    """Prove a prepared batch is the dynamic agent arm, not a template arm.

    This is a provenance gate, not a visual quality score.  It rejects an
    explicit template mode, missing per-case codegen provenance, missing
    sources, and a batch in which every distinct case reuses one source.
    """

    expected = [str(case_id) for case_id in expected_case_ids]
    jobs = index.get("jobs") if isinstance(index, dict) else None
    if index.get("generation_mode") != "agent":
        return {
            "status": "fail",
            "reason": "template_generation_mode_is_not_allowed_for_agent_training",
            "generation_mode": index.get("generation_mode"),
        }
    if not isinstance(jobs, list):
        return {"status": "fail", "reason": "agent_job_index_missing_jobs", "generation_mode": "agent"}
    by_id = {str(job.get("case_id")): job for job in jobs if isinstance(job, dict)}
    missing = sorted(set(expected) - set(by_id))
    if missing:
        return {"status": "fail", "reason": "agent_job_index_missing_cases", "missing_case_ids": missing}
    failures: list[str] = []
    provider_mode = str(index.get("provider_mode") or "").strip()
    expected_kind_by_mode = {
        "glm": "zhipu_glm_openai_compatible",
        "model": "external_openai_compatible",
        "assistant": "agent_session_structured",
    }
    require_manifest = provider_mode in {"model", "glm", "assistant"}
    expected_provider_kind = expected_kind_by_mode.get(provider_mode, "external_openai_compatible")
    source_hashes: dict[str, str] = {}
    source_texts: dict[str, str] = {}
    root = Path(run_root)
    for case_id in expected:
        job = by_id[case_id]
        if job.get("status") != "prepared":
            failures.append(f"{case_id}:job_status={job.get('status')}")
        if not str(job.get("codegen_call_id") or "").strip():
            failures.append(f"{case_id}:missing_codegen_call_id")
        if require_manifest:
            manifest_path = Path(str(job.get("provider_manifest_path") or root / case_id / "provider_manifest.json"))
            if not manifest_path.is_absolute():
                manifest_path = root / manifest_path
            if not manifest_path.is_file():
                failures.append(f"{case_id}:missing_provider_manifest")
            else:
                try:
                    provider_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    failures.append(f"{case_id}:provider_manifest_unreadable:{type(exc).__name__}")
                else:
                    if provider_manifest.get("provider_mode") != provider_mode:
                        failures.append(f"{case_id}:provider_manifest_mode_mismatch")
                    if provider_manifest.get("template_backed") is not False:
                        failures.append(f"{case_id}:provider_manifest_template_backed")
                    if provider_manifest.get("llm_generated") is not True:
                        failures.append(f"{case_id}:provider_manifest_not_llm_generated")
                    stages = provider_manifest.get("stages")
                    if not isinstance(stages, dict):
                        failures.append(f"{case_id}:provider_manifest_missing_stages")
                    else:
                        stage_call_ids: dict[str, str] = {}
                        for stage in ("director", "blender_code"):
                            call = stages.get(stage)
                            if not isinstance(call, dict) or not str(call.get("call_id") or "").strip():
                                failures.append(f"{case_id}:missing_{stage}_provider_call_id")
                            elif call.get("template_backed") is not False or call.get("llm_generated") is not True:
                                failures.append(f"{case_id}:{stage}_provider_identity_invalid")
                            else:
                                stage_call_ids[stage] = str(call["call_id"])
                                for field in (
                                    "provider_kind",
                                    "model_id",
                                    "model_version",
                                    "request_schema_hash",
                                    "response_schema_hash",
                                    "prompt_hash",
                                    "request_hash",
                                    "response_hash",
                                    "started_at",
                                    "ended_at",
                                ):
                                    if not str(call.get(field) or "").strip():
                                        failures.append(f"{case_id}:{stage}_missing_{field}")
                                if provider_mode in {"glm", "assistant"} and call.get("provider_kind") != expected_provider_kind:
                                    failures.append(f"{case_id}:{stage}_provider_is_not_{provider_mode}")
                                elif provider_mode == "assistant" and call.get("model_id") != "glm-5.3-flash":
                                    failures.append(f"{case_id}:{stage}_assistant_model_id_invalid")
                                elif provider_mode == "model":
                                    if stage == "director" and call.get("provider_kind") != "external_openai_compatible":
                                        failures.append(f"{case_id}:director_provider_is_not_external_structured")
                                    if stage == "blender_code" and call.get("provider_kind") != "codex_exec_local":
                                        failures.append(f"{case_id}:blender_code_provider_is_not_local_codex")
                        if len(stage_call_ids) == 2 and len(set(stage_call_ids.values())) != 2:
                            failures.append(f"{case_id}:director_and_codegen_call_ids_collide")
                    if provider_manifest.get("status") != "complete":
                        failures.append(f"{case_id}:provider_manifest_not_complete")
        job_path = Path(str(job.get("job_path") or root / case_id / "blender_job.py"))
        if not job_path.is_absolute():
            candidates = (root / job_path, Path.cwd() / job_path)
            job_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not job_path.is_file() or job_path.stat().st_size == 0:
            failures.append(f"{case_id}:missing_generated_source")
            continue
        source_text = job_path.read_text(encoding="utf-8", errors="replace")
        source_texts[case_id] = source_text
        # A unique hash alone is not sufficient: a shared scaffold can be
        # copied and stamped with a case id.  The diagnostic rule-template
        # arm carries its explicit profile marker; the formal model arm must
        # instead bind the source to a DirectorPlan and its hash.
        if provider_mode in {"model", "glm", "assistant"}:
            if "DIRECTOR_PLAN" not in source_text:
                failures.append(f"{case_id}:missing_director_plan_binding")
        elif (
            "CASE_SCENE_PROFILE" not in source_text
            or "codex-local-case-profile-v2" not in source_text
        ):
            failures.append(f"{case_id}:missing_case_specific_generation_profile")
        director_plan_hash = str(job.get("director_plan_hash") or "").strip()
        if director_plan_hash and director_plan_hash[:16] not in source_text:
            failures.append(f"{case_id}:case_profile_not_bound_to_director_plan")
        actual_source_hash = hashlib.sha256(job_path.read_bytes()).hexdigest()
        source_hash = str(job.get("code_hash") or "").strip()
        if not source_hash:
            source_hash = actual_source_hash
        elif source_hash != actual_source_hash:
            failures.append(f"{case_id}:generated_source_hash_mismatch")
        source_hashes[case_id] = source_hash
    unique_hashes = sorted(set(source_hashes.values()))
    if len(expected) > 1 and len(unique_hashes) == 1 and len(source_hashes) == len(expected):
        failures.append("all_cases_reuse_one_generated_source")
    source_reuse = audit_source_reuse(source_texts)
    if provider_mode in {"model", "glm"} and source_reuse.get("probable_template_reuse"):
        failures.append("probable_template_reuse")
    return {
        "status": "pass" if not failures else "fail",
        "reason": "dynamic_case_specific_source_verified" if not failures else "agent_source_provenance_failed",
        "generation_mode": "agent",
        "provider_mode": provider_mode or None,
        "case_count": len(expected),
        "prepared_count": sum(by_id[case_id].get("status") == "prepared" for case_id in expected),
        "unique_source_count": len(unique_hashes),
        "source_hashes": source_hashes,
        "source_reuse_audit": source_reuse,
        "failures": failures,
    }


def _is_missing(value: Any) -> bool:
    return value is _MISSING or value is None or (isinstance(value, str) and not value.strip())


def _first_non_missing(*values: Any) -> Any:
    return next((value for value in values if not _is_missing(value)), _MISSING)


def _read_failure_reason(path: Any, status: Any) -> str:
    """Read a preparation failure without hiding malformed evidence."""

    fallback = str(status or "preparation_failed")
    if not path:
        return fallback
    failure_path = Path(str(path))
    if not failure_path.is_file():
        return f"{fallback}: failure record missing"
    try:
        payload = json.loads(failure_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"{fallback}: failure record unreadable ({type(exc).__name__})"
    if isinstance(payload, dict):
        return str(payload.get("reason") or payload.get("status") or fallback)
    return fallback


def _numbered_path_part(parts: tuple[str, ...], prefix: str) -> Any:
    part = next((part for part in parts if part.startswith(prefix)), None)
    if part is None:
        return _MISSING
    suffix = part.removeprefix(prefix)
    return int(suffix) if suffix.isdigit() else part


def _group_case_ids(case_ids: list[str], *, expected_groups: int = 6) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for case_id in case_ids:
        family = case_id.rsplit("-", 1)[0]
        groups.setdefault(family, []).append(case_id)
    ordered = []
    for family in sorted(groups, key=lambda value: int(value.rsplit("-", 1)[-1])):
        values = sorted(groups[family])
        if len(values) != 10:
            raise ValueError(f"each round family must contain 10 cases: {family} has {len(values)}")
        ordered.append(values)
    if len(ordered) != expected_groups:
        raise ValueError(f"expected {expected_groups} round families, got {len(ordered)}")
    return ordered


def build_six_round_batches(train_ids: list[str], dev_ids: list[str]) -> list[dict[str, list[str]]]:
    """Pair six disjoint 10-case train families with six disjoint dev families."""
    if len(train_ids) != 60 or len(dev_ids) != 60:
        raise ValueError("six-round protocol requires exactly 60 train and 60 dev case IDs")
    if set(train_ids) & set(dev_ids):
        raise ValueError("train and dev case IDs must be disjoint")
    return build_round_batches(train_ids, dev_ids)


def build_round_batches(train_ids: list[str], dev_ids: list[str]) -> list[dict[str, list[str]]]:
    """Pair balanced 10-case train/dev families for the active phase."""
    if len(train_ids) != len(dev_ids) or not train_ids or len(train_ids) % 10:
        raise ValueError("train and dev must have equal non-zero counts divisible by 10")
    expected_groups = len(train_ids) // 10
    if set(train_ids) & set(dev_ids):
        raise ValueError("train and dev case IDs must be disjoint")
    train_groups = _group_case_ids(train_ids, expected_groups=expected_groups)
    dev_groups = _group_case_ids(dev_ids, expected_groups=expected_groups)
    return [
        {"round": index + 1, "train": train_group, "dev": dev_groups[index]}
        for index, train_group in enumerate(train_groups)
    ]


def build_protocol_manifest(
    train_ids: list[str], dev_ids: list[str], test_ids: list[str], *, dataset_fingerprint: str
) -> dict[str, Any]:
    batches = build_round_batches(train_ids, dev_ids)
    if len(test_ids) != 20 or set(test_ids) & (set(train_ids) | set(dev_ids)):
        raise ValueError("test split must contain 20 frozen, disjoint cases")
    rounds: list[dict[str, Any]] = []
    for batch in batches:
        rounds.append(
            {
                "round": batch["round"],
                "train": batch["train"],
                "dev": batch["dev"],
                "overall_evaluation": {
                    # The user-facing protocol requires a complete 60-train
                    # plus 60-dev evaluation at the end of every round.  The
                    # ten-case batch above is the patch-selection signal; the
                    # overall run is not a progressively shrinking sample.
                    "scope": "cumulative_train_and_dev",
                    "train_cases": list(train_ids),
                    "dev_cases": list(dev_ids),
                    "blind_test": False,
                },
            }
        )
    return {
        "protocol_version": "real-round-v4-shared-review",
        "dataset_fingerprint": dataset_fingerprint,
        "round_count": len(batches),
        "train_count": len(train_ids),
        "dev_count": len(dev_ids),
        "test_count": len(test_ids),
        "attempts_per_round_max": 5,
        "attempt_policy": build_attempt_policy(overall_case_count=len(train_ids) + len(dev_ids)),
        "batch_case_count": 20,
        "overall_case_count": len(train_ids) + len(dev_ids),
        "videos_per_round_max": 5 * 20 + len(train_ids) + len(dev_ids),
        "videos_total_max": len(batches) * (5 * 20 + len(train_ids) + len(dev_ids)),
        "selection_policy": f"{len(batches)} disjoint train families paired with {len(batches)} disjoint dev families",
        "patch_scope": "Harness-only under src/videoact; Blender, dataset, evaluator, and generated plans are immutable",
        "rounds": rounds,
        "final_evaluation": {
            "scope": "all_train_and_all_dev",
            "train_cases": train_ids,
            "dev_cases": dev_ids,
            "blind_test_cases": test_ids,
        },
    }


def build_multi_five_round_manifest(
    train_ids: list[str],
    dev_ids: list[str],
    test_ids: list[str],
    *,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    """Build the trajectory-v4-multi five-round outer-loop protocol.

    Each attempt pairs ten train cases with ten dev cases.  The round-end
    overall evaluation uses all train cases seen so far and all sixty dev
    cases, so the held-out dev set cannot be silently narrowed to the paired
    batch.
    """
    if len(train_ids) != 50 or len(dev_ids) != 60 or len(test_ids) != 30:
        raise ValueError("multi-five-round protocol requires 50 train, 60 dev, and 30 test cases")
    if set(train_ids) & set(dev_ids) or set(train_ids) & set(test_ids) or set(dev_ids) & set(test_ids):
        raise ValueError("multi-five-round splits must be disjoint")
    def fixed_groups(case_ids: list[str], group_count: int) -> list[list[str]]:
        ordered = sorted(case_ids)
        if len(ordered) % group_count:
            raise ValueError("multi-five-round IDs cannot be split into ten-case groups")
        size = len(ordered) // group_count
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]

    train_groups = fixed_groups(train_ids, 5)
    dev_groups = fixed_groups(dev_ids, 6)
    rounds: list[dict[str, Any]] = []
    cumulative_train: list[str] = []
    for index, train_group in enumerate(train_groups):
        paired_dev = dev_groups[index]
        cumulative_train.extend(train_group)
        rounds.append(
            {
                "round": index + 1,
                "train": train_group,
                "dev": paired_dev,
                "overall_evaluation": {
                    "scope": "cumulative_train_and_all_dev",
                    "train_cases": list(cumulative_train),
                    "dev_cases": list(dev_ids),
                    "blind_test": False,
                },
            }
        )
    overall_counts = [
        len(item["overall_evaluation"]["train_cases"])
        + len(item["overall_evaluation"]["dev_cases"])
        for item in rounds
    ]
    attempt_policy = build_attempt_policy(overall_case_count=110)
    attempt_policy["overall_case_counts_by_round"] = overall_counts
    attempt_policy["videos_total_max"] = 5 * 100 + sum(overall_counts)
    return {
        "protocol_version": "multi-five-rounds-v1",
        "dataset_fingerprint": dataset_fingerprint,
        "round_count": 5,
        "train_count": 50,
        "dev_count": 60,
        "test_count": 30,
        "attempts_per_round_max": 5,
        "attempt_policy": attempt_policy,
        "batch_case_count": 20,
        "overall_case_counts_by_round": overall_counts,
        "videos_per_attempt": 20,
        "videos_per_round_max": 100 + max(overall_counts),
        "videos_total_max": 5 * 100 + sum(overall_counts),
        "selection_policy": "five disjoint ten-case train families paired with five ten-case dev families; sixth dev family is overall-only",
        "patch_scope": "one Harness owner per accepted patch; dataset/evaluator/Blender are frozen",
        "rounds": rounds,
        "final_evaluation": {
            "scope": "all_train_and_all_dev_then_blind_test",
            "train_cases": list(train_ids),
            "dev_cases": list(dev_ids),
            "blind_test_cases": list(test_ids),
        },
    }


def build_active_protocol_manifest(
    train_ids: list[str], dev_ids: list[str], test_ids: list[str], *, dataset_fingerprint: str, dataset_id: str | None = None
) -> dict[str, Any]:
    if dataset_id == "trajectory-v4-multi" or len(train_ids) == 50 and len(dev_ids) == 60 and len(test_ids) == 30:
        return build_multi_five_round_manifest(train_ids, dev_ids, test_ids, dataset_fingerprint=dataset_fingerprint)
    return build_protocol_manifest(train_ids, dev_ids, test_ids, dataset_fingerprint=dataset_fingerprint)


def build_attempt_policy(max_attempts: int = 5, overall_case_count: int | None = None) -> dict[str, Any]:
    if max_attempts != 5:
        raise ValueError("this protocol fixes max_attempts at 5")
    # Keep the historical no-argument helper contract for old callers/tests;
    # the active phase passes its 100-case cumulative count explicitly.
    legacy_default = overall_case_count is None
    overall_case_count = 120 if legacy_default else overall_case_count
    return {
        "mode": "outer_loop_with_bounded_case_regeneration",
        "max_attempts": 5,
        "inner_case_attempts_max": 3,
        "render_retries_per_case": 0,
        "videos_per_attempt": 20,
        "overall_videos_per_round": overall_case_count,
        "videos_per_round_max": 5 * 20 + overall_case_count,
        # ``videos_*`` count protocol case slots.  Candidate counts include
        # the worst-case three local plan/code/render generations per slot.
        "candidate_videos_per_case_max": 3,
        "candidate_videos_per_attempt_max": 3 * 20,
        "candidate_videos_per_round_max": 3 * (5 * 20 + overall_case_count),
        # Six rounds are the active protocol.  The historical multi-five
        # adapter overwrites this field with its own exact accounting.
        "videos_total_max": 6 * (5 * 20 + overall_case_count),
        "candidate_videos_total_max": 6 * 3 * (5 * 20 + overall_case_count),
    }


def validate_harness_patch_paths(paths: list[str]) -> None:
    """Reject changes outside Harness source; never patch Blender/data/evaluator outputs."""
    for raw_path in paths:
        path = raw_path.replace("\\", "/").lstrip("./")
        if not path.startswith("src/videoact/"):
            raise ValueError(
                "Harness-only patch scope violation: allowed files are under src/videoact/; "
                f"rejected {raw_path}"
            )
        if path.endswith(("trajectory.json", "camera_plan.json", "scene_contract.json")):
            raise ValueError(
                "Harness-only patch scope violation: generated plan/contract contents are immutable; "
                f"rejected {raw_path}"
            )


def run_bounded_outer_attempts(
    *,
    run_attempt: Callable[[int], dict[str, Any]],
    transition: Callable[[int, list[dict[str, Any]]], dict[str, Any]],
    max_attempts: int = 5,
) -> dict[str, Any]:
    """Run a round's outer-loop attempts through an explicit state machine.

    The callback that returns ``{"action": "patch"}`` is the boundary where
    the Coding Agent has already applied and recorded exactly one Harness-owner
    patch.  This function never edits source itself.  With no patch transition
    available, the round stops after its first evidence-producing attempt with
    ``awaiting_harness_patch`` instead of silently rendering the same Harness
    again.  A transition can therefore be driven by the Codex Host between
    attempts while remaining bounded at five.
    """

    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts must be an integer between 1 and 5")
    reports: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    status = "max_attempts_exhausted"
    reason = "maximum outer-loop attempts reached"

    for attempt_number in range(1, max_attempts + 1):
        report = run_attempt(attempt_number)
        if not isinstance(report, dict):
            raise ValueError("run_attempt must return a JSON object")
        reports.append(report)
        decision = transition(attempt_number, list(reports))
        if not isinstance(decision, dict):
            raise ValueError("outer-loop transition must return a JSON object")
        action = decision.get("action")
        if action not in {"patch", "stop", "accept"}:
            raise ValueError("outer-loop transition action must be patch, stop, or accept")
        if action == "patch":
            proposal = decision.get("proposal")
            if not isinstance(proposal, dict) or not str(proposal.get("owner") or "").strip():
                raise ValueError("a patch transition requires a proposal with one owner")
            cross_owner_audit = validate_cross_owner_proposal(proposal)
            decision["cross_owner_audit"] = cross_owner_audit
            raw_files = proposal.get("affected_files", proposal.get("files", []))
            if raw_files is not None:
                if not isinstance(raw_files, list) or any(not isinstance(item, str) for item in raw_files):
                    raise ValueError("patch proposal affected_files must be a list of paths")
                validate_harness_patch_paths(raw_files)
            if attempt_number == max_attempts:
                transitions.append(dict(decision))
                status = "max_attempts_exhausted"
                reason = "patch proposed at the fifth attempt; no sixth attempt is permitted"
                break
            transitions.append(dict(decision))
            continue
        transitions.append(dict(decision))
        status = str(decision.get("status") or ("accepted" if action == "accept" else "stopped"))
        reason = str(decision.get("reason") or status)
        break

    return {
        "status": status,
        "reason": reason,
        "attempt_count": len(reports),
        "max_attempts": max_attempts,
        "reports": reports,
        "transitions": transitions,
    }


def build_dynamic_codex_agents(
    *,
    codex_command: str = "codex",
    timeout_s: int = 1800,
    provider_mode: str = "rule_template_baseline",
    director_base_url: str | None = None,
    director_api_key: str | None = None,
    director_model: str | None = None,
    glm_base_url: str | None = None,
    glm_api_key: str | None = None,
    glm_model: str | None = None,
):
    """Build a named provider arm with an auditable identity.

    ``rule_template_baseline`` is retained for diagnostics and regression
    comparisons only.  Formal training may select ``glm``; that arm calls two
    separate Zhipu GLM structured providers, one for Director interpretation
    and one for Blender codegen.  The two stages are intentionally separate
    provider instances and are recorded independently.  ``model`` remains a
    compatibility arm for the historical external-Director/local-Codex pair.
    Legacy names remain accepted as compatibility aliases but are normalized
    to the explicit baseline arm and never presented as an LLM call.
    """

    if provider_mode in {"rule_template_baseline", "local", "codex-local"}:
        from videoact.codex_self_provider import build_codex_local_agents

        return build_codex_local_agents()
    if provider_mode == "codex-self":
        from videoact.codex_self_provider import build_codex_self_agents

        return build_codex_self_agents()
    if provider_mode not in {"model", "external", "glm", "assistant"}:
        raise ValueError(
            "provider_mode must be rule_template_baseline, model, external, glm, or assistant"
        )

    if provider_mode == "assistant":
        # The driving coding-agent session is the LLM for both stages: each
        # structured call is materialized as a request file under
        # ASSISTANT_SESSION_ROOT and answered by an authored response file.
        # No external endpoint and no template fallback; provenance, schema
        # validation, and the static source gate are identical to the glm arm.
        from videoact.assistant_session_provider import AssistantSessionProvider

        session_root = os.getenv("ASSISTANT_SESSION_ROOT") or "out/assistant-session"
        wait_timeout_s = float(os.getenv("ASSISTANT_WAIT_TIMEOUT_S") or max(7200.0, timeout_s))
        director_provider = AssistantSessionProvider.for_director(
            session_root=session_root,
            wait_timeout_s=wait_timeout_s,
        )
        code_provider = AssistantSessionProvider.for_codegen(
            session_root=session_root,
            wait_timeout_s=wait_timeout_s,
        )
        director = DirectorAgent.from_provider(
            director_provider,
            provider_name="assistant-session-glm-flash",
            policy="director-v5-glm-structured",
        )
        return director, BlenderCodeAgent(
            provider=code_provider,
            model=code_provider.model_id,
            max_codegen_attempts=2,
        )

    if provider_mode == "glm":
        from videoact.external_structured_provider import GLMStructuredProvider

        director_provider = GLMStructuredProvider.for_director(
            base_url=glm_base_url,
            api_key=glm_api_key,
            model=glm_model,
            timeout_s=timeout_s,
        )
        code_provider = GLMStructuredProvider.for_codegen(
            base_url=glm_base_url,
            api_key=glm_api_key,
            model=glm_model,
            timeout_s=timeout_s,
        )
        director = DirectorAgent.from_provider(
            director_provider,
            provider_name="external-glm",
            policy="director-v5-glm-structured",
        )
        return director, BlenderCodeAgent(
            provider=code_provider,
            model=code_provider.model_id,
            # The first GLM code sample has repeatedly exhibited a bounded
            # syntax/serialization corruption.  Permit one repair prompt
            # carrying static-gate evidence; any second failure remains hard.
            max_codegen_attempts=2,
        )

    from videoact.external_structured_provider import OpenAICompatibleStructuredProvider

    director_provider = OpenAICompatibleStructuredProvider.for_director(
        base_url=director_base_url,
        api_key=director_api_key,
        model=director_model,
        timeout_s=timeout_s,
    )
    compatibility_name = "external-director"
    compatibility_policy = "director-v5-external-structured"
    director = DirectorAgent.from_provider(
        director_provider,
        provider_name=compatibility_name,
        policy=compatibility_policy,
    )
    code_provider = CodexExecProvider.for_codegen(command=codex_command, timeout_s=timeout_s)
    code_agent = BlenderCodeAgent(
        provider=code_provider,
        model="local-codex-exec",
    )
    return director, code_agent


def anti_overfit_gate(
    train_before: float,
    train_after: float,
    paired_dev_before: float,
    paired_dev_after: float,
    overall_dev_before: float,
    overall_dev_after: float,
    *,
    paired_train_deltas: list[float] | None = None,
    paired_dev_deltas: list[float] | None = None,
    safety_before: dict[str, Any] | None = None,
    safety_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the T10 paired gate, retaining the scalar compatibility API.

    Historical callers pass only aggregate means and continue to receive the
    original strict scalar checks.  A real paired run passes case deltas and
    safety metrics; its decision then comes from the reproducible bootstrap
    report instead of an all-cases-positive rule.
    """

    paired = None
    if paired_train_deltas is not None or paired_dev_deltas is not None:
        paired = evaluate_paired_acceptance(
            paired_train_deltas or [float(train_after) - float(train_before)],
            paired_dev_deltas or [float(paired_dev_after) - float(paired_dev_before)],
            safety_before=safety_before,
            safety_after=safety_after,
            require_safety_metrics=safety_before is not None or safety_after is not None,
        )
    checks = {
        "train_strict_gain": (
            paired["checks"]["train_min_gain"] if paired is not None else train_after > train_before
        ),
        "paired_dev_non_regression": (
            paired["checks"]["dev_noninferiority"] if paired is not None else paired_dev_after >= paired_dev_before
        ),
        "overall_dev_non_regression": overall_dev_after >= overall_dev_before,
    }
    if paired is not None:
        checks["paired_statistics"] = paired["accepted"]
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "reason": "accepted" if all(checks.values()) else "rejected_anti_overfit_gate",
        "paired_statistics": paired,
    }


def write_training_memory_markdown(destination: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write one UTF-8 Markdown table containing the complete Harness training memory."""

    def cell(value: Any) -> str:
        if _is_missing(value):
            value = "unavailable"
        return str(value).replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("|", "\\|")

    def first_available(row: dict[str, Any], *keys: str) -> Any:
        return _first_non_missing(*(row.get(key, _MISSING) for key in keys))

    lines = [
        "<!-- Canonical columns: Round | Attempt | Split | Case ID | Prompt | Proxy video address | Director plan score | Task score | Realism score | Review source/confidence | Harness problem | Harness fix location/method | Before→after delta | Natural-language handling -->",
        "# T2Blendercodeharness Training Memory v1",
        "",
        "Status: diagnostic_precalibration_pending",
        "",
        "This append-only memory records real Blender evidence, exact prompts, and every Harness decision.",
        "The active input is the verbatim VBench index; missing visual review is unavailable, never zero.",
        "A failed preparation or render is recorded as NOT_RENDERED: <reason> and is never omitted.",
        "Formal training requires training_allowed=true. Diagnostic rows are evidence only and keep formal acceptance blocked.",
        "Post-diagnostic semantic audit: object-only prompts no longer receive an invented actor; compound nouns and slide motion are preserved; the two-case real smoke is recorded under out/preflight/director-object-only-v4.",
        "",
        "# T2Blendercodeharness 训练记忆表",
        "",
        "每一行保留真实 proxy 视频、独立评分通道、Harness 问题、修复和自然语言处理结论。",
        "",
        "| 轮数 | Attempt | Split | Case ID | Prompt | Proxy 视频地址 | Director plan 分 | Task score | Realism score | Review | 检测出的 Harness 问题 | Owner | 修复位置/方法 | 提升或下降 | 自然语言处理 |",
        "|---:|---:|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|",
    ]
    for row in rows:
        fix_location = first_available(row, "fix_location")
        fix_method = first_available(row, "fix_method")
        if fix_location is _MISSING:
            fix_summary = fix_method
        elif fix_method is _MISSING:
            fix_summary = fix_location
        else:
            fix_summary = f"{fix_location}: {fix_method}"

        review = first_available(row, "review")
        if review is _MISSING:
            review_source = first_available(row, "review_source")
            review_confidence = first_available(row, "review_confidence")
            if review_source is not _MISSING and review_confidence is not _MISSING:
                review = f"{review_source} confidence={review_confidence}"
            elif review_source is not _MISSING:
                review = review_source

        values = (
            first_available(row, "round"),
            first_available(row, "attempt"),
            first_available(row, "split"),
            first_available(row, "case_id"),
            first_available(row, "prompt"),
            first_available(row, "proxy_video"),
            first_available(row, "director_plan_score"),
            first_available(row, "task_score", "video_score", "score"),
            first_available(row, "realism_score"),
            review,
            first_available(row, "detected_problem"),
            first_available(row, "owner"),
            fix_summary,
            first_available(row, "delta"),
            first_available(row, "handling"),
        )
        lines.append("| " + " | ".join(cell(value) for value in values) + " |")

    channel_rows = [
        row
        for row in rows
        if any(
            not _is_missing(row.get(name, _MISSING))
            for name in ("visual_score", "physical_score", "trajectory_score", "camera_score")
        )
    ]
    if channel_rows:
        lines.extend(
            [
                "",
                "## 真实视频多维评分通道（来自 MP4 解码 + Blender 逐帧观测）",
                "",
                "这些分数不从 plan 直接读取：MP4 必须成功解码，且 telemetry 必须包含实际逐帧变换、屏幕框和世界包围盒。",
                "",
                "| 轮数 | Attempt | Split | Case ID | 视觉分 | 物理分 | 轨迹分 | 摄像机分 | 视频证据来源 |",
                "|---:|---:|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in channel_rows:
            values = (
                row.get("round", _MISSING),
                row.get("attempt", _MISSING),
                row.get("split", _MISSING),
                row.get("case_id", _MISSING),
                row.get("visual_score", _MISSING),
                row.get("physical_score", _MISSING),
                row.get("trajectory_score", _MISSING),
                row.get("camera_score", _MISSING),
                row.get("video_evidence_source", _MISSING),
            )
            lines.append("| " + " | ".join(cell(value) for value in values) + " |")

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preparation_failure_results(run_root: str | Path) -> list[dict[str, Any]]:
    """Convert preparation failures into explicit, non-scored case records."""

    root = Path(run_root)
    index_path = root / "job_index.json"
    if not index_path.is_file():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return []
    failures: list[dict[str, Any]] = []
    for job in index.get("jobs", []) or []:
        if not isinstance(job, dict) or job.get("status") == "prepared":
            continue
        case_id = str(job.get("case_id") or "unknown-case")
        failure_status = str(job.get("status") or "preparation_failed")
        failure_path = Path(str(job.get("failure_path"))) if job.get("failure_path") else None
        reason = failure_status
        failure_ids: list[str] = [failure_status]
        if failure_path and failure_path.is_file():
            try:
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
                reason = str(payload.get("reason") or payload.get("status") or reason)
                response = payload.get("codegen_response") or {}
                for uncertainty in response.get("uncertainties", []) or []:
                    identifier = uncertainty.get("id") if isinstance(uncertainty, dict) else None
                    if identifier:
                        failure_ids.append(str(identifier))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                reason = f"{failure_status}: failure record unreadable"
        run_dir = Path(str(job.get("run_dir") or root / case_id))
        failures.append(
            {
                "case_id": case_id,
                "status": "not_rendered",
                "preparation_status": failure_status,
                "preparation_failure": reason,
                "artifact_status": "incomplete",
                "deterministic_status": "not_run",
                "deterministic_score": None,
                "deterministic_findings": list(dict.fromkeys(failure_ids)),
                "director_findings": [],
                "interaction_findings": [],
                "vlm_status": "unavailable",
                "vlm_reason": "not_rendered",
                "proxy_video": str((run_dir / "proxy.mp4").resolve()),
                "video_exists": False,
            }
        )
    return failures


def merge_real_scores(
    *, run_root: str | Path, deterministic_results: list[dict[str, Any]], vlm_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Merge shared evidence while keeping task and realism scores separate."""
    deterministic_by_id = {item["case_id"]: item for item in deterministic_results}
    vlm_by_id = {item["case_id"]: item for item in vlm_results}
    case_ids = sorted(set(deterministic_by_id) | set(vlm_by_id))
    cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        real = deterministic_by_id.get(case_id, {})
        vlm = vlm_by_id.get(case_id, {})
        aggregate = vlm.get("aggregate") or {}
        realism = (vlm.get("realism") if vlm.get("status") == "scored" else None) or real.get("realism") or {}
        local_evidence = vlm.get("local_video_evidence") or vlm.get("deterministic_video_proxy_metrics") or {}
        channels = local_evidence.get("channels") or {}
        video_path = Path(run_root) / case_id / "proxy.mp4"
        video_score = (
            vlm.get("task_score")
            if vlm.get("status") == "scored" and vlm.get("task_score") is not None
            else aggregate.get("final_score")
            if vlm.get("status") == "scored"
            else None
        )
        cases.append(
            {
                "case_id": case_id,
                "proxy_video": str(video_path.resolve()),
                "video_exists": video_path.is_file() and video_path.stat().st_size > 0,
                "artifact_status": real.get("artifact_status"),
                "deterministic_status": real.get("status"),
                "deterministic_score": real.get("score"),
                "preparation_status": real.get("preparation_status"),
                "preparation_failure": real.get("preparation_failure"),
                "deterministic_findings": real.get("findings", []),
                "deterministic_finding_details": real.get("finding_details", []),
                "director_plan_score": real.get("director_plan_score"),
                "director_findings": real.get("director_findings", []),
                "interaction_findings": real.get("interaction_findings", []),
                "vlm_status": vlm.get("status"),
                "review_source": vlm.get("review_source"),
                "review_confidence": (
                    vlm.get("review_confidence")
                    or vlm.get("confidence")
                    or (vlm.get("vlm_response") or {}).get("confidence")
                ),
                "vlm_score": aggregate.get("vlm_score"),
                "overall_vlm_score": vlm.get("overall_vlm_score"),
                "video_score": video_score,
                "vlm_reason": vlm.get("reason"),
                "task_final_score": video_score,
                "realism_score": realism.get("score"),
                "realism_score_kind": realism.get("score_kind"),
                "realism_band": realism.get("band"),
                "realism_claim": realism.get("realism_claim"),
                "realism_requires_independent_review": realism.get("requires_independent_review"),
                "realism_evaluator_version": realism.get("evaluator_version"),
                "visual_score": channels.get("visual_score"),
                "physical_score": channels.get("physical_score"),
                "trajectory_score": channels.get("trajectory_score"),
                "camera_score": channels.get("camera_score"),
                "video_evidence_source": local_evidence.get("source"),
                "video_evidence_evaluator_version": local_evidence.get("evaluator_version"),
                "render_retry_count": _render_retry_count(Path(run_root) / case_id),
            }
        )
    scored = [float(item["video_score"]) for item in cases if item["video_score"] is not None]
    review_sources = {
        item.get("review_source")
        for item in cases
        if item.get("video_score") is not None and item.get("review_source")
    }
    if review_sources == {"assistant_local_review"}:
        scoring_mode = "real_blender_video_assistant_local_review"
    elif review_sources == {"codex_local_visual_review"}:
        scoring_mode = "real_blender_video_codex_local_visual_review"
    elif review_sources and review_sources != {"external_vlm"}:
        scoring_mode = "real_blender_video_mixed_review"
    else:
        scoring_mode = "real_blender_video_vlm"
    deterministic_scores = [
        float(item["deterministic_score"])
        for item in cases
        if item["deterministic_score"] is not None
    ]
    realism_scores = [
        float(item["realism_score"])
        for item in cases
        if item.get("realism_score") is not None and item.get("realism_score_kind")
    ]
    channel_names = ("visual_score", "physical_score", "trajectory_score", "camera_score")
    channel_means = {
        f"mean_{name}": round(
            sum(float(item[name]) for item in cases if item.get(name) is not None)
            / len([item for item in cases if item.get(name) is not None]),
            4,
        )
        if any(item.get(name) is not None for item in cases)
        else None
        for name in channel_names
    }
    failure_counts: dict[str, int] = {}
    for item in cases:
        for failure_id in item["deterministic_findings"]:
            failure_counts[failure_id] = failure_counts.get(failure_id, 0) + 1
    return {
        "scoring_mode": scoring_mode,
        "run_root": str(Path(run_root).resolve()),
        "case_count": len(cases),
        "real_video_count": sum(item["video_exists"] for item in cases),
        "vlm_scored_count": len(scored),
        "preparation_failed_count": sum(item.get("preparation_status") is not None for item in cases),
        "artifact_failed_count": sum(item.get("artifact_status") not in {None, "complete"} for item in cases),
        "aggregate": {
            "mean_task_final_score": round(sum(scored) / len(scored), 4) if scored else None,
            "mean_final_score": round(sum(scored) / len(scored), 4) if scored else None,
            "mean_deterministic_score": round(sum(deterministic_scores) / len(deterministic_scores), 4)
            if deterministic_scores
            else None,
            "mean_artifact_only_realism_score": round(sum(realism_scores) / len(realism_scores), 4)
            if realism_scores
            else None,
            "realism_scored_count": len(realism_scores),
            **channel_means,
            "failure_counts": dict(sorted(failure_counts.items())),
        },
        "score_channels": {
            "task_final_score": "VLM-compatible task channel; local path is computed from decoded MP4 plus runtime evidence",
            "artifact_only_realism_score": "geometry/PNG evidence; not added to task score",
            "visual_score": "decoded MP4 clarity/detail/presentation channel",
            "physical_score": "actual runtime bounding-box, collision, ground, rig, and smoothness channel",
            "trajectory_score": "actual runtime entity trajectory, timing, and smoothness channel",
            "camera_score": "actual runtime event coverage and camera-motion cue channel",
            "combined": False,
        },
        "cases": cases,
    }


def _render_retry_count(run_dir: Path) -> int:
    path = run_dir / "render_attempts.json"
    if not path.is_file():
        return 0
    try:
        attempts = json.loads(path.read_text(encoding="utf-8"))
        return max(0, len(attempts) - 1) if isinstance(attempts, list) else 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def _dataset_records(dataset_root: str | Path) -> dict[str, dict[str, Any]]:
    return {
        record["case_id"]: record
        for record in (
            json.loads(line)
            for line in (Path(dataset_root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def write_unified_outputs(
    report: dict[str, Any], *, dataset_root: str | Path, report_root: str | Path, markdown_path: str | Path | None = None
) -> None:
    root = Path(report_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "real_unified_score.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    records = _dataset_records(dataset_root)
    lines = [
        "# Real Blender Video Evaluation",
        "",
        f"- scoring mode: `{report['scoring_mode']}`",
        f"- case count: `{report['case_count']}`",
        f"- real videos: `{report['real_video_count']}`",
        f"- video-scored cases: `{report['vlm_scored_count']}`",
        f"- mean final score: `{report['aggregate']['mean_final_score']}`",
        f"- mean task final score (separate channel): `{report['aggregate'].get('mean_task_final_score')}`",
        f"- mean artifact-only realism score (separate channel): `{report['aggregate'].get('mean_artifact_only_realism_score')}`",
        "",
        "分数只来自真实 Blender 生成的 `proxy.mp4` 的采样帧，并标注 external_vlm 或 assistant_local_review 来源；artifact 不完整、VLM unavailable 或本地复核未完成的 case 不进入 mean final score。",
        "",
        "| Case | Prompt | Proxy video | Deterministic | Video review | Task final score | Artifact-only realism | Artifact | Review status | Findings |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for item in report["cases"]:
        prompt = records.get(item["case_id"], {}).get("prompt", "").replace("|", "\\|")
        video = Path(item["proxy_video"])
        video_link = f"[{video.name}]({video.as_posix()})" if item["video_exists"] else f"NOT_RENDERED: `{video}`"
        findings = ", ".join(item["deterministic_findings"]) or "none"
        lines.append(
            "| {case_id} | {prompt} | {video} | {deterministic} | {vlm} | {final} | {realism} | {artifact} | {vlm_status} | {findings} |".format(
                case_id=item["case_id"],
                prompt=prompt,
                video=video_link,
                deterministic=item["deterministic_score"],
                vlm=item["vlm_score"],
                final=item["video_score"],
                realism=item["realism_score"],
                artifact=item["artifact_status"],
                vlm_status=f"{item['vlm_status']} ({item.get('review_source') or 'none'})",
                findings=findings,
            )
        )
    destination = Path(markdown_path) if markdown_path else root / "real_unified_score.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name == "t2blendercodeharness-agent-training-memory-v1.md":
        # The canonical training document is append-only evidence, not the
        # per-run summary format above.  Route it through the cross-root
        # aggregator so a smoke/attempt cannot erase historical rows.
        update_training_memory_table(root, dataset_root, destination)
        return
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inner_not_rendered_result(
    *,
    case_id: str,
    run_root: Path,
    split: str,
    inner_case: dict[str, Any],
) -> dict[str, Any]:
    """Convert an exhausted inner-loop case into an explicit audit row."""

    attempts = list(inner_case.get("attempts", []))
    last = attempts[-1] if attempts else {}
    reason = str(last.get("reason") or inner_case.get("reason") or "max_inner_attempts_exhausted")
    return {
        "case_id": case_id,
        "split": split,
        "status": "not_rendered",
        "preparation_status": last.get("status", "inner_loop_exhausted"),
        "preparation_failure": reason,
        "artifact_status": "incomplete",
        "deterministic_status": "not_run",
        "deterministic_score": None,
        "deterministic_findings": ["inner_loop_exhausted"],
        "director_findings": [],
        "interaction_findings": [],
        "vlm_status": "unavailable",
        "vlm_reason": "not_rendered",
        "proxy_video": str((run_root / case_id / "proxy.mp4").resolve()),
        "video_exists": False,
        "inner_loop_attempts": attempts,
    }


def run_real_batch_with_inner_loop(
    run_root: str | Path,
    *,
    split: str,
    case_ids: list[str],
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path | None = None,
    scoring_policy: str = VISUAL_PRIMARY_VERSION,
    director_agent: DirectorAgent,
    code_agent: BlenderCodeAgent,
    provider_mode: str = "injected",
    code_cache_dir: str | Path | None = None,
    codegen_examples_root: str | Path | None = None,
    max_inner_attempts: int = 3,
    assistant_review_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare, render, and evaluate a real batch with bounded regeneration.

    ``vlm_model`` is retained as the requested visual-judge model label. In
    the formal hybrid generation path, the external structured Director
    provider and the local Codex Blender-code provider are separate calls.
    The visual-review diagnostic may use the in-process Codex reviewer without
    an external VLM endpoint; incomplete evidence stays unavailable rather
    than becoming a zero or a plan-derived number.
    """

    if not case_ids:
        raise ValueError("case_ids must not be empty")
    if not 1 <= workers <= 12:
        raise ValueError("workers must be between 1 and 12")
    root = Path(run_root).resolve()
    records = _dataset_records(dataset_root)
    missing = sorted(set(case_ids) - set(records))
    if missing:
        raise ValueError(f"case IDs are missing from dataset manifest: {missing}")
    render_reports: list[dict[str, Any]] = []
    provenance_reports: list[dict[str, Any]] = []

    def prepare_callback(pending_ids: list[str], attempt: int) -> dict[str, Any]:
        # Each inner candidate gets an isolated cache namespace.  This forces
        # both local Codex stages to regenerate instead of silently reusing a
        # failed candidate's source.
        attempt_cache = Path(code_cache_dir or (root / "code_cache")) / f"inner-attempt-{attempt:02d}"
        index = prepare_jobs(
            split,
            root,
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            case_ids=pending_ids,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode=provider_mode,
            code_cache_dir=attempt_cache,
            codegen_examples_root=codegen_examples_root,
        )
        provenance = audit_dynamic_agent_index(index, run_root=root, expected_case_ids=pending_ids)
        provenance["inner_attempt"] = attempt
        provenance_reports.append(provenance)
        jobs = list(index.get("jobs", []))
        prepared_ids = [str(job["case_id"]) for job in jobs if job.get("status") == "prepared"]
        failures = {
            str(job["case_id"]): {
                "status": job.get("status"),
                "reason": _read_failure_reason(job.get("failure_path"), job.get("status")),
                "failure_path": job.get("failure_path"),
            }
            for job in jobs
            if job.get("status") != "prepared"
        }
        if provenance["status"] != "pass":
            reason = "; ".join(provenance.get("failures", [])) or provenance.get("reason", "agent_source_provenance_failed")
            prepared_ids = []
            for case_id in pending_ids:
                failures[case_id] = {
                    "status": "agent_provenance_failed",
                    "reason": reason,
                    "provenance": provenance,
                }
        return {"prepared_ids": prepared_ids, "failures": failures, "job_index": index}

    def render_callback(pending_ids: list[str], attempt: int) -> dict[str, Any]:
        combined: dict[str, dict[str, Any]] = {}
        groups: list[dict[str, Any]] = []
        for group in group_case_ids(pending_ids, group_size=12):
            report = render_jobs(
                root,
                blender_bin=blender_bin,
                workers=workers,
                timeout_s=timeout_s,
                # A failed candidate is regenerated by the outer inner loop;
                # do not spend additional hidden attempts on the same source.
                max_retries=0,
                case_ids=group,
            )
            groups.append({"case_ids": group, "report": report})
            for item in report.get("results", []):
                combined[str(item["case_id"])] = item
        payload = {"results": combined, "groups": groups, "attempt": attempt}
        render_reports.append(payload)
        return payload

    dataset_fingerprint = None
    try:
        dataset_fingerprint = (
            json.loads((Path(dataset_root) / "metadata.json").read_text(encoding="utf-8")).get("fingerprint")
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        dataset_fingerprint = None

    def evaluate_callback(case_id: str, attempt: int) -> dict[str, Any]:
        result = evaluate_real_run(
            root / case_id,
            record=records[case_id],
            blender_bin=blender_bin,
            dataset_fingerprint=dataset_fingerprint,
        )
        result["proxy_video"] = str((root / case_id / "proxy.mp4").resolve())
        result["inner_attempt"] = attempt
        return result

    inner = run_real_inner_loop(
        case_ids,
        root,
        prepare=prepare_callback,
        render=render_callback,
        evaluate=evaluate_callback,
        max_attempts=max_inner_attempts,
    )
    deterministic_results: list[dict[str, Any]] = []
    for case_id in case_ids:
        inner_case = inner["cases"][case_id]
        if inner_case.get("evaluation"):
            deterministic_results.append(dict(inner_case["evaluation"]))
        else:
            deterministic_results.append(
                _inner_not_rendered_result(
                    case_id=case_id,
                    run_root=root,
                    split=split,
                    inner_case=inner_case,
                )
            )
    preliminary = merge_real_scores(
        run_root=root,
        deterministic_results=deterministic_results,
        vlm_results=[],
    )
    preliminary["status"] = "awaiting_local_visual_review"
    preliminary["inner_loop"] = inner
    preliminary["render"] = {
        "status": "completed",
        "group_size": 12,
        "workers": workers,
        "max_render_retries_per_candidate": 0,
        "groups": render_reports,
    }
    preliminary["agent_provenance"] = provenance_reports
    preliminary["vlm_model"] = str(vlm_model).lower()
    preliminary["vlm_call_policy"] = "local_codex_visual_review_only; no external endpoint"
    write_unified_outputs(
        preliminary,
        dataset_root=dataset_root,
        report_root=root,
        markdown_path=markdown_path,
    )
    visual_results = evaluate_split(
        root,
        dataset_root=dataset_root,
        assistant_local=assistant_review_dir is None,
        assistant_review_dir=assistant_review_dir,
        scoring_policy=scoring_policy,
    )
    report = merge_real_scores(
        run_root=root,
        deterministic_results=deterministic_results,
        vlm_results=visual_results,
    )
    report.update(
        {
            "status": "complete" if report["vlm_scored_count"] == report["real_video_count"] and not inner["pending_case_ids"] else "incomplete_local_visual_review",
            "render": preliminary["render"],
            "agent_provenance": provenance_reports,
            "inner_loop": inner,
            "vlm_model": str(vlm_model).lower(),
            "vlm_call_policy": "local_codex_visual_review_only; no external endpoint",
            "evaluator_version": scoring_policy,
        }
    )
    (root / "real_unified_score.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_unified_outputs(
        report,
        dataset_root=dataset_root,
        report_root=root,
        markdown_path=markdown_path,
    )
    return report


def run_real_batch(
    run_root: str | Path,
    *,
    dataset_root: str | Path,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path | None = None,
    scoring_policy: str = VISUAL_PRIMARY_VERSION,
) -> dict[str, Any]:
    os.environ["OPENAI_VLM_MODEL"] = vlm_model
    render_report = render_jobs(run_root, blender_bin=blender_bin, workers=workers, timeout_s=timeout_s)
    deterministic_results = evaluate_real_split(run_root, dataset_root=dataset_root, blender_bin=blender_bin)
    deterministic_results.extend(preparation_failure_results(run_root))
    preliminary = merge_real_scores(run_root=run_root, deterministic_results=deterministic_results, vlm_results=[])
    preliminary["status"] = "awaiting_shared_vlm_review"
    write_unified_outputs(
        preliminary,
        dataset_root=dataset_root,
        report_root=run_root,
        markdown_path=markdown_path,
    )
    vlm_results = evaluate_split(run_root, dataset_root=dataset_root, scoring_policy=scoring_policy)
    report = merge_real_scores(
        run_root=run_root,
        deterministic_results=deterministic_results,
        vlm_results=vlm_results,
    )
    report["render"] = render_report
    report["vlm_model"] = canonical_vlm_name(vlm_model)
    report["evaluator_version"] = scoring_policy
    report["vlm_call_policy"] = "one VLM call per eligible case; geometry/PNG realism is local and separate"
    write_unified_outputs(report, dataset_root=dataset_root, report_root=run_root, markdown_path=markdown_path)
    if (
        report["vlm_scored_count"] != report["real_video_count"]
        or report.get("preparation_failed_count", 0) > 0
        or report.get("artifact_failed_count", 0) > 0
    ):
        report["status"] = "incomplete_vlm_scoring"
    else:
        report["status"] = "complete"
    (Path(run_root) / "real_unified_score.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def prepare_full_split(
    output_root: str | Path,
    *,
    split: str,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    provider_mode: str = "injected",
) -> Path:
    split_root = Path(output_root) / split
    prepare_jobs(
        split,
        split_root,
        dataset_root=dataset_root,
        harness_version=harness_version,
        evaluator_version=evaluator_version,
        provider_mode=provider_mode,
        director_agent=director_agent,
        code_agent=code_agent,
    )
    return split_root


def prepare_full_train(
    output_root: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    provider_mode: str = "injected",
) -> Path:
    return prepare_full_split(
        output_root,
        split="train",
        dataset_root=dataset_root,
        harness_version=harness_version,
        evaluator_version=evaluator_version,
        director_agent=director_agent,
        code_agent=code_agent,
        provider_mode=provider_mode,
    )


def summarize_real_reports(reports: list[dict[str, Any]], *, scope: str) -> dict[str, Any]:
    cases = [case for report in reports for case in report.get("cases", [])]
    scores = [float(case["video_score"]) for case in cases if case.get("video_score") is not None]
    realism_scores = [
        float(case["realism_score"])
        for case in cases
        if case.get("realism_score") is not None
    ]
    return {
        "scope": scope,
        "case_count": len(cases),
        "real_video_count": sum(bool(case.get("video_exists")) for case in cases),
        "vlm_scored_count": len(scores),
        "mean_final_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_task_final_score": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_artifact_only_realism_score": round(sum(realism_scores) / len(realism_scores), 4)
        if realism_scores
        else None,
        "realism_scored_count": len(realism_scores),
        "cases": cases,
    }


def _load_patch_metadata(round_root: Path) -> dict[str, Any]:
    path = round_root / "patch_manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_harness_memory_jsonl(destination: str | Path, round_reports: list[dict[str, Any]]) -> None:
    """Persist round evidence for the next Harness self-evolution decision."""
    events: list[dict[str, Any]] = []
    for report in round_reports:
        round_number = report["round"]
        patch = report.get("patch", {})
        base = {
            "round": round_number,
            "parent_version": patch.get("parent_version", "not_recorded"),
            "candidate_version": patch.get("candidate_version", "not_recorded"),
            "owner": patch.get("owner", "not_recorded"),
            "files": patch.get("files", []),
        }
        events.append({**base, "event": "proposal", "status": "recorded"})
        for split in ("train", "dev"):
            aggregate = report["splits"].get(split, {}).get("aggregate", {})
            events.append(
                {
                    **base,
                    "event": f"{split}_evaluated",
                    "mean_final_score": aggregate.get("mean_final_score"),
                    "real_video_count": report["splits"].get(split, {}).get("real_video_count"),
                    "vlm_scored_count": report["splits"].get(split, {}).get("vlm_scored_count"),
                }
            )
        events.append(
            {
                **base,
                "event": "overall_evaluated",
                "train_mean_final_score": report["overall_evaluation"]["train"].get("mean_final_score"),
                "dev_mean_final_score": report["overall_evaluation"]["dev"].get("mean_final_score"),
            }
        )
        events.append({**base, "event": "decision", "decision": patch.get("decision", "pending_patch_manifest")})
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def _protocol_round(dataset_root: str | Path, round_number: int) -> dict[str, Any]:
    dataset = Path(dataset_root)
    split_payload = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_active_protocol_manifest(
        split_payload["train"], split_payload["dev"], split_payload["test"], dataset_fingerprint=metadata["fingerprint"], dataset_id=metadata.get("dataset_id")
    )
    if not 1 <= round_number <= protocol["round_count"]:
        raise ValueError(f"round must be between 1 and {protocol['round_count']}")
    return protocol["rounds"][round_number - 1]


def _memory_rows_from_reports(output_root: str | Path, dataset_root: str | Path) -> list[dict[str, Any]]:
    records = _dataset_records(dataset_root)
    output = Path(output_root)
    rows: list[dict[str, Any]] = []
    for report_path in sorted(output.rglob("real_unified_score.json")):
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        relative_parts = report_path.relative_to(output).parts
        round_number = _first_non_missing(report.get("round", _MISSING), _numbered_path_part(relative_parts, "round-"))
        attempt_number = _first_non_missing(
            report.get("attempt", _MISSING), _numbered_path_part(relative_parts, "attempt-")
        )
        path_split = report_path.parent.name if report_path.parent.name in {"train", "dev", "test"} else _MISSING
        round_index = next(
            (index for index, part in enumerate(relative_parts) if part.startswith("round-")),
            None,
        )
        patch_root = output.joinpath(*relative_parts[: round_index + 1]) if round_index is not None else report_path.parent
        # Attempt reports have their own patch decision.  Prefer it over the
        # round-level accepted patch so baseline rows cannot inherit the
        # candidate's fix metadata in the long-term memory table.
        attempt_index = next(
            (index for index, part in enumerate(relative_parts) if part.startswith("attempt-")),
            None,
        )
        if attempt_index is not None:
            attempt_root = output.joinpath(*relative_parts[: attempt_index + 1])
            if (attempt_root / "patch_manifest.json").is_file():
                patch_root = attempt_root
        patch = _load_patch_metadata(patch_root)
        for case in report.get("cases", []):
            record = records.get(case["case_id"], {})
            director_plan_hash = None
            manifest_path = report_path.parent / case["case_id"] / "run_manifest.json"
            if manifest_path.is_file():
                try:
                    director_plan_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("director_plan_hash")
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    director_plan_hash = None
            handling = patch.get("handling", _MISSING)
            if director_plan_hash:
                hash_note = f"DirectorPlan hash={director_plan_hash}"
                handling = hash_note if _is_missing(handling) else f"{hash_note}; {handling}"
            task_score = _first_non_missing(
                case.get("task_final_score", _MISSING),
                case.get("video_score", _MISSING),
                case.get("score", _MISSING),
            )
            detected_problem = _first_non_missing(
                ", ".join(case.get("deterministic_findings", [])),
                patch.get("detected_problem", _MISSING),
            )
            if _is_missing(detected_problem) and case.get("realism_score_kind") == "artifact_only_proxy" and case.get("realism_band") == "artifact_only_weak":
                detected_problem = (
                    f"artifact-only realism is weak ({case.get('realism_score')}); "
                    "independent visual review is pending, so no semantic or physical conclusion is claimed"
                )

            rows.append(
                {
                    "round": round_number,
                    "attempt": attempt_number,
                    "split": _first_non_missing(
                        case.get("split", _MISSING), report.get("split", _MISSING), path_split
                    ),
                    "case_id": case["case_id"],
                    "prompt": record.get("prompt", _MISSING),
                    "proxy_video": case.get("proxy_video", _MISSING),
                    "director_plan_score": _first_non_missing(
                        case.get("director_plan_score", _MISSING),
                        record.get("director_plan_score", _MISSING),
                        report.get("director_plan_score", _MISSING),
                    ),
                    "task_score": task_score,
                    "realism_score": case.get("realism_score", _MISSING),
                    "visual_score": case.get("visual_score", _MISSING),
                    "physical_score": case.get("physical_score", _MISSING),
                    "trajectory_score": case.get("trajectory_score", _MISSING),
                    "camera_score": case.get("camera_score", _MISSING),
                    "video_evidence_source": case.get("video_evidence_source", _MISSING),
                    "review": case.get("review", _MISSING),
                    "review_source": case.get("review_source", _MISSING),
                    "review_confidence": case.get("review_confidence", _MISSING),
                    "detected_problem": detected_problem,
                    "owner": _first_non_missing(case.get("owner", _MISSING), patch.get("owner", _MISSING)),
                    "fix_location": patch.get("fix_location", _MISSING),
                    "fix_method": patch.get("fix_method", _MISSING),
                    "delta": patch.get("delta", _MISSING),
                    "handling": handling,
                }
            )
    return rows


def _memory_output_roots(output_root: str | Path) -> list[Path]:
    """Return the current diagnostic root plus sibling diagnostic roots.

    Diagnostic runs were intentionally split into a baseline root and later
    upgraded-Harness roots.  Rebuilding the Markdown from only the latest root
    silently erased earlier evidence, so the canonical table must discover all
    sibling diagnostic roots under the same ``out/training`` directory.
    """

    root = Path(output_root).resolve()
    roots = {root}
    parent = root.parent
    if parent.is_dir():
        for sibling in parent.iterdir():
            if sibling.is_dir() and sibling.name.startswith("diagnostic-"):
                roots.add(sibling.resolve())
    return sorted(roots, key=lambda path: str(path).lower())


def update_training_memory_table(output_root: str | Path, dataset_root: str | Path, destination: str | Path) -> None:
    """Rebuild the canonical table without dropping earlier diagnostic roots."""

    rows: list[dict[str, Any]] = []
    for root in _memory_output_roots(output_root):
        rows.extend(_memory_rows_from_reports(root, dataset_root))

    # A report can be revisited after its patch manifest is written.  Replace
    # that exact evidence row once, while retaining baseline/upgrade rows whose
    # proxy paths differ.  This makes each update idempotent and append-only at
    # the experiment-evidence level.
    deduplicated: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = tuple(
            str(row.get(field, ""))
            for field in ("round", "attempt", "split", "case_id", "proxy_video")
        )
        deduplicated[key] = row
    ordered = sorted(
        deduplicated.values(),
        key=lambda row: (
            str(row.get("round", "")),
            str(row.get("attempt", "")),
            str(row.get("split", "")),
            str(row.get("case_id", "")),
            str(row.get("proxy_video", "")),
        ),
    )
    write_training_memory_markdown(destination, ordered)


def run_outer_attempt(
    output_root: str | Path,
    *,
    round_number: int,
    attempt_number: int,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path,
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    provider_mode: str = "rule_template_baseline",
    codex_command: str = "codex",
    code_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not 1 <= attempt_number <= 5:
        raise ValueError("attempt must be between 1 and 5; this is an outer-loop attempt, not an inner repair retry")
    batch = _protocol_round(dataset_root, round_number)
    if director_agent is None or code_agent is None:
        director_agent, code_agent = build_dynamic_codex_agents(
            codex_command=codex_command,
            timeout_s=timeout_s,
            provider_mode=provider_mode,
        )
    attempt_root = Path(output_root) / f"round-{round_number:02d}" / f"attempt-{attempt_number:02d}" / "real"
    reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_root = attempt_root / split
        reports[split] = run_real_batch_with_inner_loop(
            split_root,
            split=split,
            case_ids=batch[split],
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
            markdown_path=markdown_path,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode=provider_mode,
            code_cache_dir=code_cache_dir
            or (
                Path(output_root) / "code_cache" / f"outer-{attempt_number:02d}"
            ),
            # Bounded execution recovery: at most two fresh candidates per
            # case (initial attempt + one regeneration).
            max_inner_attempts=2,
        )
    result = {"round": round_number, "attempt": attempt_number, "batch": batch, "splits": reports}
    attempt_root.parent.parent.mkdir(parents=True, exist_ok=True)
    (attempt_root.parent.parent / "attempt_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_training_memory_table(output_root, dataset_root, markdown_path)
    return result


def run_outer_overall(
    output_root: str | Path,
    *,
    round_number: int,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path,
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    provider_mode: str = "rule_template_baseline",
    codex_command: str = "codex",
    code_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    overall_root = Path(output_root) / f"round-{round_number:02d}" / "overall" / "real"
    batch = _protocol_round(dataset_root, round_number)
    if director_agent is None or code_agent is None:
        director_agent, code_agent = build_dynamic_codex_agents(
            codex_command=codex_command,
            timeout_s=timeout_s,
            provider_mode=provider_mode,
        )
    reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_root = overall_root / split
        reports[split] = run_real_batch_with_inner_loop(
            split_root,
            split=split,
            case_ids=batch["overall_evaluation"][f"{split}_cases"],
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
            markdown_path=markdown_path,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode=provider_mode,
            code_cache_dir=code_cache_dir
            or (
                Path(output_root) / "code_cache" / f"outer-overall-{round_number:02d}"
            ),
            # Bounded execution recovery: at most two fresh candidates per
            # case (initial attempt + one regeneration).
            max_inner_attempts=2,
        )
    result = {
        "round": round_number,
        "scope": "cumulative_train_and_dev",
        "batch": batch,
        "splits": reports,
    }
    overall_root.parent.parent.mkdir(parents=True, exist_ok=True)
    (overall_root.parent.parent / "overall_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    update_training_memory_table(output_root, dataset_root, markdown_path)
    return result


def run_six_round_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    codex_command: str = "codex",
    markdown_path: str | Path | None = None,
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    provider_mode: str = "rule_template_baseline",
    outer_transition: Callable[[int, list[dict[str, Any]]], dict[str, Any]] | None = None,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    split_payload = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_protocol_manifest(
        split_payload["train"],
        split_payload["dev"],
        split_payload["test"],
        dataset_fingerprint=metadata["fingerprint"],
    )
    protocol["provider_mode"] = provider_mode
    if diagnostic_only:
        protocol = {
            **protocol,
            "execution_mode": "diagnostic_precalibration",
            "formal_training_allowed": False,
            "visual_scores_permitted": False,
            "human_review": "deferred_until_after_Harness_upgrade",
        }
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    if director_agent is None or code_agent is None:
        director_agent, code_agent = build_dynamic_codex_agents(
            codex_command=codex_command,
            timeout_s=timeout_s,
            provider_mode=provider_mode,
        )
    memory_table = Path(markdown_path) if markdown_path is not None else root / "harness_training_memory.md"
    shared_cache = root / "code_cache"
    (root / "six_round_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    round_reports: list[dict[str, Any]] = []
    memory_reports: list[dict[str, Any]] = []
    for batch in protocol["rounds"]:
        round_number = batch["round"]
        def run_attempt(attempt_number: int) -> dict[str, Any]:
            return run_outer_attempt(
                root,
                round_number=round_number,
                attempt_number=attempt_number,
                dataset_root=dataset_root,
                harness_version=harness_version,
                evaluator_version=evaluator_version,
                blender_bin=blender_bin,
                workers=workers,
                timeout_s=timeout_s,
                vlm_model=vlm_model,
                markdown_path=memory_table,
                director_agent=director_agent,
                code_agent=code_agent,
                provider_mode=provider_mode,
                codex_command=codex_command,
                code_cache_dir=shared_cache,
            )

        transition = outer_transition or (
            lambda _attempt_number, _reports: {
                "action": "stop",
                "status": "awaiting_harness_patch",
                "reason": "no outer patch transition was supplied; preserve evidence and stop before a duplicate attempt",
            }
        )
        outer_loop = run_bounded_outer_attempts(
            run_attempt=run_attempt,
            transition=transition,
            max_attempts=5,
        )
        attempt = outer_loop["reports"][-1]
        # This is intentionally a separate full evaluation.  It covers all
        # 60 train and all 60 dev cases at every round, even though only the
        # new 10+10 batch is used for the candidate signal.
        overall = run_outer_overall(
            root,
            round_number=round_number,
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
            markdown_path=memory_table,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode=provider_mode,
            codex_command=codex_command,
            code_cache_dir=shared_cache,
        )
        round_report = {
            "round": round_number,
            "execution_mode": "diagnostic_precalibration" if diagnostic_only else "formal_training",
            "formal_training_allowed": False if diagnostic_only else None,
            "batch": batch,
            "attempt": attempt,
            "outer_loop": outer_loop,
            "overall": overall,
        }
        round_reports.append(round_report)
        memory_reports.append(
            {
                "round": round_number,
                "patch": _load_patch_metadata(root / f"round-{round_number:02d}"),
                "splits": attempt["splits"],
                "overall_evaluation": overall["splits"],
            }
        )
        (root / f"round-{round_number:02d}" / "overall_evaluation.json").write_text(
            json.dumps(round_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    memory_path = root / "memory" / "harness_updates.jsonl"
    write_harness_memory_jsonl(memory_path, memory_reports)
    result = {
        "protocol": protocol,
        "provider_mode": provider_mode,
        "execution_mode": "diagnostic_precalibration" if diagnostic_only else "formal_training",
        "formal_training_allowed": False if diagnostic_only else None,
        "visual_scores_permitted": False if diagnostic_only else None,
        "rounds": round_reports,
        "memory_table": str(memory_table.resolve()),
        "memory_jsonl": str(memory_path.resolve()),
    }
    (root / "six_round_training_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_multi_five_round_protocol(
    output_root: str | Path,
    *,
    dataset_root: str | Path,
    harness_version: str,
    evaluator_version: str,
    blender_bin: str,
    workers: int,
    timeout_s: int,
    vlm_model: str,
    markdown_path: str | Path,
    director_agent: DirectorAgent | None = None,
    code_agent: BlenderCodeAgent | None = None,
    provider_mode: str = "rule_template_baseline",
    codex_command: str = "codex",
    outer_transition: Callable[[int, list[dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute bounded real outer attempts plus one round-end evaluation.

    Without an explicit ``outer_transition`` callback the function performs
    one evidence-producing attempt and stops in ``awaiting_harness_patch``.
    This prevents a caller from accidentally re-rendering the same Harness;
    an injected Codex Host controller may apply one-owner patches and permit
    attempts 2--5.
    """
    dataset = Path(dataset_root)
    split_payload = json.loads((dataset / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_multi_five_round_manifest(
        split_payload["train"],
        split_payload["dev"],
        split_payload["test"],
        dataset_fingerprint=metadata["fingerprint"],
    )
    protocol["provider_mode"] = provider_mode
    root = Path(output_root)
    if director_agent is None or code_agent is None:
        director_agent, code_agent = build_dynamic_codex_agents(
            codex_command=codex_command,
            timeout_s=timeout_s,
            provider_mode=provider_mode,
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / "multi_five_protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    round_reports = []
    for batch in protocol["rounds"]:
        round_number = batch["round"]

        def run_attempt(attempt_number: int) -> dict[str, Any]:
            return run_outer_attempt(
                root,
                round_number=round_number,
                attempt_number=attempt_number,
                dataset_root=dataset_root,
                harness_version=harness_version,
                evaluator_version=evaluator_version,
                blender_bin=blender_bin,
                workers=workers,
                timeout_s=timeout_s,
                vlm_model=vlm_model,
                markdown_path=markdown_path,
                director_agent=director_agent,
                code_agent=code_agent,
                provider_mode=provider_mode,
                codex_command=codex_command,
            )

        transition = outer_transition or (
            lambda _attempt_number, _reports: {
                "action": "stop",
                "status": "awaiting_harness_patch",
                "reason": "no outer patch transition was supplied; preserve evidence and stop before a duplicate attempt",
            }
        )
        outer_loop = run_bounded_outer_attempts(
            run_attempt=run_attempt,
            transition=transition,
            max_attempts=5,
        )
        attempt = outer_loop["reports"][-1]
        overall = run_outer_overall(
            root,
            round_number=round_number,
            dataset_root=dataset_root,
            harness_version=harness_version,
            evaluator_version=evaluator_version,
            blender_bin=blender_bin,
            workers=workers,
            timeout_s=timeout_s,
            vlm_model=vlm_model,
            markdown_path=markdown_path,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode=provider_mode,
            codex_command=codex_command,
        )
        round_reports.append({"round": round_number, "attempt": attempt, "outer_loop": outer_loop, "overall": overall})
    result = {
        "protocol": protocol,
        "provider_mode": provider_mode,
        "rounds": round_reports,
        "memory_table": str(Path(markdown_path).resolve()),
    }
    (root / "multi_five_training_report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "protocol",
            "attempt",
            "overall",
            "diagnostic-attempt",
            "diagnostic-overall",
            "diagnostic-six-rounds",
            "multi-five-rounds",
            "six-rounds",
            "existing-rounds",
            "full-train",
            "all",
        ],
        required=True,
    )
    parser.add_argument("--dataset-root", default="dataset/vbench2-agent-training-index-v1")
    parser.add_argument("--readiness-report", default="out/training_readiness_report.json")
    parser.add_argument(
        "--formal-release-report",
        default="out/formal_release_gate_report.json",
        help="sealed G0-G3 pilot/shadow release report required by formal modes",
    )
    parser.add_argument("--round-root", default="out/training/agent-six-rounds-v1")
    parser.add_argument("--full-train-root", default="out/training/full-evaluation-real-v7")
    parser.add_argument("--harness-version", default="t2blendercodeharness-v5-executable-director")
    parser.add_argument("--evaluator-version", default=VISUAL_PRIMARY_VERSION)
    parser.add_argument("--blender-bin", default=r"D:\blender\blender.exe")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--vlm-model", choices=["gpt-5.6-luna", "gpt-5.6-terra"], default="gpt-5.6-luna")
    parser.add_argument("--codex-command", default="codex")
    parser.add_argument(
        "--provider-mode",
        choices=["rule_template_baseline", "model", "glm", "assistant"],
        default="glm",
        help="explicit baseline, legacy hybrid model, external Zhipu GLM, or driving-assistant-session plan+code provider",
    )
    parser.add_argument("--markdown-path", default="docs/t2blendercodeharness-agent-training-memory-v1.md")
    parser.add_argument("--round", dest="round_number", type=int)
    parser.add_argument("--attempt", dest="attempt_number", type=int)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    all_reports: dict[str, Any] = {}
    require_benchmark_training_dataset(dataset_root)
    formal_modes = {"attempt", "overall", "multi-five-rounds", "six-rounds", "full-train", "all"}
    diagnostic_modes = {"diagnostic-attempt", "diagnostic-overall", "diagnostic-six-rounds"}
    diagnostic_policy = None
    if args.mode in formal_modes:
        require_model_provider_mode(args.provider_mode)
        # Keep the prerequisite matrix as an explicit independent check; the
        # sealed G0--G3 report is an additional formal-release boundary.
        require_training_readiness(args.readiness_report)
        require_formal_training_release(args.readiness_report, args.formal_release_report)
    elif args.mode in diagnostic_modes:
        diagnostic_policy = require_diagnostic_training_readiness(args.readiness_report)
        diagnostic_manifest_path = Path(args.round_root) / "diagnostic_training_manifest.json"
        diagnostic_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_manifest_path.write_text(
            json.dumps(
                {
                    **diagnostic_policy,
                    "dataset_root": str(dataset_root.resolve()),
                    "harness_version": args.harness_version,
                    "evaluator_version": args.evaluator_version,
                    "blender_bin": str(Path(args.blender_bin).resolve()),
                    "workers": args.workers,
                    "protocol": "six_rounds_max_five_outer_attempts_10_train_plus_10_dev_then_120_overall",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    split_payload = json.loads((dataset_root / "splits.json").read_text(encoding="utf-8"))
    metadata = json.loads((dataset_root / "metadata.json").read_text(encoding="utf-8"))
    protocol = build_active_protocol_manifest(
        split_payload["train"], split_payload["dev"], split_payload["test"], dataset_fingerprint=metadata["fingerprint"], dataset_id=metadata.get("dataset_id")
    )
    protocol_path = Path(args.round_root) / ("multi_five_protocol.json" if protocol["protocol_version"] == "multi-five-rounds-v1" else "six_round_protocol.json")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.mode == "protocol":
        print(json.dumps(protocol, indent=2, sort_keys=True))
        return 0
    if args.mode in {"attempt", "overall", "diagnostic-attempt", "diagnostic-overall"}:
        if args.round_number is None:
            raise SystemExit("--round is required for --mode attempt/overall/diagnostic-attempt/diagnostic-overall")
        if args.mode in {"attempt", "diagnostic-attempt"}:
            if args.attempt_number is None:
                raise SystemExit("--attempt is required for --mode attempt/diagnostic-attempt")
            result = run_outer_attempt(
                args.round_root,
                round_number=args.round_number,
                attempt_number=args.attempt_number,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                markdown_path=args.markdown_path,
                provider_mode=args.provider_mode,
                codex_command=args.codex_command,
            )
        else:
            result = run_outer_overall(
                args.round_root,
                round_number=args.round_number,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                markdown_path=args.markdown_path,
                provider_mode=args.provider_mode,
                codex_command=args.codex_command,
            )
        if args.mode in {"diagnostic-attempt", "diagnostic-overall"}:
            result["execution_mode"] = "diagnostic_precalibration"
            result["formal_training_allowed"] = False
            result["visual_scores_permitted"] = False
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.mode in {"diagnostic-six-rounds", "multi-five-rounds", "six-rounds", "all"}:
        if args.mode == "diagnostic-six-rounds":
            all_reports["six_rounds"] = run_six_round_protocol(
                args.round_root,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                codex_command=args.codex_command,
                markdown_path=args.markdown_path,
                provider_mode=args.provider_mode,
                diagnostic_only=True,
            )
            print(json.dumps(all_reports, indent=2, sort_keys=True))
            return 0
        if protocol["protocol_version"] == "multi-five-rounds-v1":
            all_reports["multi_five_rounds"] = run_multi_five_round_protocol(
                args.round_root,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                markdown_path=args.markdown_path,
                provider_mode=args.provider_mode,
                codex_command=args.codex_command,
            )
        else:
            all_reports["six_rounds"] = run_six_round_protocol(
                args.round_root,
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
                codex_command=args.codex_command,
                markdown_path=args.markdown_path,
                provider_mode=args.provider_mode,
            )
    if args.mode == "existing-rounds":
        round_root = Path(args.round_root)
        for round_dir in sorted(round_root.glob("round-*")):
            round_reports = {}
            for split in ("train", "dev"):
                split_root = round_dir / "real" / split
                if not split_root.is_dir():
                    continue
                round_reports[split] = run_real_batch(
                    split_root,
                    dataset_root=dataset_root,
                    blender_bin=args.blender_bin,
                    workers=args.workers,
                    timeout_s=args.timeout_s,
                    vlm_model=args.vlm_model,
                )
            if round_reports:
                (round_dir / "real_round_report.json").write_text(
                    json.dumps(round_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                all_reports[round_dir.name] = round_reports
    if args.mode in {"full-train", "all"}:
        director_agent, code_agent = build_dynamic_codex_agents(
            codex_command=args.codex_command,
            timeout_s=args.timeout_s,
            provider_mode=args.provider_mode,
        )
        train_root = prepare_full_train(
            args.full_train_root,
            dataset_root=dataset_root,
            harness_version=args.harness_version,
            evaluator_version=args.evaluator_version,
            director_agent=director_agent,
            code_agent=code_agent,
            provider_mode=args.provider_mode,
        )
        all_reports["full_train"] = run_real_batch(
            train_root,
            dataset_root=dataset_root,
            blender_bin=args.blender_bin,
            workers=args.workers,
            timeout_s=args.timeout_s,
            vlm_model=args.vlm_model,
            markdown_path=args.markdown_path,
        )
        if args.mode == "all":
            dev_root = prepare_full_split(
                args.full_train_root,
                split="dev",
                dataset_root=dataset_root,
                harness_version=args.harness_version,
                evaluator_version=args.evaluator_version,
                director_agent=director_agent,
                code_agent=code_agent,
                provider_mode=args.provider_mode,
            )
            all_reports["full_dev"] = run_real_batch(
                dev_root,
                dataset_root=dataset_root,
                blender_bin=args.blender_bin,
                workers=args.workers,
                timeout_s=args.timeout_s,
                vlm_model=args.vlm_model,
            )
    summary_path = Path(args.full_train_root).parent / "real_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_reports, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(all_reports, indent=2, sort_keys=True))
    return 0 if all(item.get("status") == "complete" for item in _flatten_reports(all_reports)) else 2


def _flatten_reports(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for value in payload.values():
        if isinstance(value, dict) and "status" in value:
            result.append(value)
        elif isinstance(value, dict):
            result.extend(_flatten_reports(value))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
