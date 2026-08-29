"""Blind visual scoring and paired statistics for the three-arm experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evaluator.aggregate import grouped_vlm_score  # noqa: E402
from evaluator.realism import score_realism  # noqa: E402


ARMS = ("pretrain", "trained", "direct_code")
ARM_LABELS = {"pretrain": "pretrain", "trained": "trained", "direct_code": "direct_code"}
ACTION_VARIANTS = ("direct_transfer", "reveal_elliptical_return", "subjectless_handoff_return", "parallel_transfer")
TASK_DIMENSIONS = (
    "prompt_compliance", "physical_plausibility", "object_trajectory", "event_timing",
    "camera_coverage", "camera_innovation", "character_trajectory", "temporal_smoothness",
    "visual_clarity",
)
REALISM_DIMENSIONS = (
    "appearance_detail", "physical_realism", "spatial_consistency", "motion_naturalness", "visual_presentation",
)
VISUAL_DIMENSIONS = TASK_DIMENSIONS + REALISM_DIMENSIONS
BLIND_REVIEW_VERSION = "three-arm-blind-v1"


def _json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default


def _gm(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.prod(max(0.0, float(value)) for value in values) ** (1.0 / len(values))


def _normalise_frame(path: str | Path) -> str:
    return str(Path(path).resolve())


def validate_blind_review(
    payload: dict[str, Any],
    *,
    expected_sample_label: str,
    expected_frames: list[str | Path],
) -> dict[str, Any]:
    """Validate one sample's review and preserve nulls as unavailable."""
    if payload.get("review_version") != BLIND_REVIEW_VERSION:
        raise ValueError("invalid three-arm blind review version")
    if payload.get("reviewer") != "codex-assistant":
        raise ValueError("three-arm review must identify codex-assistant")
    if payload.get("sample_label") != expected_sample_label:
        raise ValueError("review sample label does not match the blind request")
    actual_frames = [_normalise_frame(path) for path in payload.get("sampled_frames", [])]
    requested_frames = [_normalise_frame(path) for path in expected_frames]
    if actual_frames != requested_frames:
        raise ValueError("review frames do not match the exact sampled frames")
    if payload.get("status") not in {"complete", "unavailable"}:
        raise ValueError("review status must be complete or unavailable")
    scores = payload.get("scores")
    if not isinstance(scores, dict) or set(scores) != set(VISUAL_DIMENSIONS):
        raise ValueError("review must contain exactly all fourteen visual dimensions")
    for name in VISUAL_DIMENSIONS:
        value = scores[name]
        if payload["status"] == "unavailable":
            if value is not None:
                raise ValueError("unavailable review dimensions must remain null")
        elif value is None or not isinstance(value, (int, float)) or not 0 <= float(value) <= 100:
            raise ValueError(f"complete review dimension is invalid: {name}")
    evidence = payload.get("visible_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("review requires visible_evidence")
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    if not 0 <= confidence <= 1:
        raise ValueError("review confidence must be between 0 and 1")
    return {
        "review_version": BLIND_REVIEW_VERSION,
        "status": payload["status"],
        "review_source": payload.get("review_source", "assistant_local_review"),
        "reviewer": "codex-assistant",
        "sample_label": expected_sample_label,
        "sampled_frames": [_normalise_frame(path) for path in expected_frames],
        "scores": {name: (None if scores[name] is None else float(scores[name])) for name in VISUAL_DIMENSIONS},
        "visible_evidence": [str(item) for item in evidence],
        "weaknesses": [str(item) for item in (payload.get("weaknesses") or [])],
        "confidence": confidence,
    }


def score_blind_review(
    review: dict[str, Any],
    *,
    deterministic_score: float | None,
    geometry_report: dict[str, Any] | None,
    visual_report: dict[str, Any] | None,
    action_variant: str | None = None,
    arm: str | None = None,
) -> dict[str, Any]:
    """Score one blind review; action and arm parameters are diagnostics only.

    They are accepted to make accidental conditioning visible at call sites,
    but are deliberately never read in the formula.
    """
    del action_variant, arm
    scores = review.get("scores") or {}
    complete = review.get("status") == "complete" and float(review.get("confidence", 0.0) or 0.0) >= 0.6
    if not complete:
        artifact_review = {
            "status": "unavailable",
            "source": "assistant_local_review",
            "confidence": 0.0,
            "scores": {},
        }
        realism = score_realism(geometry_report or {}, visual_report or {}, artifact_review)
        return {
            "task_vlm": None,
            "task_final": None,
            "semantic": None,
            "choreography": None,
            "trajectory_diagnostic": None,
            "camera_diagnostic": None,
            "realism_final": None,
            "artifact_only_realism": realism.get("score"),
            "realism": realism,
            "review_status": "unavailable" if review.get("status") == "unavailable" else "needs_human_review",
            "review_confidence": float(review.get("confidence", 0.0) or 0.0),
        }
    semantic = _gm([float(scores[name]) for name in ("prompt_compliance", "physical_plausibility", "object_trajectory", "event_timing")])
    choreography = _gm([float(scores[name]) for name in ("camera_coverage", "camera_innovation", "character_trajectory", "temporal_smoothness")])
    task_vlm = round(0.45 * semantic + 0.45 * choreography + 0.10 * float(scores["visual_clarity"]), 4)
    task_final = None if deterministic_score is None else round(0.20 * float(deterministic_score) + 0.80 * task_vlm, 4)
    artifact_review = {
        "status": "complete",
        "source": "assistant_local_review",
        "confidence": float(review["confidence"]),
        "scores": {name: float(scores[name]) for name in REALISM_DIMENSIONS},
    }
    realism = score_realism(geometry_report or {}, visual_report or {}, artifact_review)
    trajectory = _gm([float(scores[name]) for name in ("object_trajectory", "character_trajectory", "event_timing", "temporal_smoothness")])
    camera = _gm([float(scores[name]) for name in ("camera_coverage", "camera_innovation", "visual_clarity")])
    return {
        "task_vlm": task_vlm,
        "task_final": task_final,
        "semantic": round(semantic, 4),
        "choreography": round(choreography, 4),
        "trajectory_diagnostic": round(trajectory, 4),
        "camera_diagnostic": round(camera, 4),
        "realism_final": realism.get("score"),
        "artifact_only_realism": realism.get("artifact_only_unbounded_score"),
        "realism": realism,
        "review_status": "complete",
        "review_confidence": float(review["confidence"]),
    }


def action_variant(record: dict[str, Any], index: int | None = None) -> str:
    explicit = record.get("action_variant")
    if explicit in ACTION_VARIANTS:
        return str(explicit)
    family = str(record.get("template_family") or "")
    for variant in ACTION_VARIANTS:
        if family.endswith(variant):
            return variant
    if index is not None:
        return ACTION_VARIANTS[index % len(ACTION_VARIANTS)]
    return "unknown"


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for part in field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(mean(values), 4) if values else None


def _bootstrap_ci(values: list[float], *, seed: int = 20260827, samples: int = 2000) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        value = round(float(values[0]), 4)
        return [value, value]
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    return [round(estimates[int(0.025 * (len(estimates) - 1))], 4), round(estimates[int(0.975 * (len(estimates) - 1))], 4)]


def _issue_text(run_dir: Path, deterministic: dict[str, Any], geometry: dict[str, Any]) -> str:
    findings = []
    for payload in (deterministic.get("findings") or [], deterministic.get("director_findings") or [], deterministic.get("interaction_findings") or [], geometry.get("findings") or []):
        if isinstance(payload, list):
            findings.extend(payload)
    messages = [str(item.get("message") or item.get("failure_id")) for item in findings if isinstance(item, dict)]
    if messages:
        return "; ".join(messages[:4])
    direct = _json(run_dir / "direct_code_manifest.json", {})
    if direct:
        return "raw prompt→Blender code omitted Director event graph, interaction lifecycle, and contract-driven camera choreography"
    return "no evaluator finding"


def _run_row(run_dir: Path, record: dict[str, Any], review_payload: dict[str, Any] | None, expected_label: str | None, arm: str, variant: str) -> dict[str, Any]:
    manifest = _json(run_dir / "run_manifest.json", {})
    deterministic = _json(run_dir / "deterministic_report.json", {})
    geometry = _json(run_dir / "geometry_report.json", {})
    visual = _json(run_dir / "visual_evidence.json", {})
    expected_frames = []
    if review_payload:
        expected_frames = review_payload.get("sampled_frames") or []
    review = None
    score: dict[str, Any] = {
        "task_vlm": None, "task_final": None, "semantic": None, "choreography": None,
        "trajectory_diagnostic": None, "camera_diagnostic": None, "realism_final": None,
        "artifact_only_realism": None, "realism": {}, "review_status": "missing", "review_confidence": None,
    }
    review_scores = {name: None for name in VISUAL_DIMENSIONS}
    review_error = None
    if review_payload and expected_label:
        try:
            review = validate_blind_review(review_payload, expected_sample_label=expected_label, expected_frames=expected_frames)
            review_scores = review["scores"]
            score = score_blind_review(
                review,
                deterministic_score=deterministic.get("score") if isinstance(deterministic.get("score"), (int, float)) else None,
                geometry_report=geometry,
                visual_report=visual,
                action_variant=variant,
                arm=arm,
            )
        except (TypeError, ValueError) as exc:
            review_error = f"{type(exc).__name__}: {exc}"
            score["review_status"] = "invalid"
    return {
        "arm": arm,
        "case_id": record["case_id"],
        "split": record["split"],
        "category": record.get("category"),
        "source_dimension": record.get("source_dimension"),
        "action_variant": variant,
        "prompt": record["prompt"],
        "proxy_video": str((run_dir / "proxy.mp4").resolve()) if (run_dir / "proxy.mp4").is_file() else None,
        "run_dir": str(run_dir.resolve()),
        "harness_version": manifest.get("harness_version"),
        "planning_mode": manifest.get("planning_mode") or ("direct_prompt_code" if (run_dir / "direct_code_manifest.json").is_file() else "director"),
        "plan_hash": manifest.get("plan_hash"),
        "director_plan_hash": manifest.get("director_plan_hash"),
        "artifact_status": "complete" if (run_dir / "proxy.mp4").is_file() and (run_dir / "proxy.blend").is_file() else "incomplete",
        "deterministic_score": deterministic.get("score"),
        "deterministic_score_kind": deterministic.get("score_kind", "contract_deterministic"),
        "director_plan_score": deterministic.get("director_plan_score"),
        "harness_issue": _issue_text(run_dir, deterministic, geometry),
        "review_sample_label": expected_label,
        "review_error": review_error,
        "review_scores": review_scores,
        **score,
    }


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_curve(rows: list[dict[str, Any]], output: Path) -> None:
    width, height = 1900, 1050
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image)
    draw.text((60, 22), "Three-arm blind comparison: Director pretrain / trained / direct prompt-code", fill=(25, 35, 50), font=_font(27))
    panels = [("task_vlm", "Blind task visual score", 105), ("realism_final", "Independent realism score", 555)]
    colors = {"pretrain": (120, 130, 145), "trained": (25, 120, 78), "direct_code": (190, 95, 35)}
    max_index = max((int(row.get("case_index", 0)) for row in rows), default=1)
    for field, title, top in panels:
        left, right, bottom = 100, width - 70, top + 315
        draw.text((left, top - 30), title, fill=(45, 55, 70), font=_font(20))
        draw.line((left, bottom, right, bottom), fill=(80, 90, 105), width=2)
        draw.line((left, top, left, bottom), fill=(80, 90, 105), width=2)
        for tick in range(0, 101, 20):
            y = bottom - (bottom - top) * tick / 100
            draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
            draw.text((49, y - 8), str(tick), fill=(90, 100, 115), font=_font(14))
        for arm in ARMS:
            points = []
            for row in sorted((item for item in rows if item["arm"] == arm), key=lambda item: int(item.get("case_index", 0))):
                value = row.get(field)
                if not isinstance(value, (int, float)):
                    continue
                x = left + (right - left) * int(row.get("case_index", 0)) / max(1, max_index)
                y = bottom - (bottom - top) * float(value) / 100
                points.append((x, y))
            if len(points) >= 2:
                draw.line(points, fill=colors[arm], width=3)
            for point in points[::max(1, len(points) // 25)]:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=colors[arm])
        for index, arm in enumerate(ARMS):
            x = right - 390 + index * 130
            draw.line((x, top + 8, x + 25, top + 8), fill=colors[arm], width=4)
            draw.text((x + 31, top), arm, fill=(65, 75, 90), font=_font(14))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        payload[arm] = {
            "cases": len(subset),
            "task_vlm_mean": _mean(subset, "task_vlm"),
            "task_final_mean": _mean(subset, "task_final"),
            "realism_final_mean": _mean(subset, "realism_final"),
            "artifact_only_realism_mean": _mean(subset, "artifact_only_realism"),
            "trajectory_diagnostic_mean": _mean(subset, "trajectory_diagnostic"),
            "camera_diagnostic_mean": _mean(subset, "camera_diagnostic"),
            "review_complete": sum(row.get("review_status") == "complete" for row in subset),
            "review_unavailable": sum(row.get("review_status") != "complete" for row in subset),
            "dimensions": {name: _mean(subset, f"review_scores.{name}") for name in VISUAL_DIMENSIONS},
        }
    return payload


def _flatten_dimension(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for row in rows:
        copied = dict(row)
        for name, value in (row.get("review_scores") or {}).items():
            copied[f"review_scores.{name}"] = value
        flat.append(copied)
    return flat


def _paired(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(row["case_id"], {})[row["arm"]] = row
    deltas: dict[str, dict[str, list[float]]] = {}
    comparisons = (("trained", "pretrain"), ("trained", "direct_code"), ("pretrain", "direct_code"))
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        deltas[key] = {"task_vlm": [], "realism_final": [], "trajectory_diagnostic": [], "camera_diagnostic": []}
        for arms in by_case.values():
            if left not in arms or right not in arms:
                continue
            for field in deltas[key]:
                a, b = arms[left].get(field), arms[right].get(field)
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    deltas[key][field].append(float(a) - float(b))
    summary = {}
    for key, fields in deltas.items():
        summary[key] = {
            field: {
                "n": len(values),
                "mean": round(mean(values), 4) if values else None,
                "ci95": _bootstrap_ci(values, seed=20260827 + index),
            }
            for index, (field, values) in enumerate(fields.items())
        }
    return summary


def build_report(
    *,
    dataset_root: str | Path,
    pretrain_root: str | Path,
    trained_root: str | Path,
    direct_root: str | Path,
    blind_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    dataset = Path(dataset_root)
    records = [
        json.loads(line)
        for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    blind = _json(Path(blind_root) / "blind_manifest.json", {})
    mapping = blind.get("mapping") or {}
    review_dir = Path(blind_root) / "reviews"
    roots = {"pretrain": Path(pretrain_root), "trained": Path(trained_root), "direct_code": Path(direct_root)}
    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for case_index, record in enumerate(records):
        variant = action_variant(record, case_index)
        case_rows = []
        case_review = _json(review_dir / f"{record['case_id']}.json", {})
        samples = case_review.get("samples") if isinstance(case_review, dict) else {}
        for arm in ARMS:
            run_dir = roots[arm] / "real" / record["split"] / record["case_id"]
            label = (mapping.get(record["case_id"]) or {}).get(arm)
            payload = (samples or {}).get(label) if label else None
            row = _run_row(run_dir, record, payload, label, arm, variant)
            row["case_index"] = case_index
            rows.append(row)
            case_rows.append(row)
        pairs.append({"case_id": record["case_id"], "case_index": case_index, "split": record["split"], "action_variant": variant, "prompt": record["prompt"], "arms": {row["arm"]: row for row in case_rows}})

    flat = _flatten_dimension(rows)
    groups: dict[str, list[dict[str, Any]]] = {"all": rows, "train": [row for row in rows if row["split"] == "train"], "dev": [row for row in rows if row["split"] == "dev"]}
    for variant in ACTION_VARIANTS:
        groups[f"action:{variant}"] = [row for row in rows if row["action_variant"] == variant]
    summary = {name: _group_summary(items) for name, items in groups.items()}
    payload = {
        "benchmark_id": "vbench-derived-100-three-arm-blind-v1",
        "dataset_root": str(dataset.resolve()),
        "pretrain_root": str(Path(pretrain_root).resolve()),
        "trained_root": str(Path(trained_root).resolve()),
        "direct_root": str(Path(direct_root).resolve()),
        "blind_root": str(Path(blind_root).resolve()),
        "case_count": len(records),
        "arm_count": len(ARMS),
        "row_count": len(rows),
        "visual_dimensions": list(VISUAL_DIMENSIONS),
        "summary": summary,
        "paired_deltas": _paired(rows),
        "pairs": pairs,
        "flat_rows": flat,
        "policy": {
            "ranking": "task_vlm and realism_final are primary; task_final is a diagnostic only",
            "deterministic": "A/B contract score is diagnostic; C runtime score is eligibility only",
            "missing": "unavailable dimensions and scores remain null and are excluded from means",
            "blind": "sample labels are randomized per case and action/arm labels are absent from review requests",
            "action_variants": "all four variants are comparison targets; no action-specific bonus is applied",
        },
    }
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "three_arm_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _draw_curve(rows, output / "three-arm-blind-curves.png")
    lines = [
        "# 三臂 Harness 盲评与真实视频消融实验",
        "",
        f"样本数：{len(records)}；视频行数：{len(rows)}；视觉维度：14 个。排名主指标为 `task_vlm` 与 `realism_final`，不把 Director 专属诊断分数加回跨臂总分。",
        "",
        f"![三臂曲线]({(output / 'three-arm-blind-curves.png').resolve().as_posix()})",
        "",
        "## Overall / split / action summary",
        "",
        "| group | arm | cases | task_vlm | task_final diagnostic | realism_final | trajectory diagnostic | camera diagnostic | complete reviews |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name, group in summary.items():
        for arm in ARMS:
            item = group[arm]
            lines.append(f"| {group_name} | {arm} | {item['cases']} | {item['task_vlm_mean']} | {item['task_final_mean']} | {item['realism_final_mean']} | {item['trajectory_diagnostic_mean']} | {item['camera_diagnostic_mean']} | {item['review_complete']} |")
    lines.extend(["", "## Paired deltas and 95% bootstrap intervals", "", "| comparison | metric | n | mean delta | 95% CI |", "|---|---|---:|---:|---|"])
    for comparison, fields in payload["paired_deltas"].items():
        for field, item in fields.items():
            lines.append(f"| {comparison} | {field} | {item['n']} | {item['mean']} | {item['ci95']} |")
    lines.extend(["", "## All 14 visual dimensions", "", "| group | arm | " + " | ".join(VISUAL_DIMENSIONS) + " |", "|---|---|" + "---:|" * len(VISUAL_DIMENSIONS)])
    for group_name, group in summary.items():
        for arm in ARMS:
            lines.append("| " + " | ".join([group_name, arm] + [str(group[arm]["dimensions"][name]) for name in VISUAL_DIMENSIONS]) + " |")
    lines.extend(["", "## Per-case real-video table", "", "| case | split | action | prompt | pretrain proxy video | trained proxy video | direct-code proxy video | pretrain task_vlm | trained task_vlm | direct task_vlm | pretrain realism | trained realism | direct realism | trained-pretrain task delta | trained-pretrain realism delta | Harness error finding |", "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"])
    for pair in pairs:
        arms = pair["arms"]
        prompt = str(pair["prompt"]).replace("|", "\\|").replace("\n", " ")
        trained_task, pretrain_task = arms["trained"].get("task_vlm"), arms["pretrain"].get("task_vlm")
        trained_real, pretrain_real = arms["trained"].get("realism_final"), arms["pretrain"].get("realism_final")
        task_delta = round(trained_task - pretrain_task, 4) if isinstance(trained_task, (int, float)) and isinstance(pretrain_task, (int, float)) else None
        real_delta = round(trained_real - pretrain_real, 4) if isinstance(trained_real, (int, float)) and isinstance(pretrain_real, (int, float)) else None
        issue = "; ".join(f"{arm}: {arms[arm].get('harness_issue')}" for arm in ARMS)
        lines.append(f"| {pair['case_id']} | {pair['split']} | {pair['action_variant']} | {prompt} | {arms['pretrain'].get('proxy_video')} | {arms['trained'].get('proxy_video')} | {arms['direct_code'].get('proxy_video')} | {arms['pretrain'].get('task_vlm')} | {arms['trained'].get('task_vlm')} | {arms['direct_code'].get('task_vlm')} | {arms['pretrain'].get('realism_final')} | {arms['trained'].get('realism_final')} | {arms['direct_code'].get('realism_final')} | {task_delta} | {real_delta} | {issue} |")
    (output / "three-arm-ablation-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pretrain-root", required=True)
    parser.add_argument("--trained-root", required=True)
    parser.add_argument("--direct-root", required=True)
    parser.add_argument("--blind-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_report(dataset_root=args.dataset_root, pretrain_root=args.pretrain_root, trained_root=args.trained_root, direct_root=args.direct_root, blind_root=args.blind_root, output_root=args.out)
    print(json.dumps({"case_count": payload["case_count"], "row_count": payload["row_count"], "summary_groups": list(payload["summary"])}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
