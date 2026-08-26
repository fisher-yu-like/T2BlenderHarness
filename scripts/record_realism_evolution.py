"""Append the audited v3 realism evolution to experiment memory."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "out" / "complex-realism-v1" / "evolution-v1" / "realism_evolution_v3_report.json"
BASELINE = ROOT / "out" / "complex-realism-v1" / "real" / "test"
CANDIDATE = ROOT / "out" / "complex-realism-v1" / "evolution-v1" / "candidate-v2"
DATASET = ROOT / "dataset" / "complex-realism-v1" / "manifest.jsonl"
DOC = ROOT / "docs" / "t2blendercodeharness-six-round-training-memory.md"
EVOLUTION_ROOT = ROOT / "out" / "complex-realism-v1" / "evolution-v1"
MARKER = "<!-- REALISM_EVOLUTION_V3 -->"


def _md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    records = {
        record["case_id"]: record
        for record in (json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    rows = []
    for item in report["cases"]:
        prompt = _md(records[item["case_id"]]["prompt"])
        rows.append(
            f"| {item['case_id']} | {prompt} | [{_md(item['baseline_realism_score'])}]({_md(item['baseline_video']).replace(chr(92), '/')}) | [{_md(item['candidate_realism_score'])}]({_md(item['candidate_video']).replace(chr(92), '/')}) | {item['baseline_realism_score']:.4f} → {item['candidate_realism_score']:.4f} ({item['realism_delta']:+.4f}) | {item['candidate_realism_score_kind']} / review_required={item['candidate_requires_independent_review']} | {', '.join(item['candidate_deterministic_findings']) or 'none'} |"
        )
    aggregate = report["aggregate"]
    acceptance = report["acceptance"]
    decision = "accepted as artifact-geometry evolution only" if acceptance["accepted"] else "rejected"
    section = f"""

{MARKER}
## 真实化 evaluator v3 重审（旧 v2 结果 superseded）

本节修正了旧 `realism-v2-independent-geometry` 的饱和问题。旧版 candidate-v2 的 100 分只表示满足 renderer 自己声明的 mesh 门槛，不代表视频真实感；旧报告保留在 `realism_evolution_v2_report.json`，但不再作为接受依据。v3 使用真实 Blender geometry、真实采样 PNG 帧和明确的 artifact-only 不确定性保留。

| case | prompt | baseline 视频/分数 | candidate 视频/分数 | v3 分数变化 | 分数类型 | deterministic 发现 |
|---|---|---|---|---:|---|---|
{chr(10).join(rows)}

| 指标 | baseline | candidate | delta | 结论 |
|---|---:|---:|---:|---|
| legacy mean | {aggregate['legacy_before']:.4f} | {aggregate['legacy_after']:.4f} | {aggregate['legacy_delta']:+.4f} | {'PASS' if acceptance['legacy_mean_non_regression'] else 'FAIL'} |
| artifact-only v3 mean | {aggregate['realism_before']:.4f} | {aggregate['realism_after']:.4f} | {aggregate['realism_delta']:+.4f} | {'PASS' if acceptance['realism_mean_improved'] else 'FAIL'} |
| candidate render success | — | 8/8 | — | {'PASS' if acceptance['all_candidate_renders_ok'] else 'FAIL'} |
| candidate geometry hard fail | 8/8 | 0/8 | — | {'PASS' if acceptance['candidate_geometry_hard_fail_count'] == 0 else 'FAIL'} |

结论：**{decision}**。candidate 的 v3 分数范围为 69.8615–72.6681，均标记 `artifact_only_proxy`、`realism_claim=not_established`、`requires_independent_review=true`；它证明结构化 proxy 相对 primitive baseline 有改善，但没有证明照片级真实性、动作语义或摄像机编排已经通过独立视觉审查。

v3 公式为 `G=.20 coverage+.20 topology+.20 representation+.10 semantics+.30 structure`，`V=.20 availability+.15 resolution+.25 foreground+.20 edge+.20 temporal`，`A_raw=.60G+.40V`，无独立 review 时 `score=min(80,.80A_raw)`。只有真实完整的 `gpt-5.6-luna`、`gpt-5.6-terra` 或人工 review 才能使用 `R=.45 semantic+.45 choreography+.10 presentation` 并以 `.20 artifact + .80 R` 融合。
"""
    existing = DOC.read_text(encoding="utf-8") if DOC.exists() else "# T2Blendercodeharness training memory\n"
    if MARKER in existing:
        existing = existing.split(MARKER, 1)[0].rstrip() + section
    else:
        existing = existing.rstrip() + section
    DOC.write_text(existing, encoding="utf-8")

    patch_manifest = {
        "evolution_version": "realism-evolution-v3",
        "parent_version": "h-t2-hard-v4-camera-closeup-scene-parser",
        "candidate_version": "h-t2-hard-v4-geometry-v1",
        "owner": "proxy_renderer",
        "files": ["blender/real_proxy_job.py"],
        "evaluator_files": ["evaluator/geometry_realism.py", "evaluator/realism.py", "evaluator/visual_evidence.py", "scripts/inspect_blend_geometry.py"],
        "dataset": "dataset/complex-realism-v1",
        "baseline_root": str(BASELINE.resolve()),
        "candidate_root": str(CANDIDATE.resolve()),
        "supersedes": "realism-evolution-v2 / geometry-only 100-point saturation",
        "independent_review_available": False,
        "aggregate": aggregate,
        "acceptance": acceptance,
        "decision": decision,
        "handling": "Keep the proxy_renderer candidate because legacy scores did not regress and the artifact-only score improved, but do not call it photorealism or use it as independent semantic review.",
        "attempt_history": [
            {"attempt": "candidate-v1", "status": "superseded", "reason": "evaluator-only primitive false positive"},
            {"attempt": "candidate-v2", "status": "accepted_for_geometry_only", "reason": "paired legacy gate and geometry hard gate passed; v2 realism score later superseded by anti-saturation v3"},
        ],
    }
    (EVOLUTION_ROOT / "patch_manifest_v3.json").write_text(json.dumps(patch_manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    memory_dir = EVOLUTION_ROOT / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "evaluator_audit_and_realism_rescore",
        "evolution_version": "realism-evolution-v3",
        "owner": "evaluator",
        "decision": decision,
        "legacy_delta": aggregate["legacy_delta"],
        "artifact_only_delta": aggregate["realism_delta"],
        "superseded_report": str((EVOLUTION_ROOT / "realism_evolution_v2_report.json").resolve()),
        "evidence": str(REPORT.resolve()),
    }
    with (memory_dir / "harness_updates.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"document": str(DOC), "patch_manifest": str(EVOLUTION_ROOT / "patch_manifest_v3.json"), "decision": decision, "aggregate": aggregate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
