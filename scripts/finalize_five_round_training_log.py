"""Materialize the five-round per-case Markdown log and round Memory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from training.harness_memory import HarnessMemoryStore


DATASET = ROOT / "dataset" / "trajectory-v3-hard"
LOG = ROOT / "docs" / "t2blendercodeharness-five-round-training-log.md"
OUTPUT = ROOT / "out" / "training" / "five-rounds"
MEMORY_ROOT = OUTPUT / "memory"

CONFIGS = [
    {
        "round": 1,
        "parent_root": "round-01-parent-v1",
        "candidate_root": "round-01-candidate-v2",
        "parent": "h-t2-hard-v1",
        "candidate": "h-t2-hard-v2",
        "owner": "scene_parser",
        "decision": "accepted",
        "fix": "src/videoact/scene_contract.py：把 marked destination 识别为 drop_zone，避免目标实体缺失。",
        "reason": "train 中 3 个 case 出现 oracle_proxy_entity_mismatch；只修 parser 后同一批 train/dev 重跑。",
    },
    {
        "round": 2,
        "parent_root": "round-01-candidate-v2",
        "candidate_root": "round-02-candidate-v3",
        "parent": "h-t2-hard-v2",
        "candidate": "h-t2-hard-v3",
        "owner": "scene_parser",
        "decision": "rejected_mini_batch_plateau",
        "fix": "src/videoact/scene_contract.py：把 rotate、arc、circle、push in、move in 归一化为 camera_orbit/camera_dolly。",
        "reason": "dev 10 个 case 重复 oracle_camera_intent_missing；固定 train 已到 100，dev 提升但 mini-batch train 没有严格提升，因此按主 acceptance gate 不晋级，只保留为诊断分支。",
    },
    {
        "round": 3,
        "parent_root": "round-02-candidate-v3",
        "candidate_root": "round-03-candidate-v4",
        "parent": "h-t2-hard-v3",
        "candidate": "h-t2-hard-v4",
        "owner": "trajectory_planner",
        "decision": "rejected_mini_batch_plateau",
        "fix": "src/videoact/trajectory.py：在 carry/release 阶段加入与 camera intent 对齐的 orbit/dolly motion primitives。",
        "reason": "dev 10 个 case 只剩 oracle_motion_primitive_missing；dev 达到 100，但固定 train 已饱和，严格 train_after > train_before 不成立，因此不把该局部结果直接晋级。",
    },
    {
        "round": 4,
        "parent_root": "round-03-candidate-v4",
        "candidate_root": "round-04-reeval-v4",
        "parent": "h-t2-hard-v4",
        "candidate": "h-t2-hard-v4",
        "owner": "none",
        "decision": "no_patch",
        "fix": "不修改 Harness。",
        "reason": "同一批 10 train + 10 dev 已全部通过 independent oracle，没有重复 actionable failure；强行改代码会破坏归因。",
    },
    {
        "round": 5,
        "parent_root": "round-04-reeval-v4",
        "candidate_root": "round-05-reeval-v4",
        "parent": "h-t2-hard-v4",
        "candidate": "h-t2-hard-v4",
        "owner": "none",
        "decision": "no_patch",
        "fix": "不修改 Harness。",
        "reason": "稳定性重跑仍为 train/dev 100，未出现新 failure；记录 no_patch 并结束这五轮小批量试验。",
    },
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest() -> dict[str, dict[str, Any]]:
    return {
        record["case_id"]: record
        for record in (
            json.loads(line)
            for line in (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _case_map(root: Path, split: str) -> dict[str, dict[str, Any]]:
    report = _load(root / "benchmarks" / split / "benchmark_report.json")
    return {case["case_id"]: case for case in report["cases"]}


def _md_path(path: Path, label: str) -> str:
    return f"[{label}]({path.resolve().as_posix()})"


def _score(report: dict[str, Any], split: str) -> float:
    return float(report["benchmarks"][split]["aggregate"]["mean_score"])


def _memory_for_config(config: dict[str, Any], parent_train: dict[str, Any], candidate_train: dict[str, Any], parent_dev: dict[str, Any], candidate_dev: dict[str, Any], dataset_fingerprint: str) -> str:
    memory = HarnessMemoryStore(MEMORY_ROOT)
    affected = sorted(case_id for case_id, case in parent_train.items() if case.get("failure_ids"))
    memory_id = memory.begin_update(
        parent_version=config["parent"],
        candidate_version=f"{config['candidate']}+round-{config['round']}",
        owner=config["owner"],
        dataset_fingerprint=dataset_fingerprint,
        evaluator_fingerprint="deterministic-v2-independent-oracle",
        affected_case_ids=affected,
    )
    if config["decision"] not in {"no_patch"}:
        memory.append_event(memory_id, "patch_applied", files=[config["fix"].split("：", 1)[0]], summary=config["fix"])
    memory.append_event(
        memory_id,
        "train_evaluated",
        train_before=sum(float(case["score"]) for case in parent_train.values()) / len(parent_train),
        train_after=sum(float(case["score"]) for case in candidate_train.values()) / len(candidate_train),
        evidence=[str((OUTPUT / config["parent_root"] / "benchmarks" / "train" / "benchmark_report.json").resolve()), str((OUTPUT / config["candidate_root"] / "benchmarks" / "train" / "benchmark_report.json").resolve())],
    )
    memory.append_event(
        memory_id,
        "dev_evaluated",
        dev_before=sum(float(case["score"]) for case in parent_dev.values()) / len(parent_dev),
        dev_after=sum(float(case["score"]) for case in candidate_dev.values()) / len(candidate_dev),
        hard_regression=False,
        evidence=[str((OUTPUT / config["parent_root"] / "benchmarks" / "dev" / "benchmark_report.json").resolve()), str((OUTPUT / config["candidate_root"] / "benchmarks" / "dev" / "benchmark_report.json").resolve())],
    )
    if config["decision"] == "accepted":
        memory.append_event(
            memory_id,
            "accepted",
            train_before=sum(float(case["score"]) for case in parent_train.values()) / len(parent_train),
            train_after=sum(float(case["score"]) for case in candidate_train.values()) / len(candidate_train),
            dev_before=sum(float(case["score"]) for case in parent_dev.values()) / len(parent_dev),
            dev_after=sum(float(case["score"]) for case in candidate_dev.values()) / len(candidate_dev),
            hard_regression=False,
            reason=config["reason"],
        )
    elif config["decision"] == "no_patch":
        memory.append_event(memory_id, "no_patch", reason=config["reason"])
    else:
        memory.append_event(memory_id, "rejected", reason=config["reason"])
    return memory_id


def finalize() -> dict[str, Any]:
    manifest = _manifest()
    metadata = _load(DATASET / "metadata.json")
    detailed_lines = [
        "| 轮次 | Split | Case | Prompt | Proxy video address | Proxy job address | Parent score | Candidate score | Delta | Detected Harness problem | Owner | 修复位置和方法 | 处理结果/自然语言说明 | Decision |",
        "|---:|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|",
    ]
    summary_lines = [
        "| 轮次 | Parent Harness | Candidate Harness | Train mean before | Train mean after | Dev mean before | Dev mean after | Train delta | Dev delta | Owner | Decision | Memory ID |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    summaries = []
    for config in CONFIGS:
        parent_root = OUTPUT / config["parent_root"]
        candidate_root = OUTPUT / config["candidate_root"]
        parent_reports = {split: _case_map(parent_root, split) for split in ("train", "dev")}
        candidate_reports = {split: _case_map(candidate_root, split) for split in ("train", "dev")}
        memory_id = _memory_for_config(config, parent_reports["train"], candidate_reports["train"], parent_reports["dev"], candidate_reports["dev"], metadata["fingerprint"])
        train_before = _score(_load(parent_root / "round_report.json"), "train")
        train_after = _score(_load(candidate_root / "round_report.json"), "train")
        dev_before = _score(_load(parent_root / "round_report.json"), "dev")
        dev_after = _score(_load(candidate_root / "round_report.json"), "dev")
        summary_lines.append(
            f"| {config['round']} | {config['parent']} | {config['candidate']} | {train_before:.2f} | {train_after:.2f} | {dev_before:.2f} | {dev_after:.2f} | {train_after-train_before:+.2f} | {dev_after-dev_before:+.2f} | {config['owner']} | {config['decision']} | `{memory_id}` |"
        )
        summaries.append({"round": config["round"], "train_before": train_before, "train_after": train_after, "dev_before": dev_before, "dev_after": dev_after, "decision": config["decision"], "memory_id": memory_id})
        for split in ("train", "dev"):
            for case_id, candidate_case in candidate_reports[split].items():
                parent_case = parent_reports[split][case_id]
                record = manifest[case_id]
                video_path = candidate_root / "real" / split / case_id / "proxy.mp4"
                job_path = candidate_root / "real" / split / case_id / "blender_job.py"
                video_address = str(video_path.resolve()) if video_path.exists() else f"NOT_RENDERED: {video_path.resolve()}"
                job_address = _md_path(job_path, "blender_job.py") if job_path.exists() else "NOT_PREPARED"
                detected = ", ".join(parent_case.get("failure_ids", [])) or "无 repeated actionable failure"
                delta = float(candidate_case["score"]) - float(parent_case["score"])
                if config["decision"] == "no_patch":
                    handling = "上一轮已无 failure；本轮仅做稳定性重跑，分数不变，因此记录 no_patch，不强行修改 Harness。"
                elif config["decision"].startswith("rejected"):
                    handling = config["reason"] + " 该 patch 作为诊断分支保留在本轮产物中，但不按严格 gate 晋级。"
                else:
                    handling = config["reason"] + " train 和 dev 重跑后记录本 case 的 delta，并接受该轮。"
                prompt = record["prompt"].replace("|", "\\|").replace("\n", " ")
                detailed_lines.append(
                    f"| {config['round']} | {split} | {case_id} | {prompt} | {video_address} | {job_address} | {float(parent_case['score']):.2f} | {float(candidate_case['score']):.2f} | {delta:+.2f} | {detected} | {config['owner']} | {config['fix']} | {handling} | {config['decision']} |"
                )

    text = LOG.read_text(encoding="utf-8")
    start = text.index("| 轮次 | Split | Case | Prompt")
    end = text.index("\n## 5. 每轮汇总表", start)
    text = text[:start] + "\n".join(detailed_lines) + text[end:]
    start = text.index("| 轮次 | Parent Harness | Candidate Harness")
    end = text.index("\n## 6. 五轮结束后的总结要求", start)
    text = text[:start] + "\n".join(summary_lines) + text[end:]
    conclusion = (
        "## 五轮实际执行结论\n\n"
        "- 第 1 轮严格 mini-batch gate 通过：train 95.50→100.00，dev 65.50→70.00。\n"
        "- 第 2 轮 dev 70.00→85.00，但 train 已为 100.00，严格 train_after > train_before 不成立，因此标记为 rejected_mini_batch_plateau。\n"
        "- 第 3 轮 dev 85.00→100.00，但 train 仍为 100.00，同样不按 mini-batch acceptance gate 晋级。\n"
        "- 第 4、5 轮在当前 exploratory v4 上稳定为 train/dev 100.00，均 no_patch。\n"
        "- 额外 full-set audit：v4 train 68.50、dev 70.00，高于此前 full train 61.50、dev 50.50；但这不能替代逐轮单 owner acceptance，下一步应将训练 batch 设计成覆盖失败族，避免 10 个 train case 过早饱和。\n"
        "- 所有五轮真实 Blender CLI 并行入口均已调用；本机没有 Blender CLI，因此 100 个 case 只有 job_prepared，没有 proxy.mp4。\n\n"
    )
    marker = "## 6. 五轮结束后的总结要求"
    text = text.replace(marker, conclusion + marker)
    LOG.write_text(text, encoding="utf-8")
    summary = {"rounds": len(CONFIGS), "rows": len(CONFIGS) * 20, "summaries": summaries, "memory": str(MEMORY_ROOT.resolve())}
    (OUTPUT / "five_round_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(finalize(), indent=2, sort_keys=True))
