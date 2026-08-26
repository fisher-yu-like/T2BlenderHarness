"""Write a human-readable catalog of the phase-1 prompts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset" / "trajectory-v3-single"
OUTPUT = ROOT / "docs" / "single-dataset-prompts.md"


def main() -> int:
    records = [
        json.loads(line)
        for line in (DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines = [
        "# trajectory-v3-single prompt catalog",
        "",
        "这是 phase-1 单实体训练集的完整 prompt 清单。每条记录严格包含一个 character 和一个 prop；原始 case 通过 `dataset_source_case_id` 保留映射。",
        "",
        f"- case count: {len(records)}",
        "- split: 50 train / 50 dev / 20 test",
        "- dataset: `dataset/trajectory-v3-single`",
        "- fingerprint: `b86b25c3d4b94b2c12ec56a881acd05ffd0973df85f9b4297db4cbc15c1978a6`",
        "",
        "| case | split | family | source case | prompt |",
        "|---|---|---|---|---|",
    ]
    for record in records:
        prompt = record["prompt"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {record['case_id']} | {record['split']} | {record['template_family']} | {record['dataset_source_case_id']} | {prompt} |"
        )
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT.resolve()), "case_count": len(records)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
