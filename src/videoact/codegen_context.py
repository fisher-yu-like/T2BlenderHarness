"""Load only validated real codegen examples into the L3 provider context."""

from __future__ import annotations

import json
from pathlib import Path
from typing_extensions import Literal

from .codegen_contracts import CodegenExample


ContextStatus = Literal["pass", "none", "invalid"]


def load_validated_context_examples(
    root: str | Path = "dataset/codegen-examples-v1",
    *,
    max_examples: int = 3,
) -> tuple[list[CodegenExample], ContextStatus]:
    """Return at most three validator-approved real plan/source examples.

    A missing manifest means no examples have been collected yet and is not a
    generation failure.  A present but invalid manifest is a hard context
    boundary failure so an unreviewed or template artifact cannot enter L3.
    """

    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    examples_root = Path(root)
    manifest = examples_root / "manifest.jsonl"
    if not manifest.is_file():
        return [], "none"

    from scripts.validate_codegen_examples import validate_codegen_examples

    report = validate_codegen_examples(examples_root)
    if report["status"] != "pass":
        return [], "invalid"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    examples: list[CodegenExample] = []
    for row in rows[:max_examples]:
        raw_root = row.get("artifact_root") or row.get("artifact_path")
        if not raw_root:
            return [], "invalid"
        run_root = Path(str(raw_root))
        if not run_root.is_absolute():
            run_root = examples_root / run_root
        try:
            director_plan = json.loads((run_root / "director_plan.json").read_text(encoding="utf-8"))
            source = (run_root / "blender_job.py").read_text(encoding="utf-8")
            examples.append(
                CodegenExample(
                    case_id=str(row["case_id"]),
                    director_plan=director_plan,
                    library_calls=[str(value) for value in row.get("library_calls", [])],
                    generated_code=source,
                    artifact_path=str(run_root.resolve()),
                    plan_hash=str(row["plan_hash"]),
                    code_hash=str(row["code_hash"]),
                    review_source=str(row["review_source"]),
                )
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return [], "invalid"
    return examples, "pass"


__all__ = ["ContextStatus", "load_validated_context_examples"]
