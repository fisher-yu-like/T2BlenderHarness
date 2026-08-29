"""Top-level contract-first orchestration and resume checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .blender_adapter import BlenderAdapter
from .inner_loop import run_inner_loop
from .run_manifest import hash_prompt


class Orchestrator:
    STAGES = ["contract", "plan", "execute", "render", "evaluate", "repair", "finalize"]

    def __init__(self, *, adapter: Any | None = None):
        self.adapter = adapter or BlenderAdapter()
        self.stage_order: list[str] = []

    def run(
        self,
        case: dict[str, Any],
        harness_snapshot: dict[str, Any],
        output_dir: str | Path,
        *,
        resume: bool = False,
    ):
        root = Path(output_dir)
        selection_path = root / "final" / "selection.json"
        prompt = str(case.get("prompt", ""))
        if resume and selection_path.exists():
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            expected = {
                "prompt_hash": hash_prompt(prompt),
                "harness_version": str(harness_snapshot.get("version", "dev")),
            }
            if any(selection.get(key) != value for key, value in expected.items()):
                raise ValueError("resume fingerprint does not match the existing selection")
            self.stage_order = list(self.STAGES)
            from .contracts import RunResult

            return RunResult(
                run_id=selection["run_id"],
                status="success",
                selected_attempt=selection["selected_attempt"],
                attempts=[{"attempt": selection["selected_attempt"], "resumed": True}],
                final_score=selection.get("score"),
            )

        # This compatibility orchestrator delegates to the explicit legacy
        # baseline projection in ``run_inner_loop``.  Keep the input boundary
        # local so an empty prompt is rejected before an adapter is touched;
        # production agent runs use ``prepare_real_jobs`` and its dynamic
        # DirectorAgent gate.
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        self.stage_order = list(self.STAGES)
        return run_inner_loop(
            case,
            harness_snapshot,
            root,
            adapter=self.adapter,
        )
