"""Hash-bound run manifest helpers."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .contracts import RunManifest


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rollout_fingerprint(case_id: str, *, seed: int, harness_version: str) -> str:
    """Bind a rollout artifact to its stochastic seed and Harness version."""
    return hash_payload({"case_id": case_id, "seed": int(seed), "harness_version": harness_version})


def aggregate_rollouts(case_id: str, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independent runs of one case without collapsing their seeds."""
    if not rollouts:
        raise ValueError("at least one rollout is required")
    seeds = [int(item["seed"]) for item in rollouts]
    if len(seeds) != len(set(seeds)):
        raise ValueError("rollout seeds must have unique seeds")
    scores = [float(item["score"]) for item in rollouts]
    passed = [bool(item["passed"]) for item in rollouts]
    return {
        "case_id": case_id,
        "rollout_count": len(rollouts),
        "seeds": seeds,
        "mean_score": round(statistics.fmean(scores), 4),
        "score_std": round(statistics.pstdev(scores), 4),
        "pass_rate": round(sum(passed) / len(passed), 4),
        "rollouts": rollouts,
    }


def write_manifest(manifest: RunManifest, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
