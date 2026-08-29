"""Reproducible paired statistics for Harness patch acceptance.

The module deliberately keeps statistical tolerance separate from the hard
artifact and semantic safety checks.  A noisy visual judge may move a score by
less than the frozen non-inferiority margin, but it may not hide a missing
artifact, an invalid execution, a lost required event, or a new hard failure.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


PAIRED_STATISTICS_VERSION = "paired-statistics-v1"
DEFAULT_SEED = 20260829
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_TRAIN_MIN_GAIN = 1.0
DEFAULT_DEV_NONINFERIORITY_MARGIN = -1.0
DEFAULT_SECONDARY_NONINFERIORITY_MARGIN = -1.0
SAFETY_METRICS = (
    "artifact_completion",
    "execution_validity",
    "required_event_failure_count",
    "hard_failure_count",
)


def _clean_deltas(deltas: Sequence[float] | None) -> list[float]:
    if deltas is None:
        return []
    result: list[float] = []
    for value in deltas:
        if isinstance(value, bool):
            raise ValueError("paired deltas must be numeric, not boolean")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("paired deltas must be finite")
        result.append(number)
    return result


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower]))


def bootstrap_mean_ci(
    deltas: Sequence[float],
    *,
    seed: int = DEFAULT_SEED,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Return a deterministic percentile bootstrap CI for the paired mean."""

    values = _clean_deltas(deltas)
    if not values:
        return {
            "version": PAIRED_STATISTICS_VERSION,
            "status": "unavailable",
            "n": 0,
            "mean": None,
            "ci_lower": None,
            "ci_upper": None,
            "seed": int(seed),
            "iterations": int(iterations),
            "alpha": float(alpha),
        }
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")

    rng = random.Random(int(seed))
    size = len(values)
    means = [
        statistics.fmean(values[rng.randrange(size)] for _ in range(size))
        for _ in range(int(iterations))
    ]
    means.sort()
    return {
        "version": PAIRED_STATISTICS_VERSION,
        "status": "scored",
        "n": size,
        "mean": statistics.fmean(values),
        "ci_lower": _quantile(means, alpha / 2.0),
        "ci_upper": _quantile(means, 1.0 - alpha / 2.0),
        "seed": int(seed),
        "iterations": int(iterations),
        "alpha": float(alpha),
    }


def _effect_size(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    deviation = statistics.stdev(values)
    mean = statistics.fmean(values)
    if deviation == 0.0:
        return None if mean == 0.0 else None
    return mean / deviation


def _safety_checks(
    safety_before: Mapping[str, Any] | None,
    safety_after: Mapping[str, Any] | None,
) -> tuple[dict[str, bool], list[str]]:
    before = safety_before if isinstance(safety_before, Mapping) else {}
    after = safety_after if isinstance(safety_after, Mapping) else {}
    checks: dict[str, bool] = {}
    missing: list[str] = []
    for metric in SAFETY_METRICS:
        before_value = before.get(metric)
        after_value = after.get(metric)
        check_name = f"{metric}_non_regression"
        if before_value is None or after_value is None:
            checks[check_name] = False
            missing.append(metric)
            continue
        try:
            left = float(before_value)
            right = float(after_value)
        except (TypeError, ValueError):
            checks[check_name] = False
            missing.append(metric)
            continue
        if not math.isfinite(left) or not math.isfinite(right):
            checks[check_name] = False
            missing.append(metric)
            continue
        if metric in {"required_event_failure_count", "hard_failure_count"}:
            checks[check_name] = right <= left
        else:
            checks[check_name] = right >= left
    if missing:
        checks["safety_metrics_complete"] = False
        return checks, ["safety_metrics_missing"]
    checks["safety_metrics_complete"] = True
    return checks, []


def evaluate_paired_acceptance(
    train_deltas: Sequence[float],
    dev_deltas: Sequence[float],
    *,
    secondary_deltas: Mapping[str, Sequence[float]] | None = None,
    safety_before: Mapping[str, Any] | None = None,
    safety_after: Mapping[str, Any] | None = None,
    train_min_gain: float = DEFAULT_TRAIN_MIN_GAIN,
    dev_noninferiority_margin: float = DEFAULT_DEV_NONINFERIORITY_MARGIN,
    secondary_noninferiority_margin: float = DEFAULT_SECONDARY_NONINFERIORITY_MARGIN,
    seed: int = DEFAULT_SEED,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    alpha: float = DEFAULT_ALPHA,
    require_safety_metrics: bool = True,
) -> dict[str, Any]:
    """Evaluate a patch with paired deltas and hard safety invariants."""

    train = _clean_deltas(train_deltas)
    dev = _clean_deltas(dev_deltas)
    train_stats = bootstrap_mean_ci(train, seed=seed, iterations=iterations, alpha=alpha)
    dev_stats = bootstrap_mean_ci(dev, seed=seed + 1, iterations=iterations, alpha=alpha)
    checks: dict[str, bool] = {
        "train_min_gain": bool(train_stats["status"] == "scored" and train_stats["mean"] >= float(train_min_gain)),
        "dev_noninferiority": bool(
            dev_stats["status"] == "scored"
            and dev_stats["ci_lower"] >= float(dev_noninferiority_margin)
        ),
    }
    failed: list[str] = [name for name, passed in checks.items() if not passed]

    if require_safety_metrics:
        safety, safety_failures = _safety_checks(safety_before, safety_after)
        checks.update(safety)
        failed.extend(safety_failures)
        failed.extend(name for name, passed in safety.items() if not passed and name not in failed)

    secondary_report: dict[str, Any] = {}
    for name, raw_values in (secondary_deltas or {}).items():
        values = _clean_deltas(raw_values)
        stats = bootstrap_mean_ci(values, seed=seed + 2 + len(secondary_report), iterations=iterations, alpha=alpha)
        check_name = f"secondary_{name}_noninferiority"
        allowed = bool(
            stats["status"] == "scored"
            and stats["ci_lower"] >= float(secondary_noninferiority_margin)
        )
        checks[check_name] = allowed
        if not allowed:
            failed.append(check_name)
        secondary_report[str(name)] = {
            **stats,
            "margin": float(secondary_noninferiority_margin),
            "allowed_noise": allowed,
        }

    failed = list(dict.fromkeys(failed))
    return {
        "version": PAIRED_STATISTICS_VERSION,
        "accepted": not failed,
        "checks": checks,
        "failed_checks": failed,
        "reason": "accepted_paired_statistics" if not failed else "rejected_paired_statistics: " + ", ".join(failed),
        "train_deltas": train,
        "dev_deltas": dev,
        "train": train_stats,
        "dev": dev_stats,
        "effect_size": _effect_size(train),
        "secondary": secondary_report,
        "thresholds": {
            "train_min_gain": float(train_min_gain),
            "dev_noninferiority_margin": float(dev_noninferiority_margin),
            "secondary_noninferiority_margin": float(secondary_noninferiority_margin),
            "alpha": float(alpha),
            "seed": int(seed),
            "bootstrap_iterations": int(iterations),
            "require_safety_metrics": bool(require_safety_metrics),
        },
    }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_DEV_NONINFERIORITY_MARGIN",
    "DEFAULT_SEED",
    "DEFAULT_TRAIN_MIN_GAIN",
    "PAIRED_STATISTICS_VERSION",
    "SAFETY_METRICS",
    "bootstrap_mean_ci",
    "evaluate_paired_acceptance",
]
