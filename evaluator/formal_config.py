"""Frozen identity configuration for formal generator/judge runs.

The configuration is deliberately separate from transport credentials.  It
records the two generation boundaries (external Director and local Blender
codegen) and the two visual-judge snapshots a formal run claims to use;
readiness rejects missing or colliding identities before a training run can
start.  ``generator_model_id`` remains as a backwards-compatible composite
label for older reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


FORMAL_CONFIG_VERSION = "formal-evaluator-config-v1"
DEFAULT_PAIRED_STATISTICS_VERSION = "paired-statistics-v1"


def _required_text(value: Any, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"formal evaluator config missing {field}")
    return text


@dataclass(frozen=True)
class FormalEvaluatorConfig:
    generator_model_id: str
    primary_judge_model_id: str
    audit_judge_model_id: str
    director_model_id: str = ""
    codegen_model_id: str = ""
    director_provider_kind: str = "external_openai_compatible"
    codegen_provider_kind: str = "codex_exec_local"
    primary_blind_payload_version: str = "primary-blind-v1"
    score_policy_version: str = "scoring-v7-pending-calibration"
    sampling_policy_version: str = "event-aligned-uniform-v1"
    paired_statistics_version: str = DEFAULT_PAIRED_STATISTICS_VERSION
    bootstrap_seed: int = 20260829
    bootstrap_iterations: int = 2000
    alpha: float = 0.05
    train_min_gain: float = 1.0
    dev_noninferiority_margin: float = -1.0
    secondary_noninferiority_margin: float = -1.0
    confidence_threshold: float = 0.6
    evidence_completeness_threshold: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FormalEvaluatorConfig":
        legacy_generator = str(value.get("generator_model_id") or "").strip()
        director_model_id = _required_text(
            value.get("director_model_id") or legacy_generator,
            "director_model_id",
        )
        codegen_model_id = _required_text(
            value.get("codegen_model_id") or legacy_generator,
            "codegen_model_id",
        )
        generator_model_id = _required_text(
            value.get("generator_model_id")
            or f"{director_model_id}|{codegen_model_id}",
            "generator_model_id",
        )
        config = cls(
            generator_model_id=generator_model_id,
            primary_judge_model_id=_required_text(value.get("primary_judge_model_id"), "primary_judge_model_id"),
            audit_judge_model_id=_required_text(value.get("audit_judge_model_id"), "audit_judge_model_id"),
            director_model_id=director_model_id,
            codegen_model_id=codegen_model_id,
            director_provider_kind=_required_text(
                value.get("director_provider_kind") or "external_openai_compatible",
                "director_provider_kind",
            ),
            codegen_provider_kind=_required_text(
                value.get("codegen_provider_kind") or "codex_exec_local",
                "codegen_provider_kind",
            ),
            primary_blind_payload_version=str(value.get("primary_blind_payload_version") or "primary-blind-v1"),
            score_policy_version=str(value.get("score_policy_version") or "scoring-v7-pending-calibration"),
            sampling_policy_version=str(value.get("sampling_policy_version") or "event-aligned-uniform-v1"),
            paired_statistics_version=str(value.get("paired_statistics_version") or DEFAULT_PAIRED_STATISTICS_VERSION),
            bootstrap_seed=int(value.get("bootstrap_seed", 20260829)),
            bootstrap_iterations=int(value.get("bootstrap_iterations", 2000)),
            alpha=float(value.get("alpha", 0.05)),
            train_min_gain=float(value.get("train_min_gain", 1.0)),
            dev_noninferiority_margin=float(value.get("dev_noninferiority_margin", -1.0)),
            secondary_noninferiority_margin=float(value.get("secondary_noninferiority_margin", -1.0)),
            confidence_threshold=float(value.get("confidence_threshold", 0.6)),
            evidence_completeness_threshold=float(value.get("evidence_completeness_threshold", 1.0)),
        )
        config.validate()
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "FormalEvaluatorConfig":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("formal evaluator config must be a JSON object")
        return cls.from_mapping(payload)

    def validate(self) -> None:
        identities = (
            self.generator_model_id,
            self.primary_judge_model_id,
            self.audit_judge_model_id,
        )
        if len(set(identities)) != len(identities):
            raise ValueError("generator and judge model snapshots must be distinct")
        if not self.director_model_id.strip() or not self.codegen_model_id.strip():
            raise ValueError("director and codegen model identities are required")
        if not self.director_provider_kind.strip() or not self.codegen_provider_kind.strip():
            raise ValueError("director and codegen provider identities are required")
        if not self.primary_blind_payload_version.strip():
            raise ValueError("primary blind payload version is required")
        if not self.score_policy_version.strip():
            raise ValueError("score policy version is required")
        if not self.sampling_policy_version.strip():
            raise ValueError("sampling policy version is required")
        if not self.paired_statistics_version.strip():
            raise ValueError("paired statistics version is required")
        if self.bootstrap_iterations < 1:
            raise ValueError("bootstrap_iterations must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if self.train_min_gain < 0.0:
            raise ValueError("train_min_gain must not be negative")
        for field, value in (
            ("confidence_threshold", self.confidence_threshold),
            ("evidence_completeness_threshold", self.evidence_completeness_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be between zero and one")

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_version": FORMAL_CONFIG_VERSION,
            "generator_model_id": self.generator_model_id,
            "director_model_id": self.director_model_id,
            "codegen_model_id": self.codegen_model_id,
            "director_provider_kind": self.director_provider_kind,
            "codegen_provider_kind": self.codegen_provider_kind,
            "primary_judge_model_id": self.primary_judge_model_id,
            "audit_judge_model_id": self.audit_judge_model_id,
            "primary_blind_payload_version": self.primary_blind_payload_version,
            "score_policy_version": self.score_policy_version,
            "sampling_policy_version": self.sampling_policy_version,
            "paired_statistics_version": self.paired_statistics_version,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_iterations": self.bootstrap_iterations,
            "alpha": self.alpha,
            "train_min_gain": self.train_min_gain,
            "dev_noninferiority_margin": self.dev_noninferiority_margin,
            "secondary_noninferiority_margin": self.secondary_noninferiority_margin,
            "confidence_threshold": self.confidence_threshold,
            "evidence_completeness_threshold": self.evidence_completeness_threshold,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["DEFAULT_PAIRED_STATISTICS_VERSION", "FORMAL_CONFIG_VERSION", "FormalEvaluatorConfig"]
