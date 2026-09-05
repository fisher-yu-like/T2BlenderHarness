"""Five-layer obligation matrix and first-divergence attribution.

The matrix is a diagnostic contract, not a replacement score.  Each row is
checked in order: planned, implemented, executed, visible, judged.  Once the
first failed/unavailable/disputed layer is found, downstream failures remain
context only and do not create additional primary root causes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence, Set
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .obligations import ObligationCompilation, ObligationRecord


OBLIGATION_MATRIX_SCHEMA_VERSION = "obligation-matrix-v1"
MATRIX_STAGES = ("planned", "implemented", "executed", "visible", "judged")
MatrixStage = Literal["planned", "implemented", "executed", "visible", "judged"]
MatrixStatus = Literal[
    "pending",
    "satisfied",
    "failed",
    "unavailable",
    "disagreement",
    "not_applicable",
]
RowStatus = Literal["passed", "failed", "uncertain", "not_applicable"]


class MatrixStageEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stage: MatrixStage
    status: MatrixStatus
    evidence_refs: list[str] = Field(default_factory=list)
    expected: Any = None
    observed: Any = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class ObligationMatrixRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    obligation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    required: bool
    applicable: bool
    planned_status: MatrixStatus
    implemented_status: MatrixStatus
    executed_status: MatrixStatus
    visible_status: MatrixStatus
    judged_status: MatrixStatus
    stage_evidence: dict[str, MatrixStageEvidence] = Field(default_factory=dict)
    status: RowStatus
    first_divergence_stage: MatrixStage | None = None
    first_divergence_status: MatrixStatus | None = None
    primary_root_cause_id: str | None = None
    owner_candidate: str | None = None
    owner_confidence: float = Field(default=0.0, ge=0, le=1)
    evidence_refs: list[str] = Field(default_factory=list)
    primary_failure: bool = False

    @property
    def stage_statuses(self) -> dict[str, MatrixStatus]:
        return {
            "planned": self.planned_status,
            "implemented": self.implemented_status,
            "executed": self.executed_status,
            "visible": self.visible_status,
            "judged": self.judged_status,
        }


class ObligationMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = OBLIGATION_MATRIX_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    rows: list[ObligationMatrixRow] = Field(default_factory=list)
    obligation_ids: list[str] = Field(default_factory=list)
    primary_failures: list[dict[str, Any]] = Field(default_factory=list)
    na_dimensions: list[str] = Field(default_factory=list)
    fingerprint: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def validate_rows(self) -> "ObligationMatrix":
        row_ids = [row.obligation_id for row in self.rows]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("obligation matrix IDs must be unique")
        ids = list(self.obligation_ids or row_ids)
        if set(ids) != set(row_ids) or len(ids) != len(set(ids)):
            raise ValueError("obligation matrix obligation_ids must match rows")
        object.__setattr__(self, "obligation_ids", ids)
        return self


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(child) for child in value]
    return value


def _hash(value: Any) -> str:
    encoded = json.dumps(_dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    dumped = _dump(value)
    return dumped if isinstance(dumped, dict) else {}


def _records(value: ObligationCompilation | Sequence[ObligationRecord]) -> list[ObligationRecord]:
    if isinstance(value, ObligationCompilation):
        return list(value.obligations)
    return list(value)


def _parse_stage_value(value: Any, *, default: MatrixStatus, obligation_id: str) -> tuple[MatrixStatus, dict[str, Any]]:
    if value is None:
        return default, {}
    if isinstance(value, bool):
        return ("satisfied" if value else "failed"), {}
    if isinstance(value, str):
        normalized = value.casefold().strip()
        aliases = {
            "pass": "satisfied",
            "passed": "satisfied",
            "ok": "satisfied",
            "fail": "failed",
            "error": "failed",
            "uncertain": "unavailable",
            "na": "not_applicable",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"pending", "satisfied", "failed", "unavailable", "disagreement", "not_applicable"}:
            raise ValueError(f"invalid matrix status for {obligation_id}: {value}")
        return normalized, {}
    if isinstance(value, Mapping):
        raw = dict(value)
        status, _ = _parse_stage_value(raw.get("status"), default=default, obligation_id=obligation_id)
        return status, raw
    return ("satisfied" if obligation_id in value else "failed"), {}


def _stage_from_input(
    source: Any,
    *,
    record: ObligationRecord,
    field_name: str,
) -> tuple[MatrixStatus, dict[str, Any]]:
    default = getattr(record, field_name)
    if source is None:
        return default, {"status": default}
    if isinstance(source, Mapping):
        if record.obligation_id not in source:
            return "failed", {}
        return _parse_stage_value(source[record.obligation_id], default=default, obligation_id=record.obligation_id)
    if isinstance(source, (Set, list, tuple)):
        return _parse_stage_value(record.obligation_id in source, default=default, obligation_id=record.obligation_id)
    return _parse_stage_value(source, default=default, obligation_id=record.obligation_id)


def _root_and_owner(kind: str, stage: MatrixStage, stage_status: MatrixStatus) -> tuple[str, str, float]:
    normalized = kind.casefold()
    if stage == "planned":
        return "obligation_planning", "director_event_scheduler", 0.9
    if stage == "implemented":
        return "obligation_implementation", "blender_code_agent", 0.85
    if stage == "executed":
        owner = "interaction_library" if any(token in normalized for token in ("ownership", "transfer", "contact", "support")) else "blender_executor"
        return "obligation_execution", owner, 0.8
    if stage == "visible":
        return "obligation_visibility", "director_camera", 0.85
    if stage == "judged":
        return ("judge_disagreement" if stage_status == "disagreement" else "judge_insensitivity"), "evaluator", 0.6
    return "obligation_failure", "unassigned", 0.0


def _first_divergence(statuses: Mapping[str, MatrixStatus]) -> tuple[MatrixStage | None, MatrixStatus | None]:
    for stage in MATRIX_STAGES:
        status = statuses.get(stage, "pending")
        if status in {"failed", "unavailable", "disagreement"}:
            return stage, status  # type: ignore[return-value]
    for stage in MATRIX_STAGES:
        if statuses.get(stage, "pending") == "pending":
            return stage, "pending"  # type: ignore[return-value]
    return None, None


def first_divergence(row: ObligationMatrixRow) -> str | None:
    """Return the first divergent stage for a matrix row."""

    return row.first_divergence_stage


def build_obligation_matrix(
    obligations: ObligationCompilation | Sequence[ObligationRecord],
    *,
    planned: Any = None,
    implemented: Any = None,
    executed: Any = None,
    visible: Any = None,
    judged: Any = None,
) -> ObligationMatrix:
    """Build a deterministic five-layer matrix from stage evidence maps."""

    sources = {
        "planned": planned,
        "implemented": implemented,
        "executed": executed,
        "visible": visible,
        "judged": judged,
    }
    rows: list[ObligationMatrixRow] = []
    primary_failures: list[dict[str, Any]] = []
    na_dimensions: list[str] = []
    records = _records(obligations)
    case_ids = {record.case_id for record in records}
    if len(case_ids) > 1:
        raise ValueError(f"obligation matrix accepts one case at a time: {sorted(case_ids)}")
    case_id = next(iter(case_ids), "unknown-case")
    for record in records:
        if not record.applicable:
            statuses = {stage: "not_applicable" for stage in MATRIX_STAGES}
            na_dimensions.append(record.kind)
        else:
            statuses = {}
            evidence_by_stage: dict[str, dict[str, Any]] = {}
            for stage, source in sources.items():
                status, raw = _stage_from_input(source, record=record, field_name=f"{stage}_status")
                statuses[stage] = status
                evidence_by_stage[stage] = raw
        if not record.applicable:
            evidence_by_stage = {stage: {"status": "not_applicable"} for stage in MATRIX_STAGES}
        first_stage, first_status = _first_divergence(statuses)
        failure_status = first_status in {"failed", "unavailable", "disagreement"}
        pending_status = first_status == "pending"
        if not record.applicable:
            row_status: RowStatus = "not_applicable"
        elif failure_status:
            row_status = "failed"
        elif pending_status:
            row_status = "uncertain"
        else:
            row_status = "passed"
        root = owner = None
        confidence = 0.0
        refs: list[str] = []
        if first_stage is not None and failure_status:
            root, owner, confidence = _root_and_owner(record.kind, first_stage, first_status)  # type: ignore[arg-type]
            refs = _strings(evidence_by_stage.get(first_stage, {}).get("evidence_refs"))
        stage_evidence: dict[str, MatrixStageEvidence] = {}
        for stage in MATRIX_STAGES:
            raw = evidence_by_stage.get(stage, {})
            stage_evidence[stage] = MatrixStageEvidence(
                stage=stage,  # type: ignore[arg-type]
                status=statuses[stage],  # type: ignore[arg-type]
                evidence_refs=_strings(raw.get("evidence_refs") or raw.get("evidence")),
                expected=raw.get("expected"),
                observed=raw.get("observed"),
                confidence=(float(raw["confidence"]) if raw.get("confidence") is not None else None),
            )
        primary = bool(record.required and record.applicable and failure_status)
        row = ObligationMatrixRow(
            obligation_id=record.obligation_id,
            case_id=record.case_id,
            kind=str(record.kind),
            required=record.required,
            applicable=record.applicable,
            planned_status=statuses["planned"],
            implemented_status=statuses["implemented"],
            executed_status=statuses["executed"],
            visible_status=statuses["visible"],
            judged_status=statuses["judged"],
            stage_evidence=stage_evidence,
            status=row_status,
            first_divergence_stage=first_stage,
            first_divergence_status=first_status,
            primary_root_cause_id=root,
            owner_candidate=owner,
            owner_confidence=confidence,
            evidence_refs=refs,
            primary_failure=primary,
        )
        rows.append(row)
        if primary:
            primary_failures.append(
                {
                    "failure_id": f"obligation_failed:{record.obligation_id}",
                    "obligation_id": record.obligation_id,
                    "root_cause_id": root,
                    "first_divergence_stage": first_stage,
                    "owner_candidate": owner,
                    "owner_confidence": confidence,
                    "severity": "hard" if record.required else "error",
                    "evidence_refs": refs,
                    "expected": record.expected,
                    "observed": stage_evidence.get(first_stage or "planned", MatrixStageEvidence(stage="planned", status="pending")).observed,
                }
            )
    payload = {
        "schema_version": OBLIGATION_MATRIX_SCHEMA_VERSION,
        "case_id": case_id,
        "rows": [row.model_dump(mode="json") for row in rows],
        "obligation_ids": [row.obligation_id for row in rows],
        "primary_failures": primary_failures,
        "na_dimensions": sorted(set(na_dimensions)),
    }
    return ObligationMatrix(
        case_id=case_id,
        rows=rows,
        obligation_ids=[row.obligation_id for row in rows],
        primary_failures=primary_failures,
        na_dimensions=sorted(set(na_dimensions)),
        fingerprint=_hash(payload),
    )


def validate_obligation_matrix(
    matrix: ObligationMatrix,
    *,
    expected_ids: Sequence[str] | None = None,
) -> ObligationMatrix:
    """Fail closed if a required matrix row was deleted or duplicated."""

    actual = list(matrix.obligation_ids)
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate obligation in matrix")
    row_ids = [row.obligation_id for row in matrix.rows]
    if actual != row_ids and set(actual) != set(row_ids):
        missing_rows = [identifier for identifier in actual if identifier not in set(row_ids)]
        raise ValueError(f"missing obligation in matrix: {missing_rows}")
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("duplicate obligation row in matrix")
    expected = list(expected_ids or actual)
    missing = [identifier for identifier in expected if identifier not in set(row_ids)]
    if missing:
        raise ValueError(f"missing obligation in matrix: {missing}")
    return matrix


__all__ = [
    "MATRIX_STAGES",
    "OBLIGATION_MATRIX_SCHEMA_VERSION",
    "MatrixStageEvidence",
    "ObligationMatrix",
    "ObligationMatrixRow",
    "build_obligation_matrix",
    "first_divergence",
    "validate_obligation_matrix",
]
