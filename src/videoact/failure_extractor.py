"""Normalize all train-run failure evidence before Harness evolution.

The extractor is deliberately separate from scoring.  A high aggregate score
does not erase a repeated visual/semantic bottleneck, while unavailable or
incomplete evidence can never become an actionable patch request.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluator.findings import finding_from_failure_evidence, normalize_root_cause_id


FAILURE_EVIDENCE_SCHEMA_VERSION = "failure-evidence-v1"
ACTIONABLE_CONFIDENCE_THRESHOLD = 0.6


class FailureEvidence(BaseModel):
    """One normalized, potentially patchable failure observation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = FAILURE_EVIDENCE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    split: Literal["train"]
    failure_id: str = Field(min_length=1)
    root_cause_id: str = Field(min_length=1)
    first_divergence_stage: str = Field(min_length=1)
    owner_candidate: str | None = None
    owner_confidence: float = Field(default=0.0, ge=0, le=1)
    severity: Literal["info", "warning", "error", "hard", "semantic_hard"]
    category: str = Field(min_length=1)
    message: str = Field(min_length=1)
    applicable: bool = True
    evidence_complete: bool = False
    expected: Any = None
    observed: Any = None
    affected_obligation_ids: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    source_kind: str = Field(min_length=1)
    source_path: str | None = None
    actionable: bool = False
    abstain: bool = True
    abstain_reason: str | None = None
    repair_route: str = "candidate_recovery"

    @model_validator(mode="after")
    def validate_actionability(self) -> "FailureEvidence":
        refs = list(dict.fromkeys(str(item) for item in self.evidence_refs if str(item).strip()))
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "affected_obligation_ids", list(dict.fromkeys(self.affected_obligation_ids)))
        can_act = (
            self.applicable
            and self.evidence_complete
            and bool(refs)
            and bool(self.owner_candidate)
            and self.owner_confidence >= ACTIONABLE_CONFIDENCE_THRESHOLD
            and not self.abstain
        )
        object.__setattr__(self, "actionable", bool(self.actionable and can_act))
        if not self.actionable:
            object.__setattr__(self, "abstain", True)
            if self.abstain_reason is None:
                reason = "evidence_incomplete" if not self.evidence_complete or not refs else "not_patchable"
                object.__setattr__(self, "abstain_reason", reason)
        return self

    def to_finding(self) -> dict[str, object]:
        """Return a legacy Finding payload only when this record is actionable."""

        return finding_from_failure_evidence(self)


class FailureExtractionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = FAILURE_EVIDENCE_SCHEMA_VERSION
    case_id: str = Field(min_length=1)
    split: Literal["train"]
    status: Literal["no_failure", "actionable", "abstain", "mixed"]
    evidence: list[FailureEvidence] = Field(default_factory=list)
    actionable_count: int = Field(ge=0)
    abstain_count: int = Field(ge=0)

    def to_patch_findings(self) -> list[dict[str, object]]:
        return [item.to_finding() for item in self.evidence if item.actionable]


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): child for key, child in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {}
    return {}


def _items(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _status_failure(value: Any) -> bool:
    return str(value or "").casefold() in {
        "fail",
        "failed",
        "failure",
        "error",
        "timeout",
        "timed_out",
        "incomplete",
        "unavailable",
        "needs_human_review",
        "codegen_failed",
        "director_failed",
        "coverage_failed",
    }


def _stage_for(*, source_kind: str, category: str, owner: str | None = None) -> str:
    source = source_kind.casefold()
    category_text = category.casefold()
    if "disagreement" in source or "judge" in source:
        return "judged"
    if "visual" in source:
        return "visible"
    if "artifact" in source or "observer" in source or "provenance" in source or "telemetry" in category_text:
        return "runtime_execution"
    if "codegen" in source:
        return "implemented"
    if "director" in source or "plan" in source:
        return "planned"
    if "camera" in category_text or "visibility" in category_text:
        return "visible"
    if owner and any(token in owner.casefold() for token in ("director", "planner", "parser")):
        return "planned"
    return "runtime_execution"


def _owner_for(category: str, source_kind: str, explicit: Any = None) -> str | None:
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    text = f"{category} {source_kind}".casefold()
    if "camera" in text or "visibility" in text:
        return "director_camera"
    if "trajectory" in text or "motion" in text:
        return "director_trajectory"
    if "semantic" in text or "event" in text:
        return "director_event_scheduler"
    if "codegen" in text or "source" in text:
        return "blender_code_agent"
    if "artifact" in text or "render" in text:
        return "proxy_renderer"
    if "observer" in text or "telemetry" in text or "runtime" in text:
        return "blender_executor"
    if "provenance" in text or "fingerprint" in text:
        return "blender_executor"
    return None


def _repair_route(owner: str | None, explicit: Any = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit)
    return {
        "director_camera": "camera_repair",
        "director_trajectory": "trajectory_repair",
        "director_event_scheduler": "scene_contract_repair",
        "blender_code_agent": "candidate_recovery",
        "proxy_renderer": "runtime_repair",
        "blender_executor": "runtime_repair",
    }.get(owner or "", "candidate_recovery")


def _obligation_ids(payload: Mapping[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    values: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    for key in ("obligation_ids",):
        values.extend(_strings(payload.get(key)))
    compilation = payload.get("obligations") or payload.get("obligation_compilation") or {}
    if isinstance(compilation, Mapping):
        values.extend(_strings(compilation.get("obligation_ids")))
        for raw in _items(compilation.get("obligations")):
            row = _as_dict(raw)
            identifier = str(row.get("obligation_id") or "").strip()
            if identifier:
                values.append(identifier)
                rows[identifier] = row
    manifest = _as_dict(payload.get("manifest"))
    values.extend(_strings(manifest.get("obligation_ids")))
    return list(dict.fromkeys(values)), rows


def _select_obligation_ids(all_ids: Sequence[str], rows: Mapping[str, Mapping[str, Any]], category: str, kind: str) -> list[str]:
    text = f"{category} {kind}".casefold()
    if not all_ids:
        return []
    wanted: set[str] = set()
    if "camera" in text or "visibility" in text:
        wanted = {"camera_coverage", "camera_visibility", "camera_motion", "camera_innovation"}
    elif "trajectory" in text or "motion" in text:
        wanted = {"trajectory"}
    elif "semantic" in text or "event" in text:
        wanted = {"event", "event_order", "event_timing", "participant", "target"}
    elif "ownership" in text or "handoff" in text or "interaction" in text:
        wanted = {"ownership_transition", "transfer_window", "contact_support"}
    selected = [identifier for identifier in all_ids if str(rows.get(identifier, {}).get("kind", "")) in wanted]
    return selected or list(all_ids)


class FailureExtractor:
    """Extract normalized evidence from one or more train run artifacts."""

    def __init__(self, *, visual_failure_threshold: float = 60.0, confidence_threshold: float = ACTIONABLE_CONFIDENCE_THRESHOLD) -> None:
        if not 0 <= visual_failure_threshold <= 100:
            raise ValueError("visual_failure_threshold must be between 0 and 100")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.visual_failure_threshold = float(visual_failure_threshold)
        self.confidence_threshold = float(confidence_threshold)

    def _load_run_dir(self, run_dir: str | Path) -> dict[str, Any]:
        root = Path(run_dir)
        if not root.is_dir():
            raise ValueError(f"run directory is missing: {root}")
        payload: dict[str, Any] = {"run_dir": str(root)}
        names = {
            "run_manifest": "run_manifest.json",
            "manifest": "run_manifest.json",
            "preparation_failure": "preparation_failure.json",
            "director_failure": "director_failure.json",
            "codegen_failure": "codegen_failure.json",
            "coverage_report": "coverage_report.json",
            "artifact_report": "artifact_report.json",
            "observer_report": "observer_report.json",
            "deterministic_report": "deterministic_report.json",
            "vlm_report": "vlm_report.json",
            "visual_primary": "visual_primary.json",
            "judge_disagreement": "judge_disagreement.json",
            "provenance_report": "provenance_report.json",
            "latency_report": "latency_report.json",
            "obligations": "obligations.json",
            "obligation_matrix": "obligation_matrix.json",
        }
        for key, name in names.items():
            path = root / name
            if not path.is_file():
                continue
            loaded = _json_load(path)
            if key == "manifest":
                payload["manifest"] = loaded
            elif key == "run_manifest":
                payload["manifest"] = loaded
            else:
                payload[key] = loaded
        return payload

    def _validate_scope(
        self,
        payload: Mapping[str, Any],
        *,
        expected_split: str | None,
        enforce_train_only: bool,
        forbidden_case_ids: set[str] | None,
        forbidden_prompt_hashes: set[str] | None,
    ) -> tuple[str, str]:
        manifest = _as_dict(payload.get("manifest"))
        case_id = str(payload.get("case_id") or manifest.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("failure extraction requires case_id")
        actual_split = str(payload.get("split") or manifest.get("split") or expected_split or "train").casefold()
        if expected_split is not None and actual_split != str(expected_split).casefold():
            raise ValueError(f"run split mismatch: expected {expected_split}, got {actual_split}")
        if enforce_train_only and actual_split != "train":
            raise ValueError(f"failure extractor is train-only; received split={actual_split}")
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        forbidden = {str(item) for item in (forbidden_case_ids or set()) if str(item).strip()}
        if forbidden & set(re.findall(r"[A-Za-z0-9_.:-]+", serialized)):
            raise ValueError(f"forbidden case IDs entered failure extraction: {sorted(forbidden & set(re.findall(r'[A-Za-z0-9_.:-]+', serialized)))}")
        prompt_hashes = {str(item) for item in (forbidden_prompt_hashes or set()) if str(item).strip()}
        leaked_hashes = {item for item in prompt_hashes if item in serialized}
        if leaked_hashes:
            raise ValueError(f"forbidden prompt hashes entered failure extraction: {sorted(leaked_hashes)}")
        if actual_split != "train":
            # Dev can be summarized for acceptance only through the explicit
            # non-train compatibility switch; it cannot be fed to proposals.
            if enforce_train_only:
                raise ValueError(f"failure extractor is train-only; received split={actual_split}")
        return case_id, actual_split

    def _make(
        self,
        *,
        case_id: str,
        source_kind: str,
        source_path: str,
        raw: Mapping[str, Any] | None = None,
        failure_id: str,
        category: str,
        message: str,
        owner: str | None = None,
        severity: str = "error",
        first_divergence_stage: str | None = None,
        expected: Any = None,
        observed: Any = None,
        evidence_refs: Sequence[str] = (),
        evidence_complete: bool | None = None,
        applicable: bool = True,
        affected_obligation_ids: Sequence[str] = (),
        owner_confidence: float | None = None,
        abstain: bool = False,
        abstain_reason: str | None = None,
        repair_route: str | None = None,
        all_obligation_ids: Sequence[str] = (),
        obligation_rows: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> FailureEvidence:
        item = dict(raw or {})
        category = str(item.get("category") or category)
        owner = _owner_for(category, source_kind, item.get("owner") or owner)
        root_cause = normalize_root_cause_id(
            item.get("root_cause_id"),
            failure_id=str(item.get("failure_id") or failure_id),
            category=category,
            message=str(item.get("message") or message),
        )
        refs = list(dict.fromkeys([*(_strings(item.get("evidence_refs"))), *(_strings(item.get("evidence"))), *(_strings(item.get("evidence_paths"))), *(str(ref) for ref in evidence_refs if str(ref).strip())]))
        if source_path and source_path not in refs:
            refs.append(source_path)
        explicit_complete = item.get("evidence_complete")
        complete = bool(explicit_complete) if explicit_complete is not None else (bool(refs) and bool(item.get("evidence") or item.get("evidence_refs") or evidence_complete))
        if evidence_complete is not None:
            complete = bool(evidence_complete)
        confidence = _number(item.get("owner_confidence", item.get("confidence", owner_confidence if owner_confidence is not None else 1.0)), 0.0)
        if not complete:
            confidence = 0.0
            abstain = True
            abstain_reason = abstain_reason or "evidence_incomplete"
        if confidence < self.confidence_threshold:
            abstain = True
            abstain_reason = abstain_reason or "low_confidence"
        if str(item.get("status") or "").casefold() in {"unavailable", "needs_human_review", "uncertain"}:
            abstain = True
            abstain_reason = abstain_reason or "evidence_unavailable"
        if severity not in {"info", "warning", "error", "hard", "semantic_hard"}:
            severity = "error"
        selected_obligations = list(affected_obligation_ids or _strings(item.get("affected_obligation_ids")))
        if not selected_obligations:
            selected_obligations = _select_obligation_ids(all_obligation_ids, obligation_rows or {}, category, source_kind)
        return FailureEvidence(
            case_id=case_id,
            split="train",
            failure_id=str(item.get("failure_id") or failure_id),
            root_cause_id=root_cause,
            first_divergence_stage=str(item.get("first_divergence_stage") or first_divergence_stage or _stage_for(source_kind=source_kind, category=category, owner=owner)),
            owner_candidate=owner,
            owner_confidence=max(0.0, min(1.0, confidence)),
            severity=severity,  # type: ignore[arg-type]
            category=category,
            message=str(item.get("message") or message),
            applicable=bool(item.get("applicable", applicable)),
            evidence_complete=complete,
            expected=item.get("expected", expected),
            observed=item.get("observed", observed),
            affected_obligation_ids=list(dict.fromkeys(str(value) for value in selected_obligations if str(value).strip())),
            evidence_refs=refs,
            source_kind=source_kind,
            source_path=source_path,
            actionable=not abstain,
            abstain=abstain,
            abstain_reason=abstain_reason,
            repair_route=str(item.get("repair_route") or repair_route or _repair_route(owner)),
        )

    def extract(
        self,
        payload: Mapping[str, Any] | str | Path,
        *,
        expected_split: str | None = None,
        split: str | None = None,
        enforce_train_only: bool = True,
        forbidden_case_ids: set[str] | None = None,
        forbidden_prompt_hashes: set[str] | None = None,
        split_policy: Any | None = None,
        experiment_contract: Any | None = None,
    ) -> list[FailureEvidence]:
        if isinstance(payload, (str, Path)):
            data = self._load_run_dir(payload)
        else:
            data = dict(payload)
        if split_policy is not None:
            validator = getattr(split_policy, "validate_payload", None)
            if not callable(validator):
                raise ValueError("split_policy must expose validate_payload")
            validator(data)
        if split is not None:
            expected_split = split
        contract = _as_dict(experiment_contract)
        contract_splits = contract.get("split_cases") if isinstance(contract, Mapping) else None
        contract_test = contract_splits.get("test", []) if isinstance(contract_splits, Mapping) else []
        contract_test_ids = {str(_as_dict(item).get("case_id")) for item in _items(contract_test)}
        contract_test_hashes = {str(_as_dict(item).get("prompt_hash")) for item in _items(contract_test)}
        case_id, actual_split = self._validate_scope(
            data,
            expected_split=expected_split,
            enforce_train_only=enforce_train_only,
            forbidden_case_ids=set(forbidden_case_ids or set()) | contract_test_ids,
            forbidden_prompt_hashes=set(forbidden_prompt_hashes or set()) | contract_test_hashes,
        )
        if actual_split != "train" and enforce_train_only:
            raise ValueError("failure extractor is train-only")
        all_obligation_ids, obligation_rows = _obligation_ids(data)
        evidence: list[FailureEvidence] = []

        def add(**kwargs: Any) -> None:
            evidence.append(
                self._make(
                    case_id=case_id,
                    all_obligation_ids=all_obligation_ids,
                    obligation_rows=obligation_rows,
                    **kwargs,
                )
            )

        # Preparation/codegen failures are real failures even before a video
        # exists.  A missing failure payload remains an abstention.
        for key, owner, category in (
            ("preparation_failure", "blender_executor", "preparation"),
            ("director_failure", "director_prompt_interpreter", "director_planning"),
            ("codegen_failure", "blender_code_agent", "codegen"),
            ("coverage_report", "blender_code_agent", "coverage"),
        ):
            raw = _as_dict(data.get(key))
            if not raw:
                continue
            status = raw.get("status") or raw.get("terminal_status")
            failed = key == "coverage_report" and status not in {None, "pass"} or key != "coverage_report" and (_status_failure(status) or bool(raw.get("reason")))
            if not failed:
                continue
            refs = _strings(raw.get("evidence")) or _strings(raw.get("failures"))
            add(
                source_kind=key,
                source_path=f"{key}.json",
                raw=raw,
                failure_id=str(raw.get("failure_id") or f"{key}_failed"),
                category=category,
                message=str(raw.get("reason") or raw.get("message") or f"{key} failed"),
                owner=owner,
                severity="hard",
                evidence_refs=refs,
                evidence_complete=True if raw.get("reason") or refs else False,
            )

        artifact = _as_dict(data.get("artifact_report"))
        if artifact and str(artifact.get("artifact_status") or "").casefold() != "complete":
            failures = _strings(artifact.get("hard_failures"))
            add(
                source_kind="artifact_report",
                source_path="artifact_report.json",
                raw=artifact,
                failure_id="incomplete_real_artifacts",
                category="artifact",
                message=f"real artifacts are incomplete: {failures or ['artifact status is not complete']}",
                owner="proxy_renderer",
                severity="hard",
                evidence_refs=failures,
                evidence_complete=bool(failures),
            )

        observer = _as_dict(data.get("observer_report"))
        manifest = _as_dict(data.get("manifest"))
        if observer:
            if observer.get("status") != "pass" or observer.get("trusted") is False:
                failures = _strings(observer.get("failures")) or _strings(observer.get("error"))
                add(
                    source_kind="observer_report",
                    source_path="observer_report.json",
                    raw=observer,
                    failure_id="trusted_observer_failed",
                    category="runtime_telemetry",
                    message=f"trusted observer evidence is invalid: {failures or ['observer did not pass']}",
                    owner="blender_executor",
                    severity="hard",
                    evidence_refs=["observer_report.json:failures"] if failures else [],
                    evidence_complete=bool(failures),
                )
            else:
                trusted_telemetry = _as_dict(observer.get("telemetry"))
                runtime_failures = _items(trusted_telemetry.get("runtime_failures"))
                for index, raw_failure in enumerate(runtime_failures):
                    row = _as_dict(raw_failure)
                    add(
                        source_kind="trusted_telemetry",
                        source_path=f"observer_report.json:runtime_failures[{index}]",
                        raw=row,
                        failure_id=str(row.get("failure_id") or f"trusted_runtime_failure_{index}"),
                        category=str(row.get("category") or "runtime_telemetry"),
                        message=str(row.get("message") or "trusted runtime observation failure"),
                        owner="blender_executor",
                        severity=str(row.get("severity") or "error"),
                        evidence_refs=_strings(row.get("evidence")),
                        evidence_complete=bool(row.get("evidence")),
                    )
        elif manifest.get("trusted_observer_required") is True:
            add(
                source_kind="observer_report",
                source_path="observer_report.json",
                failure_id="trusted_observer_unavailable",
                category="runtime_telemetry",
                message="trusted observer report is unavailable for a formal run",
                owner="blender_executor",
                severity="hard",
                evidence_complete=False,
                abstain=True,
                abstain_reason="evidence_unavailable",
            )

        # Obligation matrix rows are semantic evidence, but only explicit
        # failed/unavailable statuses create a failure record.
        matrix = _as_dict(data.get("obligation_matrix"))
        matrix_failures = _items(matrix.get("primary_failures"))
        for raw_failure in matrix_failures:
            row = _as_dict(raw_failure)
            identifier = str(row.get("obligation_id") or "obligation")
            root = str(row.get("root_cause_id") or "obligation_failure")
            stage = str(row.get("first_divergence_stage") or "runtime_execution")
            owner = str(row.get("owner_candidate") or "") or None
            category = (
                "camera_coverage"
                if "visibility" in root or owner == "director_camera"
                else "ownership"
                if "ownership" in root or owner == "interaction_library"
                else "semantic"
            )
            refs = _strings(row.get("evidence_refs"))
            add(
                source_kind="obligation_matrix",
                source_path=f"obligation_matrix.json:{identifier}",
                raw={
                    **row,
                    "failure_id": str(row.get("failure_id") or f"obligation_failed:{identifier}"),
                    "root_cause_id": root,
                    "category": category,
                    "owner": owner,
                    "message": str(row.get("message") or f"obligation {identifier} failed at {stage}"),
                    "first_divergence_stage": stage,
                },
                failure_id=str(row.get("failure_id") or f"obligation_failed:{identifier}"),
                category=category,
                message=str(row.get("message") or f"obligation {identifier} failed at {stage}"),
                owner=owner,
                severity=str(row.get("severity") or "hard"),
                first_divergence_stage=stage,
                expected=row.get("expected"),
                observed=row.get("observed"),
                evidence_refs=refs,
                evidence_complete=bool(refs or row.get("expected") is not None or row.get("observed") is not None),
                affected_obligation_ids=[identifier],
            )

        compilation = data.get("obligations") or data.get("obligation_compilation") or {}
        compilation_rows = _items(_as_dict(compilation).get("obligations"))
        for raw_row in compilation_rows if not matrix_failures else []:
            row = _as_dict(raw_row)
            if not row.get("required") or not row.get("applicable", True):
                continue
            statuses = (("planned", row.get("planned_status")), ("implemented", row.get("implemented_status")), ("executed", row.get("executed_status")), ("visible", row.get("visible_status")), ("judged", row.get("judged_status")))
            failed_stage, failed_status = next(((stage, status) for stage, status in statuses if status in {"failed", "unavailable"}), (None, None))
            if failed_stage is None:
                continue
            identifier = str(row.get("obligation_id") or "obligation")
            kind = str(row.get("kind") or "semantic")
            add(
                source_kind="obligation_matrix",
                source_path=f"obligations.json:{identifier}",
                raw=row,
                failure_id=f"obligation_failed:{identifier}",
                category=kind,
                message=f"obligation {identifier} failed at {failed_stage}",
                owner=None,
                severity="hard" if row.get("required") else "error",
                first_divergence_stage=failed_stage,
                expected=row.get("expected"),
                observed=row.get(f"{failed_stage}_observed", row.get("observed")),
                evidence_refs=_strings(row.get("evidence_refs")) or [f"obligations.json:{identifier}"],
                evidence_complete=True,
                affected_obligation_ids=[identifier],
            )

        deterministic = _as_dict(data.get("deterministic_report"))
        if deterministic:
            raw_findings = [
                *_items(deterministic.get("findings")),
                *_items(deterministic.get("director_findings")),
                *_items(deterministic.get("interaction_findings")),
            ]
            for raw_finding in raw_findings:
                row = _as_dict(raw_finding)
                add(
                    source_kind="deterministic_finding",
                    source_path="deterministic_report.json",
                    raw=row,
                    failure_id=str(row.get("failure_id") or "deterministic_failure"),
                    category=str(row.get("category") or "deterministic"),
                    message=str(row.get("message") or "deterministic evaluator finding"),
                    owner=str(row.get("owner") or "") or None,
                    severity=str(row.get("severity") or "error"),
                    evidence_refs=_strings(row.get("evidence")),
                    evidence_complete=bool(row.get("evidence")),
                )
            if not raw_findings and _status_failure(deterministic.get("terminal_status")):
                add(
                    source_kind="deterministic_report",
                    source_path="deterministic_report.json",
                    failure_id="deterministic_failure_without_evidence",
                    category="deterministic",
                    message="deterministic report failed without normalized finding evidence",
                    owner=None,
                    severity="hard",
                    evidence_complete=False,
                )

        vlm = _as_dict(data.get("vlm_report"))
        visual = _as_dict(vlm.get("visual_primary")) or _as_dict(data.get("visual_primary")) or vlm
        if visual:
            visual_status = str(visual.get("status") or vlm.get("status") or "unavailable").casefold()
            confidence = _number(visual.get("confidence", vlm.get("confidence")), 0.0)
            dimensions = {
                "camera_coverage": ("camera_coverage", "director_camera"),
                "camera_effectiveness": ("camera_coverage", "director_camera"),
                "camera_innovation": ("camera_motion", "director_camera"),
                "semantic_score": ("semantic", "director_event_scheduler"),
                "prompt_compliance": ("semantic", "director_event_scheduler"),
                "character_trajectory": ("trajectory", "director_trajectory"),
                "object_trajectory": ("trajectory", "director_trajectory"),
                "temporal_smoothness": ("trajectory", "director_trajectory"),
                "motion_naturalness": ("trajectory", "director_trajectory"),
                "event_timing": ("event_timing", "director_event_scheduler"),
                "visual_clarity": ("visual_quality", "blender_code_agent"),
                "appearance_detail": ("visual_quality", "blender_code_agent"),
                "physical_plausibility": ("physical_realism", "blender_code_agent"),
                "physical_realism": ("physical_realism", "blender_code_agent"),
                "spatial_consistency": ("spatial_consistency", "blender_code_agent"),
                "visual_presentation": ("visual_quality", "blender_code_agent"),
            }
            dimension_evidence = visual.get("dimension_evidence") or {}
            if visual_status != "scored" or confidence < self.confidence_threshold:
                add(
                    source_kind="visual_primary",
                    source_path="vlm_report.json",
                    raw=visual,
                    failure_id="visual_evidence_unavailable",
                    category="visual_review",
                    message=f"blind visual evidence is {visual_status} or below confidence threshold",
                    owner="director_camera",
                    severity="warning",
                    evidence_refs=_strings(vlm.get("sampled_frames")) or _strings(visual.get("visible_evidence")),
                    evidence_complete=False,
                    owner_confidence=confidence,
                    abstain=True,
                    abstain_reason="evidence_unavailable" if visual_status != "scored" else "low_confidence",
                )
            else:
                for dimension, (category, owner) in dimensions.items():
                    value = visual.get(dimension)
                    if value is None or _number(value, 100.0) >= self.visual_failure_threshold:
                        continue
                    row = _as_dict(dimension_evidence.get(dimension))
                    refs = _strings(row.get("evidence_refs"))
                    complete = bool(row) and _number(row.get("evidence_completeness"), 0.0) >= 1.0 and bool(refs)
                    add(
                        source_kind="visual_primary",
                        source_path="vlm_report.json",
                        raw={
                            "failure_id": f"visual_{dimension}_low",
                            "category": category,
                            "owner": owner,
                            "severity": "error",
                            "message": f"blind Judge dimension {dimension} is below the failure threshold",
                            "root_cause_id": category,
                            "confidence": confidence,
                        },
                        failure_id=f"visual_{dimension}_low",
                        category=category,
                        message=f"blind Judge dimension {dimension} is below the failure threshold",
                        owner=owner,
                        severity="error",
                        evidence_refs=refs,
                        evidence_complete=complete,
                        observed=value,
                        owner_confidence=confidence,
                    )

        disagreement = data.get("judge_disagreement")
        if disagreement is None:
            disagreement = vlm.get("judge_disagreement")
        disagreement_value = _number(disagreement, 0.0) if not isinstance(disagreement, Mapping) else _number(disagreement.get("magnitude", disagreement.get("score")), 1.0)
        if disagreement is not None and disagreement_value >= 0.4:
            add(
                source_kind="judge_disagreement",
                source_path="judge_disagreement.json" if data.get("judge_disagreement") is not None else "vlm_report.json",
                raw=_as_dict(disagreement),
                failure_id="judge_disagreement",
                category="judge_disagreement",
                message="blind Judge outputs disagree beyond the frozen tolerance",
                owner=None,
                severity="warning",
                evidence_refs=_strings(_as_dict(disagreement).get("evidence_refs")) or ["judge_disagreement.json"],
                evidence_complete=True,
                owner_confidence=0.0,
                abstain=True,
                abstain_reason="judge_disagreement",
            )

        for key, category, owner in (
            ("provenance_report", "provenance", "blender_executor"),
            ("latency_report", "latency", "blender_executor"),
        ):
            report = _as_dict(data.get(key))
            if not report:
                continue
            status = report.get("status")
            failed = _status_failure(status) or report.get("valid") is False or report.get("within_budget") is False
            if not failed:
                continue
            refs = _strings(report.get("evidence_refs")) or _strings(report.get("failures"))
            add(
                source_kind=key,
                source_path=f"{key}.json",
                raw=report,
                failure_id=str(report.get("failure_id") or f"{category}_failure"),
                category=category,
                message=str(report.get("message") or report.get("reason") or f"{category} report failed"),
                owner=owner,
                severity="error",
                evidence_refs=refs,
                evidence_complete=bool(refs),
            )
        return evidence

    def extract_report(self, payload: Mapping[str, Any] | str | Path, **kwargs: Any) -> FailureExtractionReport:
        evidence = self.extract(payload, **kwargs)
        case_id = evidence[0].case_id if evidence else self._case_id_from_payload(payload)
        split = "train"
        actionable = sum(item.actionable for item in evidence)
        abstain = len(evidence) - actionable
        status = "no_failure" if not evidence else "actionable" if actionable and not abstain else "abstain" if abstain and not actionable else "mixed"
        return FailureExtractionReport(
            case_id=case_id,
            split=split,
            status=status,  # type: ignore[arg-type]
            evidence=evidence,
            actionable_count=actionable,
            abstain_count=abstain,
        )

    def extract_run(self, run_dir: str | Path, **kwargs: Any) -> list[FailureEvidence]:
        return self.extract(run_dir, **kwargs)

    def extract_many(
        self,
        payloads: Sequence[Mapping[str, Any] | str | Path],
        **kwargs: Any,
    ) -> list[FailureEvidence]:
        return [
            item
            for payload in payloads
            for item in self.extract(payload, **kwargs)
        ]

    @staticmethod
    def _case_id_from_payload(payload: Mapping[str, Any] | str | Path) -> str:
        if isinstance(payload, (str, Path)):
            path = Path(payload)
            manifest = _json_load(path / "run_manifest.json") if path.is_dir() else {}
            return str(_as_dict(manifest).get("case_id") or path.name)
        return str(payload.get("case_id") or _as_dict(payload.get("manifest")).get("case_id") or "unknown-case")


def extract_failures(payload: Mapping[str, Any] | str | Path, **kwargs: Any) -> list[FailureEvidence]:
    """Functional API for callers that do not need an extractor instance."""

    return FailureExtractor().extract(payload, **kwargs)


__all__ = [
    "ACTIONABLE_CONFIDENCE_THRESHOLD",
    "FAILURE_EVIDENCE_SCHEMA_VERSION",
    "FailureEvidence",
    "FailureExtractionReport",
    "FailureExtractor",
    "extract_failures",
]
