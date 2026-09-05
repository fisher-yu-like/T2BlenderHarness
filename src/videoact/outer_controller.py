"""Formal, fail-closed controller for the Harness outer transition.

The controller is deliberately small and evidence-first.  It accepts only
train evidence, delegates normalization/aggregation/attribution to the T04-
T06 components, and treats a Coding Agent response as a *proposal* until a
real, hashed, in-scope source diff is verified.  It never edits files itself;
the later Patch Executor owns that mutation boundary.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field

from .cross_owner import validate_cross_owner_proposal
from .evolution import FailureSummary, aggregate_failures
from .failure_attribution import CounterfactualAttributor
from .failure_extractor import FailureEvidence, FailureExtractor
from .meta_harness import MetaHarnessOptimizer, PatchProposal
from .split_access import SplitAccessPolicy


OUTER_CONTROLLER_SCHEMA_VERSION = "outer-controller-v1"
MAX_OUTER_ATTEMPTS = 5
ACTIONABLE_ACTIONS = {"patch", "no_patch", "blocked", "accepted", "rejected", "rollback"}

# These components are experiment inputs or frozen evidence boundaries.  A
# controller may diagnose them, but it can never return a patch that changes
# them.  The path checks are repeated here rather than relying on proposal
# metadata supplied by a caller.
_FORBIDDEN_PATH_PARTS = {
    "dataset",
    "datasets",
    "evaluator",
    "observer",
    "observers",
    "test",
    "tests",
}
_FORBIDDEN_PATH_PATTERNS = (
    re.compile(r"(^|/)test_[^/]+", re.IGNORECASE),
    re.compile(r"(^|/)observer(?:_|/|$)", re.IGNORECASE),
    re.compile(r"(^|/)evaluator(?:_|/|$)", re.IGNORECASE),
)


class OuterTransition(BaseModel):
    """One append-only decision made by the formal outer controller."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = OUTER_CONTROLLER_SCHEMA_VERSION
    transition_id: str = Field(min_length=1)
    transition_index: int = Field(ge=0)
    attempt: int = Field(ge=1, le=MAX_OUTER_ATTEMPTS)
    action: Literal["patch", "no_patch", "blocked", "accepted", "rejected", "rollback"]
    status: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    source_split: Literal["train"] = "train"
    owner: str | None = None
    root_cause_id: str | None = None
    source_case_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    proposal: dict[str, Any] | None = None
    attribution: dict[str, Any] | None = None
    diff_sha256: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    patch_manifest: dict[str, Any] | None = None
    append_only: bool = True
    timestamp: str = Field(min_length=1)


class OuterControllerResult(BaseModel):
    """Machine-readable result for one explicitly requested attempt."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = OUTER_CONTROLLER_SCHEMA_VERSION
    status: Literal["patch", "no_patch", "blocked", "accepted", "rejected", "rollback"]
    action: Literal["patch", "no_patch", "blocked", "accepted", "rejected", "rollback"]
    reason: str = Field(min_length=1)
    attempt: int = Field(ge=1, le=MAX_OUTER_ATTEMPTS)
    attempt_count: int = Field(ge=1, le=MAX_OUTER_ATTEMPTS)
    max_attempts: int = Field(ge=1, le=MAX_OUTER_ATTEMPTS)
    train_case_ids: list[str] = Field(default_factory=list)
    normalized_records: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    attributions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    proposal: dict[str, Any] | None = None
    diff_sha256: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    patch_manifest: dict[str, Any] | None = None
    transition: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(child) for child in value]
    return value


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        _dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_hash(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _normalise_path(value: str) -> str:
    path = str(value).replace("\\", "/").strip()
    if not path:
        raise ValueError("patch path cannot be empty")
    if path.startswith("/") or re.match(r"^[A-Za-z]:/", path):
        raise ValueError(f"patch path must be workspace-relative: {value}")
    parts = [part for part in path.split("/") if part not in {"", "."}]
    if ".." in parts:
        raise ValueError(f"patch path cannot escape the workspace: {value}")
    return "/".join(parts)


def _path_is_forbidden(path: str) -> bool:
    normalized = _normalise_path(path)
    parts = normalized.casefold().split("/")
    if any(part in _FORBIDDEN_PATH_PARTS for part in parts[:-1]):
        return True
    basename = parts[-1] if parts else normalized.casefold()
    if basename in _FORBIDDEN_PATH_PARTS:
        return True
    return any(pattern.search(normalized) for pattern in _FORBIDDEN_PATH_PATTERNS)


def _forbidden_metadata(value: Any) -> bool:
    """Detect explicit attempts to edit a frozen component in a manifest."""

    if not isinstance(value, Mapping):
        return False
    for key in (
        "modified_components",
        "modified_owners",
        "frozen_components_changed",
        "policy_changes",
        "target_components",
    ):
        values = value.get(key)
        if isinstance(values, str):
            values = [values]
        if isinstance(values, (list, tuple, set)):
            text = " ".join(str(item).casefold() for item in values)
            if any(token in text for token in ("evaluator", "dataset", "observer", "test policy", "test_policy")):
                return True
    return False


def _case_id(record: Mapping[str, Any]) -> str:
    case_id = str(record.get("case_id") or "").strip()
    if not case_id:
        manifest = record.get("manifest")
        if isinstance(manifest, Mapping):
            case_id = str(manifest.get("case_id") or "").strip()
    if not case_id:
        raise ValueError("train evidence record requires case_id")
    return case_id


def _split(record: Mapping[str, Any]) -> str:
    manifest = record.get("manifest")
    manifest_split = manifest.get("split") if isinstance(manifest, Mapping) else None
    return str(record.get("split") or manifest_split or "train").casefold()


def _finding_payload(raw: Any, *, case_id: str) -> dict[str, Any]:
    """Normalize a legacy Finding without inventing a score or root cause."""

    row = _dump(raw)
    if not isinstance(row, Mapping):
        raise ValueError(f"finding for {case_id} must be an object")
    data = {str(key): value for key, value in row.items()}
    # Some callers attach case_id to each Finding for convenience.  The
    # legacy Finding contract is case-local and intentionally does not carry
    # that routing field.
    data.pop("case_id", None)
    failure_id = str(data.get("failure_id") or "failure").strip()
    category = str(data.get("category") or "deterministic").strip()
    message = str(data.get("message") or failure_id).strip()
    owner = str(data.get("owner") or data.get("owner_candidate") or "").strip()
    if not owner:
        raise ValueError(f"actionable finding for {case_id} requires one owner")
    from evaluator.findings import normalize_root_cause_id

    data["failure_id"] = failure_id
    data["category"] = category
    data["message"] = message
    data["owner"] = owner
    data["root_cause_id"] = normalize_root_cause_id(
        data.get("root_cause_id"), failure_id=failure_id, category=category, message=message
    )
    data.setdefault("severity", "error")
    data.setdefault("evidence", [])
    data.setdefault("repair_route", "candidate_recovery")
    # Validate the exact legacy boundary now, so aggregation cannot receive a
    # partially-shaped object and silently discard it later.
    from .contracts import Finding

    return Finding.model_validate(data).model_dump(mode="json")


class OuterTransitionController:
    """Connect train evidence to one bounded, auditable outer transition."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        transition_log: str | Path | None = None,
        max_attempts: int = MAX_OUTER_ATTEMPTS,
        coding_agent: Callable[[dict[str, Any]], Any] | None = None,
        extractor: FailureExtractor | None = None,
        attributor: CounterfactualAttributor | None = None,
        optimizer: MetaHarnessOptimizer | None = None,
        split_policy: SplitAccessPolicy | Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= MAX_OUTER_ATTEMPTS:
            raise ValueError("max_attempts must be an integer between 1 and 5")
        self.max_attempts = int(max_attempts)
        self.coding_agent = coding_agent
        self.split_policy = (
            split_policy
            if isinstance(split_policy, SplitAccessPolicy)
            else SplitAccessPolicy.model_validate(split_policy)
            if split_policy is not None
            else None
        )
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if transition_log is None and self.output_dir is not None:
            transition_log = self.output_dir / "outer_transitions.jsonl"
        self.transition_log = Path(transition_log) if transition_log is not None else None
        if self.transition_log is not None:
            self.transition_log.parent.mkdir(parents=True, exist_ok=True)
        self.extractor = extractor or FailureExtractor()
        self.attributor = attributor or CounterfactualAttributor()
        self.optimizer = optimizer or MetaHarnessOptimizer(
            output_dir=self.output_dir or Path.cwd() / ".outer-controller"
        )
        # Rehydrate append-only history before accepting a new decision.  A
        # resumed host process must continue transition IDs instead of
        # silently reusing transition-0001 and making the audit ambiguous.
        self.transitions: list[dict[str, Any]] = []
        if self.transition_log is not None and self.transition_log.is_file():
            try:
                for line in self.transition_log.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict) and row.get("event") == "outer_transition":
                        self.transitions.append(row)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"outer transition log is missing or invalid: {self.transition_log}") from exc

    @property
    def transition_count(self) -> int:
        return len(self.transitions)

    def _append_transition(self, transition: OuterTransition) -> dict[str, Any]:
        row = transition.model_dump(mode="json")
        row["event"] = "outer_transition"
        row["append_only"] = True
        self.transitions.append(row)
        if self.transition_log is not None:
            with self.transition_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return row

    def append_decision(
        self,
        *,
        action: str,
        status: str,
        reason: str,
        attempt: int = 1,
        owner: str | None = None,
        root_cause_id: str | None = None,
        source_case_ids: Sequence[str] = (),
        proposal: Mapping[str, Any] | None = None,
        attribution: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append an externally completed accepted/rejected/blocked decision."""

        if action not in ACTIONABLE_ACTIONS:
            raise ValueError(f"unsupported outer transition action: {action}")
        if not 1 <= int(attempt) <= self.max_attempts:
            raise ValueError("outer transition attempt exceeds the configured budget")
        transition = OuterTransition(
            transition_id=f"transition-{self.transition_count + 1:04d}",
            transition_index=self.transition_count,
            attempt=int(attempt),
            action=action,  # type: ignore[arg-type]
            status=str(status),
            reason=str(reason),
            owner=str(owner) if owner else None,
            root_cause_id=str(root_cause_id) if root_cause_id else None,
            source_case_ids=list(dict.fromkeys(str(item) for item in source_case_ids if str(item).strip())),
            proposal=dict(proposal) if isinstance(proposal, Mapping) else None,
            attribution=dict(attribution) if isinstance(attribution, Mapping) else None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        return self._append_transition(transition)

    def collect_train_evidence(
        self,
        source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
        *,
        forbidden_case_ids: set[str] | None = None,
        forbidden_prompt_hashes: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Collect and normalize only train records or train run directories."""

        forbidden_ids = {str(item) for item in (forbidden_case_ids or set()) if str(item).strip()}
        forbidden_hashes = {str(item) for item in (forbidden_prompt_hashes or set()) if str(item).strip()}
        if isinstance(source, (str, Path)):
            root = Path(source)
            if not root.exists():
                raise ValueError(f"train evidence source is missing: {root}")
            run_dirs = [root] if (root / "run_manifest.json").is_file() else sorted(
                path for path in root.rglob("run_manifest.json") if path.parent.is_dir()
            )
            records: list[dict[str, Any]] = []
            for manifest_path in run_dirs:
                evidence = self.extractor.extract_run(
                    manifest_path.parent,
                    expected_split="train",
                    forbidden_case_ids=forbidden_ids,
                    forbidden_prompt_hashes=forbidden_hashes,
                    split_policy=self.split_policy,
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                case_id = _case_id({"manifest": manifest})
                records.append(
                    {
                        "case_id": case_id,
                        "split": "train",
                        "manifest": manifest,
                        "failure_evidence": [item.model_dump(mode="json") for item in evidence],
                        "findings": [item.to_finding() for item in evidence if item.actionable],
                        "abstentions": [item.model_dump(mode="json") for item in evidence if item.abstain],
                    }
                )
            return self._normalize_records(
                records,
                forbidden_case_ids=forbidden_ids,
                forbidden_prompt_hashes=forbidden_hashes,
            )
        if isinstance(source, Mapping):
            raw_records: Sequence[Mapping[str, Any]] = [source]
        else:
            raw_records = source
        return self._normalize_records(
            list(raw_records),
            forbidden_case_ids=forbidden_ids,
            forbidden_prompt_hashes=forbidden_hashes,
        )

    def _normalize_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        forbidden_case_ids: set[str],
        forbidden_prompt_hashes: set[str],
    ) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        if self.split_policy is not None:
            self.split_policy.validate_records(list(records))
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                raise ValueError("train evidence records must be objects")
            record = dict(raw_record)
            case_id = _case_id(record)
            split = _split(record)
            if split != "train":
                raise ValueError(f"outer controller is train-only; received split={split}")
            forbidden_keys = {
                "dev",
                "dev_records",
                "dev_metrics",
                "test",
                "test_records",
                "test_metrics",
                "test_scores",
            }
            if forbidden_keys.intersection(record):
                raise ValueError("dev/test context cannot enter the outer controller")
            serialized = json.dumps(_dump(record), ensure_ascii=False, sort_keys=True, default=str)
            leaked_ids = sorted(item for item in forbidden_case_ids if item in serialized)
            if leaked_ids:
                raise ValueError(f"forbidden case IDs entered outer controller: {leaked_ids}")
            leaked_hashes = sorted(item for item in forbidden_prompt_hashes if item in serialized)
            if leaked_hashes:
                raise ValueError(f"forbidden prompt hashes entered outer controller: {leaked_hashes}")

            evidence_rows = record.get("failure_evidence")
            if evidence_rows is None:
                evidence_rows = record.get("normalized_evidence")
            if isinstance(evidence_rows, Mapping):
                evidence_rows = [evidence_rows]
            evidence_rows = list(evidence_rows) if isinstance(evidence_rows, (list, tuple)) else []
            findings: list[dict[str, Any]] = []
            normalized_evidence: list[dict[str, Any]] = []
            for evidence in evidence_rows:
                if isinstance(evidence, FailureEvidence):
                    item = evidence
                elif isinstance(evidence, Mapping) and evidence.get("source_kind"):
                    item = FailureEvidence.model_validate(evidence)
                else:
                    continue
                if item.split != "train" or item.case_id != case_id:
                    raise ValueError("failure evidence case/split does not match its train record")
                normalized_evidence.append(item.model_dump(mode="json"))
                if item.actionable:
                    findings.append(item.to_finding())
            raw_findings = record.get("findings", [])
            if isinstance(raw_findings, Mapping):
                raw_findings = [raw_findings]
            for finding in raw_findings if isinstance(raw_findings, (list, tuple)) else []:
                normalized_finding = _finding_payload(finding, case_id=case_id)
                if normalized_finding not in findings:
                    findings.append(normalized_finding)
            normalized.append(
                {
                    **record,
                    "case_id": case_id,
                    "split": "train",
                    "findings": findings,
                    "failure_evidence": normalized_evidence,
                }
            )
        return normalized

    def attribute_train_failures(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        forbidden_case_ids: set[str] | None = None,
        supplied: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Attribute each root cause without looking at dev/test evidence."""

        forbidden = {str(item) for item in (forbidden_case_ids or set())}
        if self.split_policy is not None:
            forbidden.update(self.split_policy.forbidden_case_ids)
        if supplied is not None:
            values: list[Any]
            if isinstance(supplied, Mapping):
                values = list(supplied.values())
            else:
                values = list(supplied)
            result: dict[str, dict[str, Any]] = {}
            for value in values:
                row = _dump(value)
                if not isinstance(row, Mapping):
                    raise ValueError("supplied attribution must be an object")
                if str(row.get("split") or "train").casefold() != "train":
                    raise ValueError("supplied attribution is train-only")
                if str(row.get("case_id") or "") in forbidden:
                    raise ValueError("forbidden case entered attribution")
                root = str(row.get("root_cause_id") or row.get("failure_id") or "").strip()
                if not root:
                    raise ValueError("supplied attribution requires root_cause_id")
                result[root] = dict(row)
            return result

        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            case_id = _case_id(record)
            for evidence in record.get("failure_evidence", []) or []:
                row = _dump(evidence)
                if not isinstance(row, Mapping) or not bool(row.get("actionable")):
                    continue
                root = str(row.get("root_cause_id") or row.get("failure_id") or "").strip()
                if root:
                    grouped.setdefault(root, []).append(dict(row))
            # Legacy Finding records have already crossed the evidence gate.
            # A confidence of 1.0 here means “the Finding validator accepted
            # the owner”, not a model score or an invented visual metric.
            if not record.get("failure_evidence"):
                for finding in record.get("findings", []) or []:
                    row = _dump(finding)
                    if not isinstance(row, Mapping):
                        continue
                    root = str(row.get("root_cause_id") or row.get("failure_id") or "").strip()
                    if root:
                        grouped.setdefault(root, []).append(
                            {
                                **dict(row),
                                "case_id": case_id,
                                "split": "train",
                                "failure_id": str(row.get("failure_id") or root),
                                "owner_candidate": str(row.get("owner") or ""),
                                "owner_confidence": 1.0,
                                "evidence_refs": list(row.get("evidence") or []),
                                "evidence_complete": bool(row.get("evidence")),
                            }
                        )
        result = {}
        for root, rows in grouped.items():
            candidates: list[dict[str, Any]] = []
            for row in rows:
                counterfactuals = row.get("counterfactuals") or row.get("counterfactual_runs") or []
                attribution = self.attributor.attribute(
                    {
                        **row,
                        "case_id": str(row.get("case_id") or ""),
                        "split": "train",
                        "root_cause_id": root,
                        "evidence_refs": row.get("evidence_refs") or row.get("evidence") or [],
                    },
                    counterfactuals=counterfactuals,
                    forbidden_case_ids=forbidden,
                )
                candidates.append(attribution.model_dump(mode="json"))
            if not candidates:
                continue
            owners = {
                str(item.get("owner_candidate"))
                for item in candidates
                if not item.get("abstain") and item.get("owner_candidate")
            }
            if len(owners) == 1 and all(not item.get("abstain") for item in candidates):
                selected = next(item for item in candidates if item.get("owner_candidate"))
                selected["case_ids"] = sorted({str(item.get("case_id")) for item in candidates})
                result[root] = selected
            else:
                result[root] = {
                    "schema_version": "failure-attribution-v1",
                    "root_cause_id": root,
                    "split": "train",
                    "owner_candidate": None,
                    "owner_confidence": 0.0,
                    "abstain": True,
                    "reason": "owner_attribution_not_consistent_across_train_cases",
                    "case_ids": sorted({str(item.get("case_id")) for item in candidates}),
                    "candidate_attributions": candidates,
                }
        return result

    @staticmethod
    def _repeated_groups(summary: FailureSummary) -> list[Any]:
        return [
            group for group in summary.groups if len(set(group.affected_case_ids)) >= 2
        ]

    def _verify_patch_response(
        self,
        response: Any,
        proposal: PatchProposal,
    ) -> dict[str, Any]:
        if isinstance(response, str):
            payload: dict[str, Any] = {"diff": response}
        else:
            dumped = _dump(response)
            if not isinstance(dumped, Mapping):
                raise ValueError("Coding Agent response must contain a patch diff and manifest")
            payload = {str(key): value for key, value in dumped.items()}
        manifest = payload.get("patch_manifest") or payload.get("manifest")
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_file():
                raise ValueError(f"patch manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            manifest = {}
        manifest = {str(key): value for key, value in manifest.items()}
        diff_value = payload.get("diff")
        if diff_value is None:
            diff_value = payload.get("patch_diff", payload.get("source_diff"))
        if not isinstance(diff_value, (str, bytes)) or not (diff_value if isinstance(diff_value, str) else diff_value.decode("utf-8", "ignore")).strip():
            raise ValueError("proposal has no real source diff")
        diff_sha256 = _text_hash(diff_value)
        declared_hash = (
            payload.get("diff_sha256")
            or payload.get("source_diff_sha256")
            or manifest.get("diff_sha256")
            or manifest.get("source_diff_sha256")
            or manifest.get("diff_hash")
        )
        if not isinstance(declared_hash, str) or declared_hash.casefold() != diff_sha256:
            raise ValueError("source diff hash is missing or does not match the diff")
        changed_raw = (
            payload.get("changed_files")
            or payload.get("files")
            or manifest.get("changed_files")
            or manifest.get("files")
        )
        if not isinstance(changed_raw, (list, tuple)) or not changed_raw:
            raise ValueError("patch manifest must declare changed_files")
        changed_files = list(dict.fromkeys(_normalise_path(item) for item in changed_raw))
        allowed_files = list(dict.fromkeys(_normalise_path(item) for item in proposal.affected_files))
        if not allowed_files:
            raise ValueError("proposal must declare affected_files")
        forbidden = [path for path in [*allowed_files, *changed_files] if _path_is_forbidden(path)]
        if forbidden or _forbidden_metadata(manifest):
            raise ValueError("controller cannot modify evaluator, dataset, test, or observer policy")
        outside = sorted(set(changed_files) - set(allowed_files))
        if outside:
            raise ValueError(f"patch changed files outside proposal scope: {outside}")
        if manifest.get("source_diff_present") is False:
            raise ValueError("patch manifest marks source_diff_present=false")
        # Cross-owner validation happens before this method, but keep a stable
        # serialized manifest that includes the verified hash for downstream
        # Patch Executor/Impact Proof consumers.
        verified_manifest = {
            **manifest,
            "source_diff_present": True,
            "diff_sha256": diff_sha256,
            "changed_files": changed_files,
            "source_split": "train",
        }
        return {
            "diff_sha256": diff_sha256,
            "changed_files": changed_files,
            "patch_manifest": verified_manifest,
            "diff": diff_value.decode("utf-8") if isinstance(diff_value, bytes) else diff_value,
        }

    def transition(
        self,
        attempt_number: int,
        reports: Sequence[Mapping[str, Any]],
        *,
        coding_agent: Callable[[dict[str, Any]], Any] | None = None,
    ) -> dict[str, Any]:
        """Adapt a real batch report to ``run_bounded_outer_attempts``.

        ``run_outer_attempt`` exposes the immutable train run root in
        ``reports[-1]["splits"]["train"]["run_root"]``.  Resolving that
        path here means the controller reads the same per-case artifacts as
        the standalone CLI, rather than making a second scoring path.
        """

        if not reports:
            raise ValueError("outer controller transition requires an attempt report")
        latest = reports[-1]
        train_report: Any = latest.get("train_records")
        if train_report is None:
            splits = latest.get("splits")
            if isinstance(splits, Mapping):
                train_report = splits.get("train")
        source: Any = None
        if isinstance(train_report, Mapping):
            source = (
                train_report.get("run_root")
                or train_report.get("records")
                or train_report.get("case_records")
            )
        elif isinstance(train_report, (list, tuple, str, Path)):
            source = train_report
        if source is None:
            reason = "formal outer controller could not locate the train evidence root"
            self.append_decision(action="blocked", status="blocked", reason=reason, attempt=attempt_number)
            return {"action": "stop", "status": "blocked", "reason": reason}
        result = self.run(
            source,
            coding_agent=coding_agent,
            attempt=attempt_number,
        )
        if result["action"] == "patch":
            return {
                "action": "patch",
                "status": "patch_ready",
                "reason": result["reason"],
                "proposal": result["proposal"],
                "source_split": "train",
                "diff_sha256": result["diff_sha256"],
                "changed_files": result["changed_files"],
                "patch_manifest": result["patch_manifest"],
            }
        return {
            "action": "stop",
            "status": result["status"],
            "reason": result["reason"],
        }

    @staticmethod
    def _invoke_coding_agent(coding_agent: Callable[[dict[str, Any]], Any], proposal: PatchProposal) -> Any:
        """Invoke once; retrying a provider is an untracked hidden attempt."""

        if not callable(coding_agent):
            raise ValueError("coding_agent must be callable")
        payload = proposal.model_dump(mode="json")
        # A callable accepting one positional argument is the public contract.
        # Inspect only for a useful error message; never call it twice.
        try:
            signature = inspect.signature(coding_agent)
            if not any(
                parameter.kind in {parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD}
                for parameter in signature.parameters.values()
            ):
                raise ValueError("coding_agent must accept the proposal payload")
        except (TypeError, ValueError):
            # Builtins/opaque callables are still invoked once below.
            pass
        return coding_agent(payload)

    def run(
        self,
        train_source: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path,
        *,
        coding_agent: Callable[[dict[str, Any]], Any] | None = None,
        forbidden_case_ids: set[str] | None = None,
        forbidden_prompt_hashes: set[str] | None = None,
        attributions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        experiment_contract: Mapping[str, Any] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        """Process exactly one explicit outer attempt.

        No hidden retry is performed here.  Callers that want attempts 2--5
        must call :meth:`run` again with the next attempt number (or use
        :meth:`run_attempts`, whose callback makes that budget explicit).
        """

        if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= self.max_attempts:
            raise ValueError(f"outer attempt must be between 1 and {self.max_attempts}")
        records = self.collect_train_evidence(
            train_source,
            forbidden_case_ids=forbidden_case_ids,
            forbidden_prompt_hashes=forbidden_prompt_hashes,
        )
        case_ids = sorted({_case_id(record) for record in records})
        summary = self.optimizer.summarize_train_records(
            records, forbidden_case_ids=forbidden_case_ids, split_policy=self.split_policy
        )
        repeated = self._repeated_groups(summary)
        attribution_map = self.attribute_train_failures(
            records,
            forbidden_case_ids=forbidden_case_ids,
            supplied=attributions,
        )
        summary_payload = summary.model_dump(mode="json")
        if not repeated:
            reason = "no repeated actionable train root cause; no Harness patch is justified"
            transition = self.append_decision(
                action="no_patch",
                status="no_patch",
                reason=reason,
                attempt=attempt,
                source_case_ids=case_ids,
            )
            result = OuterControllerResult(
                status="no_patch",
                action="no_patch",
                reason=reason,
                attempt=attempt,
                attempt_count=1,
                max_attempts=self.max_attempts,
                train_case_ids=case_ids,
                normalized_records=[dict(record) for record in records],
                summary=summary_payload,
                attributions=attribution_map,
                transition=transition,
            )
            return result.as_dict()

        try:
            proposal = self.optimizer.propose(
                records,
                forbidden_case_ids=forbidden_case_ids,
                attributions=attribution_map,
            )
        except ValueError as exc:
            reason = f"proposal blocked: {exc}"
            transition = self.append_decision(
                action="blocked",
                status="blocked",
                reason=reason,
                attempt=attempt,
                source_case_ids=case_ids,
            )
            return OuterControllerResult(
                status="blocked",
                action="blocked",
                reason=reason,
                attempt=attempt,
                attempt_count=1,
                max_attempts=self.max_attempts,
                train_case_ids=case_ids,
                normalized_records=[dict(record) for record in records],
                summary=summary_payload,
                attributions=attribution_map,
                transition=transition,
            ).as_dict()

        # T09 is intentionally one-owner.  T12 may rank a mixed-owner backlog,
        # but it must not silently turn that future policy into this gate.
        audit = validate_cross_owner_proposal(proposal.model_dump(mode="json"))
        owners = list(dict.fromkeys(str(item) for item in audit.get("owners", []) if str(item).strip()))
        if len(owners) != 1 or proposal.cross_owner_exception:
            reason = "proposal blocked: formal outer transition requires exactly one Harness owner"
            transition = self.append_decision(
                action="blocked",
                status="blocked",
                reason=reason,
                attempt=attempt,
                owner=proposal.owner,
                root_cause_id=proposal.root_cause_id,
                source_case_ids=proposal.source_case_ids,
                proposal=proposal.model_dump(mode="json"),
            )
            return OuterControllerResult(
                status="blocked",
                action="blocked",
                reason=reason,
                attempt=attempt,
                attempt_count=1,
                max_attempts=self.max_attempts,
                train_case_ids=case_ids,
                normalized_records=[dict(record) for record in records],
                summary=summary_payload,
                attributions=attribution_map,
                proposal=proposal.model_dump(mode="json"),
                transition=transition,
            ).as_dict()

        selected_attribution = attribution_map.get(proposal.root_cause_id)
        if not selected_attribution or selected_attribution.get("abstain") is True or not selected_attribution.get("owner_candidate"):
            reason = "proposal blocked: first-divergence owner attribution is unavailable"
            transition = self.append_decision(
                action="blocked",
                status="blocked",
                reason=reason,
                attempt=attempt,
                owner=proposal.owner,
                root_cause_id=proposal.root_cause_id,
                source_case_ids=proposal.source_case_ids,
                proposal=proposal.model_dump(mode="json"),
                attribution=selected_attribution,
            )
            return OuterControllerResult(
                status="blocked",
                action="blocked",
                reason=reason,
                attempt=attempt,
                attempt_count=1,
                max_attempts=self.max_attempts,
                train_case_ids=case_ids,
                normalized_records=[dict(record) for record in records],
                summary=summary_payload,
                attributions=attribution_map,
                proposal=proposal.model_dump(mode="json"),
                transition=transition,
            ).as_dict()

        effective_coding_agent = coding_agent or self.coding_agent
        if effective_coding_agent is None:
            reason = "proposal ready but no Coding Agent is attached; formal training is blocked"
            transition = self.append_decision(
                action="blocked",
                status="blocked",
                reason=reason,
                attempt=attempt,
                owner=proposal.owner,
                root_cause_id=proposal.root_cause_id,
                source_case_ids=proposal.source_case_ids,
                proposal=proposal.model_dump(mode="json"),
                attribution=selected_attribution,
            )
            return OuterControllerResult(
                status="blocked",
                action="blocked",
                reason=reason,
                attempt=attempt,
                attempt_count=1,
                max_attempts=self.max_attempts,
                train_case_ids=case_ids,
                normalized_records=[dict(record) for record in records],
                summary=summary_payload,
                attributions=attribution_map,
                proposal=proposal.model_dump(mode="json"),
                transition=transition,
            ).as_dict()

        if experiment_contract is not None:
            from .real_artifacts import validate_proposal_split_access

            validate_proposal_split_access(proposal.model_dump(mode="json"), experiment_contract)
        try:
            response = self._invoke_coding_agent(effective_coding_agent, proposal)
            verified = self._verify_patch_response(response, proposal)
        except Exception as exc:
            reason = f"patch blocked: {type(exc).__name__}: {exc}"
            transition = self.append_decision(
                action="blocked",
                status="blocked",
                reason=reason,
                attempt=attempt,
                owner=proposal.owner,
                root_cause_id=proposal.root_cause_id,
                source_case_ids=proposal.source_case_ids,
                proposal=proposal.model_dump(mode="json"),
                attribution=selected_attribution,
            )
            return OuterControllerResult(
                status="blocked",
                action="blocked",
                reason=reason,
                attempt=attempt,
                attempt_count=1,
                max_attempts=self.max_attempts,
                train_case_ids=case_ids,
                normalized_records=[dict(record) for record in records],
                summary=summary_payload,
                attributions=attribution_map,
                proposal=proposal.model_dump(mode="json"),
                transition=transition,
            ).as_dict()

        transition_model = OuterTransition(
            transition_id=f"transition-{self.transition_count + 1:04d}",
            transition_index=self.transition_count,
            attempt=attempt,
            action="patch",
            status="patch_ready",
            reason="verified train-derived one-owner source diff and patch manifest",
            owner=proposal.owner,
            root_cause_id=proposal.root_cause_id,
            source_case_ids=proposal.source_case_ids,
            evidence_count=sum(
                len(record.get("failure_evidence") or record.get("findings", []))
                for record in records
            ),
            proposal=proposal.model_dump(mode="json"),
            attribution=selected_attribution,
            diff_sha256=verified["diff_sha256"],
            changed_files=verified["changed_files"],
            patch_manifest=verified["patch_manifest"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        transition = self._append_transition(transition_model)
        return OuterControllerResult(
            status="patch",
            action="patch",
            reason="verified train-derived one-owner source diff and patch manifest",
            attempt=attempt,
            attempt_count=1,
            max_attempts=self.max_attempts,
            train_case_ids=case_ids,
            normalized_records=[dict(record) for record in records],
            summary=summary_payload,
            attributions=attribution_map,
            proposal=proposal.model_dump(mode="json"),
            diff_sha256=verified["diff_sha256"],
            changed_files=verified["changed_files"],
            patch_manifest=verified["patch_manifest"],
            transition=transition,
        ).as_dict()

    def run_attempts(
        self,
        collect_attempt: Callable[[int], Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path],
        *,
        coding_agent: Callable[[dict[str, Any]], Any] | None = None,
        forbidden_case_ids: set[str] | None = None,
        forbidden_prompt_hashes: set[str] | None = None,
    ) -> dict[str, Any]:
        """Run an explicit attempt provider with no more than five calls."""

        results: list[dict[str, Any]] = []
        for attempt in range(1, self.max_attempts + 1):
            source = collect_attempt(attempt)
            result = self.run(
                source,
                coding_agent=coding_agent,
                forbidden_case_ids=forbidden_case_ids,
                forbidden_prompt_hashes=forbidden_prompt_hashes,
                attempt=attempt,
            )
            results.append(result)
            if result["action"] != "patch":
                break
            # A patch transition is handed to the Patch Executor/Host.  A
            # second attempt requires a new explicit run_attempts invocation,
            # so a successful patch can never hide another render or edit.
            break
        return {
            "schema_version": OUTER_CONTROLLER_SCHEMA_VERSION,
            "status": results[-1]["status"] if results else "blocked",
            "action": results[-1]["action"] if results else "blocked",
            "attempt_count": len(results),
            "max_attempts": self.max_attempts,
            "results": results,
            "transitions": list(self.transitions),
        }


__all__ = [
    "ACTIONABLE_ACTIONS",
    "MAX_OUTER_ATTEMPTS",
    "OUTER_CONTROLLER_SCHEMA_VERSION",
    "OuterControllerResult",
    "OuterTransition",
    "OuterTransitionController",
]
