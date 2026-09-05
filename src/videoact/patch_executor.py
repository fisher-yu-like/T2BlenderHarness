"""Restricted source Patch Executor for Harness self-evolution.

The executor is the only component in the closed loop that may mutate a
working tree.  It accepts a train-derived, one-owner proposal, snapshots the
entire relevant workspace, applies a caller-supplied patch operation, and
fails closed on scope, challenge, contract, or production checks.  Rejected
changes are restored byte-for-byte while their diff and failure evidence are
kept in append-only records.
"""

from __future__ import annotations

import difflib
import hashlib
import inspect
import json
import py_compile
import shutil
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field

from .cross_owner import validate_cross_owner_proposal
from .patch_attribution import (
    PATCH_SCOPE_VERSION,
    normalize_patch_path,
    validate_patch_paths,
)
from .patch_impact import PatchImpactProof, validate_patch_impact


PATCH_EXECUTOR_SCHEMA_VERSION = "patch-executor-v1"
_IGNORED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".outer-controller",
    ".patch-executor",
}


class PatchExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = PATCH_EXECUTOR_SCHEMA_VERSION
    status: Literal["accepted", "rejected", "blocked"]
    action: Literal["accepted", "rejected", "blocked"]
    edit_id: str = Field(min_length=1)
    owner: str | None = None
    source_split: Literal["train"] = "train"
    reason: str = Field(min_length=1)
    parent_hashes: dict[str, str] = Field(default_factory=dict)
    child_hashes: dict[str, str] = Field(default_factory=dict)
    changed_files: list[str] = Field(default_factory=list)
    diff_sha256: str | None = None
    patch_manifest: dict[str, Any] = Field(default_factory=dict)
    gates: dict[str, Any] = Field(default_factory=dict)
    impact_proof: dict[str, Any] | None = None
    restored: bool = False
    failure_evidence: list[str] = Field(default_factory=list)
    failure_record: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def __getitem__(self, key: str) -> Any:
        return self.model_dump(mode="json")[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.model_dump(mode="json").get(key, default)


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(child) for child in value]
    return value


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_text(value: str | bytes) -> str:
    return _hash_bytes(value if isinstance(value, bytes) else value.encode("utf-8"))


def _canonical_json(value: Any) -> str:
    return json.dumps(_dump(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple, set)):
        return list(dict.fromkeys(str(item) for item in value if str(item).strip()))
    return []


def _call_gate(callback: Callable[..., Any], context: dict[str, Any]) -> Any:
    """Call a gate once, supporting one-argument and zero-argument hooks."""

    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return callback(context)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }
    ]
    if positional or any(parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in signature.parameters.values()):
        return callback(context)
    return callback()


def _gate_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    row = _dump(value)
    if isinstance(row, Mapping):
        if row.get("status") is not None:
            return str(row.get("status")).casefold() in {"pass", "passed", "success", "accepted", "ok"}
        if row.get("passed") is not None:
            return bool(row.get("passed"))
        if row.get("accepted") is not None:
            return bool(row.get("accepted"))
    return False


class PatchExecutor:
    """Apply one constrained patch and restore it on any failed gate."""

    def __init__(
        self,
        *,
        repo_root: str | Path = ".",
        output_dir: str | Path | None = None,
        owner_challenge_runner: Callable[..., Any] | None = None,
        unit_test_runner: Callable[..., Any] | None = None,
        production_test_runner: Callable[..., Any] | None = None,
        blender_rerun_runner: Callable[..., Any] | None = None,
        require_impact_proof: bool = False,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        if not self.repo_root.is_dir():
            raise ValueError(f"repo root is missing: {self.repo_root}")
        self.output_dir = Path(output_dir).resolve() if output_dir is not None else None
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.owner_challenge_runner = owner_challenge_runner
        self.unit_test_runner = unit_test_runner
        self.production_test_runner = production_test_runner
        self.blender_rerun_runner = blender_rerun_runner
        self.require_impact_proof = bool(require_impact_proof)
        self.execution_count = 0
        self.records: list[dict[str, Any]] = []
        self._last_accepted_snapshot: dict[str, bytes] | None = None
        self._last_accepted_result: dict[str, Any] | None = None

    def _workspace_files(self) -> dict[str, Path]:
        files: dict[str, Path] = {}
        for path in self.repo_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.repo_root)
            if any(part in _IGNORED_DIRS for part in relative.parts):
                continue
            if self.output_dir is not None:
                try:
                    path.resolve().relative_to(self.output_dir)
                except ValueError:
                    pass
                else:
                    continue
            files[relative.as_posix()] = path
        return files

    def _snapshot(self) -> dict[str, bytes]:
        snapshot: dict[str, bytes] = {}
        for relative, path in self._workspace_files().items():
            try:
                snapshot[relative] = path.read_bytes()
            except OSError:
                continue
        return snapshot

    def _hash_snapshot(self, snapshot: Mapping[str, bytes]) -> dict[str, str]:
        return {path: _hash_bytes(value) for path, value in sorted(snapshot.items())}

    def _restore(self, snapshot: Mapping[str, bytes]) -> None:
        current = self._workspace_files()
        for relative, path in current.items():
            if relative not in snapshot:
                try:
                    path.unlink()
                except OSError:
                    pass
        for relative, content in snapshot.items():
            path = self.repo_root / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

    @staticmethod
    def _proposal_payload(proposal: Mapping[str, Any] | Any) -> dict[str, Any]:
        row = _dump(proposal)
        if not isinstance(row, Mapping):
            raise ValueError("patch proposal must be an object")
        return {str(key): value for key, value in row.items()}

    def _validate_proposal(
        self,
        proposal: Mapping[str, Any],
        *,
        forbidden_case_ids: set[str],
        train_evidence: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[str, list[str], list[str]]:
        if str(proposal.get("source_split") or "").casefold() != "train":
            raise ValueError("patch proposal must be sourced from train")
        audit = validate_cross_owner_proposal(proposal)
        owners = list(dict.fromkeys(str(item) for item in audit.get("owners", []) if str(item).strip()))
        if len(owners) != 1 or bool(proposal.get("cross_owner_exception")):
            raise ValueError("Patch Executor permits exactly one owner per patch")
        affected = proposal.get("affected_files", proposal.get("allowed_files", []))
        if not isinstance(affected, (list, tuple)):
            raise ValueError("proposal must declare affected_files")
        allowed = validate_patch_paths(list(affected))
        if not allowed:
            raise ValueError("proposal must declare at least one affected file")
        forbidden_paths = proposal.get("forbidden_paths", [])
        if forbidden_paths:
            validate_patch_paths(list(forbidden_paths))
        predicted_cases = set(_strings(proposal.get("source_case_ids")))
        predicted_cases.update(_strings(proposal.get("predicted_fixes")))
        leaked = predicted_cases & forbidden_case_ids
        if leaked:
            raise ValueError(f"forbidden test case IDs entered patch proposal: {sorted(leaked)}")
        if train_evidence is not None:
            for raw in train_evidence:
                row = _dump(raw)
                if not isinstance(row, Mapping):
                    raise ValueError("train evidence must contain objects")
                split = str(row.get("split") or "train").casefold()
                if split != "train":
                    raise ValueError("Patch Executor accepts train evidence only")
                if {"dev", "test", "dev_records", "test_records"}.intersection(row):
                    raise ValueError("dev/test context cannot enter Patch Executor")
        return owners[0], allowed, sorted(predicted_cases)

    def build_coding_context(
        self,
        proposal: Mapping[str, Any] | Any,
        train_evidence: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build the only context that may be passed to a Coding Agent."""

        row = self._proposal_payload(proposal)
        owner, allowed, _ = self._validate_proposal(
            row, forbidden_case_ids=set(), train_evidence=train_evidence
        )
        evidence = []
        for item in train_evidence:
            payload = _dump(item)
            if isinstance(payload, Mapping):
                evidence.append(dict(payload))
        return {
            "train_evidence": evidence,
            "owner_contract": {
                "owner": owner,
                "allowed_files": allowed,
                "frozen_components": ["dataset", "evaluator", "observer", "test", "Blender binary"],
                "scope_version": PATCH_SCOPE_VERSION,
            },
        }

    def _patch_payload(self, patch: Any) -> tuple[dict[str, Any], dict[str, str] | None]:
        row = _dump(patch)
        if isinstance(row, str):
            return {"diff": row}, None
        if not isinstance(row, Mapping):
            raise ValueError("patch response must be an object or diff string")
        payload = {str(key): value for key, value in row.items()}
        replacements = payload.get("file_contents") or payload.get("replacement_files")
        if replacements is None and isinstance(payload.get("files"), Mapping):
            replacements = payload.get("files")
        normalized_replacements: dict[str, str] | None = None
        if isinstance(replacements, Mapping):
            normalized_replacements = {
                normalize_patch_path(str(path)): str(content)
                for path, content in replacements.items()
            }
        return payload, normalized_replacements

    def _apply_patch(
        self,
        payload: Mapping[str, Any],
        replacements: Mapping[str, str] | None,
        *,
        apply_callback: Callable[..., Any] | None,
        context: dict[str, Any],
    ) -> None:
        if apply_callback is not None:
            outcome = _call_gate(apply_callback, {"patch": dict(payload), "context": context, "repo_root": str(self.repo_root)})
            if outcome is False or isinstance(outcome, Mapping) and outcome.get("status") in {"failed", "rejected", "blocked"}:
                raise ValueError("patch application callback rejected the patch")
            return
        if replacements is None:
            raise ValueError("Patch Executor requires apply_callback or file_contents")
        for relative, content in replacements.items():
            path = self.repo_root / Path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

    def _derive_diff(
        self,
        payload: Mapping[str, Any],
        before: Mapping[str, bytes],
        after: Mapping[str, bytes],
        changed_files: Sequence[str],
    ) -> str:
        raw = payload.get("diff") or payload.get("patch_diff") or payload.get("source_diff")
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        if isinstance(raw, str) and raw.strip():
            return raw
        chunks: list[str] = []
        for relative in changed_files:
            old = before.get(relative, b"").decode("utf-8", "replace").splitlines(keepends=True)
            new = after.get(relative, b"").decode("utf-8", "replace").splitlines(keepends=True)
            chunks.extend(
                difflib.unified_diff(
                    old,
                    new,
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        return "".join(chunks)

    def _record(
        self,
        result: PatchExecutionResult,
        *,
        diff: str,
        failure_payload: Mapping[str, Any] | None = None,
    ) -> PatchExecutionResult:
        row = result.model_dump(mode="json")
        row["event"] = "patch_execution"
        row["append_only"] = True
        self.records.append(row)
        if self.output_dir is not None:
            event_path = self.output_dir / "patch_executions.jsonl"
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            edit_id = result.edit_id
            evidence_path = self.output_dir / f"{edit_id}.failure_evidence.json"
            if evidence_path.exists():
                # The event log is append-only, and a resumed executor must
                # not overwrite the first rejection/rollback evidence for a
                # repeated proposal ID.
                suffix = 1
                while True:
                    candidate = self.output_dir / f"{edit_id}.failure_evidence.{suffix:04d}.json"
                    if not candidate.exists():
                        evidence_path = candidate
                        break
                    suffix += 1
            evidence_path.write_text(
                json.dumps(
                    {
                        "schema_version": PATCH_EXECUTOR_SCHEMA_VERSION,
                        "edit_id": edit_id,
                        "status": result.status,
                        "diff": diff,
                        "failure": _dump(failure_payload) if failure_payload is not None else None,
                        "result": row,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return result

    def execute(
        self,
        proposal: Mapping[str, Any] | Any,
        patch: Mapping[str, Any] | str | Any,
        *,
        train_evidence: Sequence[Mapping[str, Any]] | None = None,
        forbidden_case_ids: set[str] | None = None,
        apply_callback: Callable[..., Any] | None = None,
        owner_challenge_runner: Callable[..., Any] | None = None,
        unit_test_runner: Callable[..., Any] | None = None,
        production_test_runner: Callable[..., Any] | None = None,
        blender_rerun_runner: Callable[..., Any] | None = None,
        impact_proof: Mapping[str, Any] | PatchImpactProof | None = None,
        require_impact_proof: bool | None = None,
    ) -> dict[str, Any]:
        """Apply one patch, run ordered gates, and return an auditable result."""

        self.execution_count += 1
        edit_id = str(self._proposal_payload(proposal).get("proposal_id") or self._proposal_payload(proposal).get("root_cause_id") or f"patch-{self.execution_count:04d}")
        row = self._proposal_payload(proposal)
        forbidden = {str(item) for item in (forbidden_case_ids or set())}
        empty_snapshot: dict[str, bytes] = {}
        parent_hashes: dict[str, str] = {}
        diff = ""
        owner: str | None = None
        try:
            owner, allowed, predicted_cases = self._validate_proposal(
                row, forbidden_case_ids=forbidden, train_evidence=train_evidence
            )
            before = self._snapshot()
            empty_snapshot = before
            parent_hashes = {
                relative: _hash_bytes(before[relative]) if relative in before else "<missing>"
                for relative in allowed
            }
            context = self.build_coding_context(row, train_evidence or [])
            payload, replacements = self._patch_payload(patch)
            self._apply_patch(payload, replacements, apply_callback=apply_callback, context=context)
            after = self._snapshot()
            before_paths = set(before)
            after_paths = set(after)
            changed_files = sorted(
                path
                for path in before_paths | after_paths
                if before.get(path) != after.get(path)
            )
            diff = self._derive_diff(payload, before, after, changed_files)
            if not diff.strip():
                raise ValueError("patch produced no real source diff")
            declared_files = payload.get("changed_files") or payload.get("files")
            if isinstance(declared_files, Mapping):
                declared_files = list(declared_files)
            if declared_files is not None:
                if not isinstance(declared_files, (list, tuple)):
                    raise ValueError("patch changed_files must be a list")
                declared = sorted({normalize_patch_path(item) for item in declared_files})
                undeclared = sorted(set(changed_files) - set(allowed))
                if undeclared:
                    raise ValueError(f"patch changed undeclared files: {undeclared}")
                if declared != changed_files:
                    raise ValueError(f"actual changed files do not match patch manifest: {changed_files} != {declared}")
            if not changed_files:
                raise ValueError("patch produced no changed files")
            validate_patch_paths(changed_files)
            outside = sorted(set(changed_files) - set(allowed))
            if outside:
                raise ValueError(f"patch changed undeclared files: {outside}")
            child_hashes = {relative: _hash_bytes(after[relative]) for relative in changed_files if relative in after}
            diff_sha256 = _hash_text(diff)
            declared_hash = payload.get("diff_sha256") or payload.get("source_diff_sha256")
            if declared_hash is not None and str(declared_hash).casefold() != diff_sha256:
                raise ValueError("patch diff_sha256 does not match the applied diff")

            gate_context = {
                "edit_id": edit_id,
                "owner": owner,
                "proposal": row,
                "train_evidence": [_dump(item) for item in train_evidence or []],
                "changed_files": changed_files,
                "parent_hashes": parent_hashes,
                "child_hashes": child_hashes,
                "diff_sha256": diff_sha256,
                "repo_root": str(self.repo_root),
            }
            gates: dict[str, Any] = {}
            challenge = owner_challenge_runner or self.owner_challenge_runner
            unit = unit_test_runner or self.unit_test_runner
            production = production_test_runner or self.production_test_runner
            blender = blender_rerun_runner or self.blender_rerun_runner
            if challenge is None:
                raise ValueError("owner challenge gate is not configured")
            if unit is None:
                raise ValueError("owner/unit test gate is not configured")
            challenge_result = _call_gate(challenge, gate_context)
            gates["owner_challenge"] = _dump(challenge_result)
            if not _gate_passed(challenge_result):
                raise ValueError("owner challenge gate failed")
            unit_result = _call_gate(unit, gate_context)
            gates["unit_tests"] = _dump(unit_result)
            if not _gate_passed(unit_result):
                raise ValueError("owner/unit test gate failed")
            # Every changed Python module must at least compile even when a
            # caller supplies a narrow unit hook.
            for relative in changed_files:
                if relative.endswith(".py"):
                    py_compile.compile(str(self.repo_root / Path(relative)), doraise=True)
            if production is not None:
                production_result = _call_gate(production, gate_context)
                gates["production_contract"] = _dump(production_result)
                if not _gate_passed(production_result):
                    raise ValueError("production module/contract gate failed")
            else:
                gates["production_contract"] = {"status": "pass", "method": "py_compile"}
            if blender is not None:
                blender_result = _call_gate(blender, gate_context)
                gates["blender_rerun"] = _dump(blender_result)
                if not _gate_passed(blender_result):
                    raise ValueError("Blender rerun gate failed")
            supplied_impact = impact_proof or payload.get("impact_proof")
            require_impact = self.require_impact_proof if require_impact_proof is None else bool(require_impact_proof)
            impact_model: PatchImpactProof | None = None
            if supplied_impact is not None:
                impact_model = validate_patch_impact(supplied_impact)
            elif require_impact:
                raise ValueError("accepted patch requires a complete Patch Impact Proof")
            manifest = {
                "schema_version": PATCH_EXECUTOR_SCHEMA_VERSION,
                "scope_version": PATCH_SCOPE_VERSION,
                "edit_id": edit_id,
                "owner": owner,
                "source_split": "train",
                "affected_files": allowed,
                "changed_files": changed_files,
                "source_case_ids": predicted_cases,
                "parent_hashes": parent_hashes,
                "child_hashes": child_hashes,
                "diff_sha256": diff_sha256,
                "source_diff_present": True,
                "gates": gates,
                "impact_proof": impact_model.model_dump(mode="json") if impact_model is not None else None,
            }
            result = PatchExecutionResult(
                status="accepted",
                action="accepted",
                edit_id=edit_id,
                owner=owner,
                reason="patch applied and owner/unit/production gates passed",
                parent_hashes=parent_hashes,
                child_hashes=child_hashes,
                changed_files=changed_files,
                diff_sha256=diff_sha256,
                patch_manifest=manifest,
                gates=gates,
                impact_proof=impact_model.model_dump(mode="json") if impact_model is not None else None,
                restored=False,
            )
            self._last_accepted_snapshot = dict(before)
            self._last_accepted_result = result.model_dump(mode="json")
            return self._record(result, diff=diff)
        except Exception as exc:
            if empty_snapshot:
                self._restore(empty_snapshot)
            restored_hashes = self._hash_snapshot(self._snapshot()) if empty_snapshot else {}
            restore_ok = bool(not empty_snapshot or all(restored_hashes.get(path) == value for path, value in self._hash_snapshot(empty_snapshot).items()))
            failure = f"{type(exc).__name__}: {exc}"
            # Even a validation failure before apply gets an evidence file;
            # callers can distinguish it from a rejected post-apply patch.
            manifest = {
                "schema_version": PATCH_EXECUTOR_SCHEMA_VERSION,
                "scope_version": PATCH_SCOPE_VERSION,
                "edit_id": edit_id,
                "owner": owner,
                "source_split": "train",
                "parent_hashes": parent_hashes,
                "source_diff_present": bool(diff.strip()),
                "failure": failure,
            }
            result = PatchExecutionResult(
                status="rejected" if empty_snapshot else "blocked",
                action="rejected" if empty_snapshot else "blocked",
                edit_id=edit_id,
                owner=owner,
                reason=failure,
                parent_hashes=parent_hashes,
                patch_manifest=manifest,
                restored=restore_ok,
                failure_evidence=[failure],
            )
            return self._record(result, diff=diff, failure_payload={"error": failure})

    def apply(self, proposal: Mapping[str, Any] | Any, patch: Any, **kwargs: Any) -> dict[str, Any]:
        """Compatibility alias for :meth:`execute`."""

        return self.execute(proposal, patch, **kwargs)

    def rollback_last(self) -> dict[str, Any]:
        """Restore the last accepted patch and append an auditable rollback event."""

        snapshot = self._last_accepted_snapshot
        accepted = self._last_accepted_result
        if snapshot is None or accepted is None:
            return {
                "status": "blocked",
                "action": "blocked",
                "reason": "no accepted patch is available for rollback",
                "restored": False,
            }
        self._restore(snapshot)
        current = self._hash_snapshot(self._snapshot())
        expected = self._hash_snapshot(snapshot)
        restored = current == expected
        result = {
            "schema_version": PATCH_EXECUTOR_SCHEMA_VERSION,
            "event": "patch_rollback",
            "status": "rolled_back" if restored else "rollback_failed",
            "action": "rollback" if restored else "blocked",
            "reason": (
                "last accepted patch was restored after dev non-regression failure"
                if restored
                else "last accepted patch could not be restored byte-for-byte"
            ),
            "edit_id": accepted.get("edit_id"),
            "owner": accepted.get("owner"),
            "changed_files": accepted.get("changed_files", []),
            "diff_sha256": accepted.get("diff_sha256"),
            "restored": restored,
            "append_only": True,
        }
        self.records.append(result)
        if self.output_dir is not None:
            event_path = self.output_dir / "patch_executions.jsonl"
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            rollback_path = self.output_dir / f"{accepted.get('edit_id', 'patch')}.rollback.json"
            rollback_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self._last_accepted_snapshot = None
        self._last_accepted_result = None
        return result


__all__ = [
    "PATCH_EXECUTOR_SCHEMA_VERSION",
    "PatchExecutionResult",
    "PatchExecutor",
]
