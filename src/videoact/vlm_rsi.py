"""VLM-backed Harness-RSI transition and patch boundaries.

This module is the bridge between real-frame visual evidence and the existing
train-only outer controller.  It deliberately keeps three boundaries explicit:

* only scored Codex visual-review evidence can request a patch;
* only one-owner Harness proposals can reach :class:`PatchExecutor`;
* a patch is accepted only after the next attempt passes the dev gate.

The module never edits files directly.  ``PatchExecutor`` remains the sole
mutation boundary and owns byte-for-byte rollback.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .codex_exec_provider import CodexExecProvider
from .outer_controller import OuterTransitionController
from .patch_attribution import normalize_patch_path
from .patch_executor import PatchExecutor


VLM_RSI_MODE = "ai_only_vlm_rsi"
VLM_RSI_REVIEW_SOURCE = "codex_local_visual_review"


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dump(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _dump(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_dump(child) for child in value]
    return value


def _text_hash(value: str | bytes) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unified_diff(current: Mapping[str, str], updated: Mapping[str, str]) -> str:
    chunks: list[str] = []
    for relative in sorted(updated):
        before = str(current.get(relative, "")).splitlines(keepends=True)
        after = str(updated[relative]).splitlines(keepends=True)
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
            )
        )
    return "".join(chunks)


_PATCH_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_contents": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "changed_files": {"type": "array", "items": {"type": "string"}},
        "diff": {"type": "string"},
        "diff_sha256": {"type": "string"},
        "patch_manifest": {"type": "object", "additionalProperties": True},
    },
    "required": ["file_contents", "changed_files"],
    "additionalProperties": False,
}


class CodexHarnessPatchAgent:
    """Ask local Codex for a bounded full-file Harness patch response."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        command: str = "codex",
        model: str = "gpt-5.6-luna",
        timeout_s: int = 1800,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.command = command
        self.model = model
        self.timeout_s = int(timeout_s)
        self.provider = CodexExecProvider(
            command=command,
            timeout_s=timeout_s,
            response_schema=_PATCH_RESPONSE_SCHEMA,
            prompt_builder=self._build_prompt,
            stage="harness_patch",
            provider_kind="codex_exec_harness_patch",
            model_id="codex-cli-harness-patch",
            model_version="codex-exec-harness-patch-v1",
            template_backed=False,
            llm_generated=True,
            model=model,
            reasoning_effort="low",
        )

    def _current_files(self, affected_files: Sequence[str]) -> dict[str, str]:
        current: dict[str, str] = {}
        for raw_path in affected_files:
            relative = normalize_patch_path(str(raw_path))
            path = self.repo_root / Path(relative)
            try:
                path.resolve().relative_to(self.repo_root)
            except ValueError as exc:
                raise ValueError(f"patch file escapes repo root: {relative}") from exc
            if not path.is_file():
                raise ValueError(f"patch target is missing: {relative}")
            current[relative] = path.read_text(encoding="utf-8")
        return current

    def _build_prompt(self, payload: Any) -> str:
        request = payload if isinstance(payload, Mapping) else {}
        proposal = request.get("proposal") or request
        return (
            "You are the Harness-RSI coding agent. Generate one bounded source patch from the supplied "
            "train-only visual evidence. Modify only the listed affected Harness files under src/videoact. "
            "Return complete replacement text in file_contents for every changed file. Do not edit dataset, "
            "evaluator, observer, tests, frozen test policy, generated videos, or configuration gates. "
            "Do not use a template or fallback. Preserve existing public contracts and keep the patch small. "
            "Return only JSON matching the schema: file_contents, changed_files, diff, diff_sha256, "
            "patch_manifest. The host recomputes the diff hash, so the response must be internally consistent.\n"
            + json.dumps(
                {
                    "proposal": _dump(proposal),
                    "train_evidence": _dump(request.get("train_evidence") or []),
                    "current_files": _dump(request.get("current_files") or {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    def __call__(self, request: Mapping[str, Any] | Any) -> dict[str, Any]:
        payload = _dump(request)
        if not isinstance(payload, Mapping):
            raise ValueError("Harness patch request must be an object")
        proposal = payload.get("proposal") or payload
        if not isinstance(proposal, Mapping):
            raise ValueError("Harness patch request is missing proposal")
        affected = proposal.get("affected_files") or []
        current = self._current_files([str(item) for item in affected])
        model_request = {
            "proposal": dict(proposal),
            "train_evidence": payload.get("train_evidence") or [],
            "current_files": current,
        }
        result = dict(self.provider(model_request))
        raw_contents = result.get("file_contents")
        if not isinstance(raw_contents, Mapping) or not raw_contents:
            raise ValueError("Harness patch response must include file_contents")
        updated = {
            normalize_patch_path(str(path)): str(content)
            for path, content in raw_contents.items()
        }
        allowed = set(current)
        if not set(updated).issubset(allowed):
            outside = sorted(set(updated) - allowed)
            raise ValueError(f"Harness patch response changed files outside proposal: {outside}")
        changed_files = sorted(
            path for path in updated if updated[path] != current.get(path, "")
        )
        if not changed_files:
            raise ValueError("Harness patch response contains no changed file")
        updated = {path: updated[path] for path in changed_files}
        before = {path: current[path] for path in changed_files}
        diff = _unified_diff(before, updated)
        if not diff.strip():
            raise ValueError("Harness patch response produced an empty diff")
        diff_sha256 = _text_hash(diff)
        result.update(
            {
                "file_contents": updated,
                "changed_files": changed_files,
                "diff": diff,
                "diff_sha256": diff_sha256,
                "patch_manifest": {
                    **dict(result.get("patch_manifest") or {}),
                    "source_diff_present": True,
                    "changed_files": changed_files,
                    "diff_sha256": diff_sha256,
                    "source_split": "train",
                    "template_backed": False,
                    "llm_generated": True,
                },
            }
        )
        return result


def _split_report(report: Mapping[str, Any], split: str) -> Mapping[str, Any] | None:
    splits = report.get("splits")
    if isinstance(splits, Mapping):
        candidate = splits.get(split)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _train_run_root(report: Mapping[str, Any]) -> Path | None:
    train = _split_report(report, "train")
    if not train:
        return None
    value = train.get("run_root")
    if not value:
        return None
    return Path(str(value))


def _aggregate(split_report: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(split_report, Mapping):
        return {}
    value = split_report.get("aggregate")
    return value if isinstance(value, Mapping) else {}


class VlmRsiTransitionController:
    """Callable transition controller for a VLM-backed round."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        repo_root: str | Path = ".",
        coding_agent: Callable[[Mapping[str, Any]], Any] | None = None,
        patch_executor: PatchExecutor | None = None,
        forbidden_case_ids: Sequence[str] = (),
        dev_tolerance: float = 2.0,
        unit_test_runner: Callable[..., Any] | None = None,
        owner_challenge_runner: Callable[..., Any] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.repo_root = Path(repo_root).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.coding_agent = coding_agent
        self.forbidden_case_ids = {str(item) for item in forbidden_case_ids}
        self.dev_tolerance = float(dev_tolerance)
        self.controller = OuterTransitionController(output_dir=self.output_dir / "outer")
        self.patch_executor = patch_executor or PatchExecutor(
            repo_root=self.repo_root,
            output_dir=self.output_dir / "patch",
            owner_challenge_runner=owner_challenge_runner or self._default_owner_challenge,
            unit_test_runner=unit_test_runner or self._default_unit_gate,
        )
        self._pending: dict[str, Any] | None = None

    @staticmethod
    def _validate_train_reports(reports: Sequence[Mapping[str, Any]]) -> None:
        """Reject any direct test/dev record from the patch boundary."""

        for report in reports:
            split = str(report.get("split") or "").casefold()
            if split and split != "train":
                raise ValueError("VLM-RSI patch transition is train-only")
            splits = report.get("splits")
            if not isinstance(splits, Mapping):
                continue
            train = splits.get("train")
            if isinstance(train, Mapping) and str(train.get("split") or "train").casefold() != "train":
                raise ValueError("VLM-RSI patch transition is train-only")
            records = train.get("records") if isinstance(train, Mapping) else None
            if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
                for record in records:
                    if isinstance(record, Mapping) and str(record.get("split") or "train").casefold() != "train":
                        raise ValueError("VLM-RSI patch transition is train-only")

    @staticmethod
    def _vlm_scored_count(report: Mapping[str, Any]) -> int:
        train = _split_report(report, "train")
        if not train:
            return 0
        try:
            return int(train.get("vlm_scored_count") or 0)
        except (TypeError, ValueError):
            return 0

    def _default_owner_challenge(self, context: Mapping[str, Any]) -> dict[str, Any]:
        changed = context.get("changed_files") or []
        passed = bool(changed) and all(str(path).replace("\\", "/").startswith("src/videoact/") for path in changed)
        return {"status": "pass" if passed else "fail", "method": "Harness_scope_challenge"}

    def _default_unit_gate(self, context: Mapping[str, Any]) -> dict[str, Any]:
        changed = [str(path) for path in context.get("changed_files") or [] if str(path).endswith(".py")]
        if not changed:
            return {"status": "pass", "method": "no_python_files_changed"}
        completed = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", *changed],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "status": "pass" if completed.returncode == 0 else "fail",
            "method": "compileall",
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "")[-2000:],
        }

    def _validate_vlm_report(self, report: Mapping[str, Any]) -> None:
        if self._vlm_scored_count(report) < 1:
            raise ValueError("VLM-RSI requires at least one scored train VLM case before patch selection")
        train = _split_report(report, "train") or {}
        cases = train.get("cases") if isinstance(train, Mapping) else None
        if isinstance(cases, Sequence):
            sources = {
                str(case.get("review_source"))
                for case in cases
                if isinstance(case, Mapping) and case.get("vlm_status") == "scored"
            }
            if sources and sources != {VLM_RSI_REVIEW_SOURCE}:
                raise ValueError("VLM-RSI train evidence contains a non-Codex visual review source")

    def _dev_gate(self, before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
        before_dev = _split_report(before, "dev")
        after_dev = _split_report(after, "dev")
        before_agg = _aggregate(before_dev)
        after_agg = _aggregate(after_dev)
        metrics = ("mean_task_final_score", "mean_visual_score", "mean_deterministic_score")
        deltas: dict[str, float | None] = {}
        failures: list[str] = []
        for metric in metrics:
            old = _number(before_agg.get(metric))
            new = _number(after_agg.get(metric))
            deltas[metric] = None if old is None or new is None else round(new - old, 4)
            if old is None or new is None:
                failures.append(f"{metric}_unavailable")
            elif new < old - self.dev_tolerance:
                failures.append(f"{metric}_regressed")
        before_count = int((before_dev or {}).get("vlm_scored_count") or 0)
        after_count = int((after_dev or {}).get("vlm_scored_count") or 0)
        if after_count < before_count:
            failures.append("vlm_scored_coverage_regressed")
        return {
            "status": "pass" if not failures else "fail",
            "reason": "dev_non_regression_pass" if not failures else ";".join(failures),
            "deltas": deltas,
            "before_vlm_scored_count": before_count,
            "after_vlm_scored_count": after_count,
        }

    def _collect_train_evidence(self, report: Mapping[str, Any]) -> list[dict[str, Any]]:
        root = _train_run_root(report)
        if root is None:
            raise ValueError("VLM-RSI cannot locate the train evidence root")
        return self.controller.collect_train_evidence(
            root,
            forbidden_case_ids=self.forbidden_case_ids,
        )

    def _fallback_train_evidence(self, report: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Build a strict train-only adapter from the already-merged report."""

        train = _split_report(report, "train") or {}
        cases = train.get("cases")
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
            return []
        evidence: list[dict[str, Any]] = []
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            if str(case.get("review_source") or "") != VLM_RSI_REVIEW_SOURCE:
                continue
            score = _number(case.get("task_final_score"))
            confidence = _number(case.get("review_confidence"))
            if score is None or confidence is None or confidence < 0.6:
                continue
            evidence.append(
                {
                    "case_id": str(case.get("case_id")),
                    "split": "train",
                    "findings": [],
                    "vlm": {
                        "status": "scored",
                        "review_source": VLM_RSI_REVIEW_SOURCE,
                        "task_final_score": score,
                        "confidence": confidence,
                    },
                }
            )
        return evidence

    def __call__(self, attempt_number: int, reports: list[dict[str, Any]]) -> dict[str, Any]:
        self._validate_train_reports(reports)
        if not reports:
            return {"action": "stop", "status": "blocked", "reason": "no train attempt report"}
        latest = reports[-1]
        if self._pending is not None:
            gate = self._dev_gate(self._pending["before"], latest)
            if gate["status"] == "pass":
                self.controller.append_decision(
                    action="accepted",
                    status="accepted",
                    reason="VLM-RSI patch passed the post-patch dev non-regression gate",
                    attempt=attempt_number,
                    owner=self._pending["proposal"].get("owner"),
                    source_case_ids=self._pending["proposal"].get("source_case_ids", []),
                    proposal=self._pending["proposal"],
                    attribution={"dev_gate": gate},
                )
                accepted = {"action": "accept", "status": "accepted", "reason": gate["reason"], "dev_gate": gate}
                self._pending = None
                return accepted
            rollback = self.patch_executor.rollback_last()
            self.controller.append_decision(
                action="rollback" if rollback.get("status") == "rolled_back" else "blocked",
                status=str(rollback.get("status")),
                reason=str(gate["reason"]),
                attempt=attempt_number,
                owner=self._pending["proposal"].get("owner"),
                source_case_ids=self._pending["proposal"].get("source_case_ids", []),
                proposal=self._pending["proposal"],
                attribution={"dev_gate": gate, "rollback": rollback},
            )
            self._pending = None
            return {
                "action": "stop",
                "status": "rollback" if rollback.get("status") == "rolled_back" else "blocked",
                "reason": f"post-patch dev gate failed: {gate['reason']}",
                "dev_gate": gate,
                "rollback": rollback,
            }

        if attempt_number >= 5:
            reason = "VLM-RSI cannot apply an unverified patch on the final outer attempt"
            self.controller.append_decision(
                action="blocked",
                status="blocked",
                reason=reason,
                attempt=attempt_number,
            )
            return {"action": "stop", "status": "blocked", "reason": reason}

        try:
            self._validate_vlm_report(latest)
            try:
                evidence = self._collect_train_evidence(latest)
            except ValueError:
                evidence = self._fallback_train_evidence(latest)
            if not evidence:
                raise ValueError("VLM-RSI cannot normalize scored train VLM evidence")
        except ValueError as exc:
            self.controller.append_decision(
                action="blocked",
                status="blocked",
                reason=str(exc),
                attempt=attempt_number,
            )
            return {"action": "stop", "status": "blocked", "reason": str(exc)}

        if self.coding_agent is None:
            reason = "VLM-RSI requires a Harness coding agent for patch generation"
            self.controller.append_decision(action="blocked", status="blocked", reason=reason, attempt=attempt_number)
            return {"action": "stop", "status": "blocked", "reason": reason}

        captured: dict[str, Any] = {}

        def coding_agent(proposal: Mapping[str, Any]) -> Any:
            request = {
                "proposal": dict(proposal),
                "train_evidence": evidence,
            }
            response = self.coding_agent(request)
            captured["response"] = response
            return response

        transition = self.controller.transition(
            attempt_number,
            reports,
            coding_agent=coding_agent,
        )
        if transition.get("action") != "patch":
            return {
                "action": "stop",
                "status": transition.get("status", "no_patch"),
                "reason": transition.get("reason", "no VLM-backed patch was justified"),
            }
        proposal = transition.get("proposal")
        patch = captured.get("response")
        if not isinstance(proposal, Mapping) or patch is None:
            return {"action": "stop", "status": "blocked", "reason": "patch response was not captured"}
        execution = self.patch_executor.execute(
            proposal,
            patch,
            train_evidence=evidence,
            forbidden_case_ids=self.forbidden_case_ids,
        )
        if execution.get("status") != "accepted":
            return {
                "action": "stop",
                "status": execution.get("status", "blocked"),
                "reason": execution.get("reason", "Patch Executor rejected the patch"),
                "patch_execution": execution,
            }
        self._pending = {
            "before": latest,
            "proposal": dict(proposal),
            "execution": execution,
        }
        return {
            "action": "patch",
            "status": "patch_applied_waiting_for_dev_gate",
            "reason": "VLM-backed Harness patch applied; next attempt is the dev non-regression gate",
            "proposal": dict(proposal),
            "patch_execution": execution,
        }


__all__ = [
    "CodexHarnessPatchAgent",
    "VLM_RSI_MODE",
    "VLM_RSI_REVIEW_SOURCE",
    "VlmRsiTransitionController",
]
