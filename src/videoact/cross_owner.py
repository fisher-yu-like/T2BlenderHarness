"""Fail-closed contract for exceptional multi-owner Harness patches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


CROSS_OWNER_CONTRACT_VERSION = "cross-owner-ablation-v1"
DEPENDENCY_MANIFEST_VERSION = "dependency-manifest-v1"
_FORBIDDEN_CROSS_OWNER_PATH_MARKERS = (
    "outer_loop.py",
    "evaluator",
    "dataset",
    "acceptance",
    "threshold",
)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_owners(proposal: Mapping[str, Any]) -> list[str]:
    raw = proposal.get("owners")
    if raw is None or raw == []:
        raw = [proposal.get("owner")]
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError("owners must be a list")
    owners = list(dict.fromkeys(str(value or "").strip() for value in raw))
    owners = [owner for owner in owners if owner]
    if not owners:
        raise ValueError("at least one Harness owner is required")
    primary = str(proposal.get("owner") or "").strip()
    if primary and primary not in owners:
        raise ValueError("primary owner must be listed in owners")
    return owners


def _validate_file_scope(proposal: Mapping[str, Any]) -> list[str]:
    raw = proposal.get("affected_files", proposal.get("files", []))
    if raw is None:
        raw = []
    if not isinstance(raw, (list, tuple)) or any(not isinstance(item, str) for item in raw):
        raise ValueError("affected_files must be a list of paths")
    paths = [item.replace("\\", "/").lstrip("./") for item in raw]
    for path in paths:
        if not path.startswith("src/videoact/"):
            raise ValueError(f"cross-owner patch is outside Harness scope: {path}")
        lowered = path.casefold()
        if any(marker in lowered for marker in _FORBIDDEN_CROSS_OWNER_PATH_MARKERS):
            raise ValueError(f"cross-owner patch cannot modify acceptance/evaluator paths: {path}")
    return paths


def _number(arm: Mapping[str, Any], field: str) -> float:
    value = arm.get(field)
    if isinstance(value, bool):
        raise ValueError(f"ablation arm field {field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ablation arm missing numeric {field}") from exc
    if not math.isfinite(number):
        raise ValueError(f"ablation arm {field} must be finite")
    return number


def _validate_ablation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("cross-owner patch requires A-only, B-only, and A+B ablation arms")
    required = ("a_only", "b_only", "a_plus_b")
    if any(not isinstance(value.get(key), Mapping) for key in required):
        raise ValueError("cross-owner patch requires A-only, B-only, and A+B ablation arms")
    arms: dict[str, dict[str, Any]] = {}
    for key in required:
        arm = value[key]
        arms[key] = {
            "train_delta": _number(arm, "train_delta"),
            "dev_delta": _number(arm, "dev_delta"),
            "contract_satisfied": bool(arm.get("contract_satisfied")),
            "accepted": bool(arm.get("accepted")),
        }
    if arms["a_only"]["contract_satisfied"] or arms["b_only"]["contract_satisfied"]:
        raise ValueError("single-owner ablation arm unexpectedly satisfies the joint contract")
    if arms["a_only"]["accepted"] or arms["b_only"]["accepted"]:
        raise ValueError("single-owner ablation arm cannot be accepted for a cross-owner exception")
    if not arms["a_plus_b"]["contract_satisfied"] or not arms["a_plus_b"]["accepted"]:
        raise ValueError("A+B ablation must satisfy the contract and pass")
    interaction = {
        "train": arms["a_plus_b"]["train_delta"]
        - arms["a_only"]["train_delta"]
        - arms["b_only"]["train_delta"],
        "dev": arms["a_plus_b"]["dev_delta"]
        - arms["a_only"]["dev_delta"]
        - arms["b_only"]["dev_delta"],
    }
    return {
        "version": str(value.get("version") or CROSS_OWNER_CONTRACT_VERSION),
        "arms": arms,
        "train_dev_deltas": {
            key: {"train": arm["train_delta"], "dev": arm["dev_delta"]}
            for key, arm in arms.items()
        },
        "interaction_effect": interaction,
    }


def validate_cross_owner_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a proposal before it can enter the outer-loop state machine."""

    if not isinstance(proposal, Mapping):
        raise ValueError("patch proposal must be an object")
    owners = _normalise_owners(proposal)
    paths = _validate_file_scope(proposal)
    exception = bool(proposal.get("cross_owner_exception", False))
    if len(owners) == 1:
        if exception:
            raise ValueError("cross_owner_exception requires at least two owners")
        return {
            "version": CROSS_OWNER_CONTRACT_VERSION,
            "status": "pass",
            "owners": owners,
            "affected_files": paths,
            "cross_owner_exception": False,
        }
    if not exception:
        raise ValueError("multiple owners require cross_owner_exception=true")

    dependency = proposal.get("dependency_manifest")
    if not isinstance(dependency, Mapping):
        raise ValueError("cross-owner exception requires a dependency manifest")
    if str(dependency.get("version") or "") != DEPENDENCY_MANIFEST_VERSION:
        raise ValueError("dependency manifest version is missing or unsupported")
    dependency_owners = list(dict.fromkeys(str(item) for item in dependency.get("owners", [])))
    if dependency_owners != owners:
        raise ValueError("dependency manifest owners do not match proposal owners")
    if dependency.get("joint_required") is not True:
        raise ValueError("dependency manifest must prove joint_required=true")
    edges = dependency.get("required_contract_edges")
    if not isinstance(edges, list) or not edges:
        raise ValueError("dependency manifest must contain required contract edges")
    for edge in edges:
        if not isinstance(edge, Mapping) or not str(edge.get("from_owner") or "").strip() or not str(edge.get("to_owner") or "").strip():
            raise ValueError("dependency manifest contains an invalid contract edge")
        if edge["from_owner"] not in owners or edge["to_owner"] not in owners:
            raise ValueError("dependency edge references an owner outside the proposal")

    ablation = _validate_ablation(proposal.get("ablation_report"))
    return {
        "version": CROSS_OWNER_CONTRACT_VERSION,
        "status": "pass",
        "owners": owners,
        "affected_files": paths,
        "cross_owner_exception": True,
        "dependency_manifest_hash": _canonical_hash(dependency),
        "ablation": ablation,
        "train_dev_deltas": ablation["train_dev_deltas"],
        "interaction_effect": ablation["interaction_effect"],
    }


__all__ = [
    "CROSS_OWNER_CONTRACT_VERSION",
    "DEPENDENCY_MANIFEST_VERSION",
    "validate_cross_owner_proposal",
]
