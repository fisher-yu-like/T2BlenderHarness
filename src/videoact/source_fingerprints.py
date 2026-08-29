"""Semantic fingerprints for detecting parameter-only generated sources."""

from __future__ import annotations

import ast
import hashlib
import json
from typing import Any


SOURCE_FINGERPRINT_VERSION = "source-fingerprint-v1"
_ANIMATION_MARKERS = ("frame_set", "keyframe_insert", "animation", "render", "fcurve")
_GEOMETRY_MARKERS = (
    "box",
    "cube",
    "cylinder",
    "sphere",
    "mesh",
    "curve",
    "bezier",
    "character",
    "actor",
    "prop",
)


def _hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return type(node).__name__


class _LiteralAndNameNormalizer(ast.NodeTransformer):
    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="<name>", ctx=node.ctx), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if node.value is None:
            value: Any = None
        elif isinstance(node.value, bool):
            value = False
        elif isinstance(node.value, (int, float, complex)):
            value = 0
        elif isinstance(node.value, str):
            value = "<literal>"
        else:
            value = "<literal>"
        return ast.copy_location(ast.Constant(value=value), node)

    def _strip_docstring(self, node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node = self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        return self._strip_docstring(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node = self.generic_visit(node)
        return self._strip_docstring(node)


def _parse(source: str) -> ast.AST | None:
    try:
        return ast.parse(str(source or ""))
    except SyntaxError:
        return None


def source_fingerprint(source: str) -> dict[str, Any]:
    tree = _parse(source)
    if tree is None:
        return {
            "version": SOURCE_FINGERPRINT_VERSION,
            "parse_status": "invalid",
            "normalized_ast_hash": None,
            "library_call_sequence_hash": None,
            "control_flow_hash": None,
            "scene_graph_signature_hash": None,
            "animation_signature_hash": None,
        }

    normalized = _LiteralAndNameNormalizer().visit(ast.parse(str(source or "")))
    ast.fix_missing_locations(normalized)
    imports: list[str] = []
    calls: list[str] = []
    control_flow: list[str] = []
    scene_graph: list[str] = []
    animation: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(f"{node.module or ''}:{','.join(alias.name for alias in node.names)}")
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            calls.append(name)
            lower = name.casefold()
            if any(marker in lower for marker in _GEOMETRY_MARKERS):
                scene_graph.append(name)
            if any(marker in lower for marker in _ANIMATION_MARKERS):
                animation.append(name)
        elif isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
            control_flow.append(type(node).__name__)
        elif isinstance(node, ast.Attribute) and node.attr in {"keyframe_insert", "animation_data", "fcurves"}:
            animation.append(node.attr)

    return {
        "version": SOURCE_FINGERPRINT_VERSION,
        "parse_status": "valid",
        "normalized_ast_hash": _hash(ast.dump(normalized, annotate_fields=True, include_attributes=False)),
        "library_call_sequence_hash": _hash(imports + calls),
        "control_flow_hash": _hash(control_flow),
        "scene_graph_signature_hash": _hash(scene_graph),
        "animation_signature_hash": _hash(animation),
        "library_call_sequence": imports + calls,
        "control_flow": control_flow,
        "scene_graph_signature": scene_graph,
        "animation_signature": animation,
    }


def compare_source_fingerprints(first: str, second: str) -> dict[str, Any]:
    left = source_fingerprint(first)
    right = source_fingerprint(second)
    fields = (
        "normalized_ast_hash",
        "library_call_sequence_hash",
        "control_flow_hash",
        "scene_graph_signature_hash",
        "animation_signature_hash",
    )
    equal = {field: left.get(field) == right.get(field) for field in fields}
    return {
        "version": SOURCE_FINGERPRINT_VERSION,
        "normalized_ast_equal": equal["normalized_ast_hash"],
        "library_call_sequence_equal": equal["library_call_sequence_hash"],
        "control_flow_equal": equal["control_flow_hash"],
        "scene_graph_signature_equal": equal["scene_graph_signature_hash"],
        "animation_signature_equal": equal["animation_signature_hash"],
        "probable_template_reuse": bool(all(equal.values()) and left.get("parse_status") == "valid" and right.get("parse_status") == "valid"),
        "first": left,
        "second": right,
    }


def audit_source_reuse(source_texts: dict[str, str]) -> dict[str, Any]:
    """Audit a prepared batch for parameter-only code reuse.

    A single shared code shape is not rejected when it is the literal same
    source: the exact source-hash gate reports that more direct failure.  This
    audit targets the evasive case where an agent stamps different plan
    constants, names, or coordinates into one unchanged control-flow and
    animation scaffold.  It only reports a batch-level failure when every
    valid case has the same semantic fingerprint and at least two raw sources
    are distinct, avoiding a false positive for one legitimate repeated
    structure in a larger batch.
    """

    fingerprints = {case_id: source_fingerprint(source) for case_id, source in source_texts.items()}
    valid = {
        case_id: fingerprint
        for case_id, fingerprint in fingerprints.items()
        if fingerprint.get("parse_status") == "valid"
    }
    semantic_fields = (
        "normalized_ast_hash",
        "library_call_sequence_hash",
        "control_flow_hash",
        "scene_graph_signature_hash",
        "animation_signature_hash",
    )
    signatures = {
        case_id: tuple(fingerprint.get(field) for field in semantic_fields)
        for case_id, fingerprint in valid.items()
    }
    unique_signatures = {signature for signature in signatures.values()}
    raw_hashes = {
        hashlib.sha256(str(source_texts[case_id]).encode("utf-8")).hexdigest()
        for case_id in source_texts
    }
    all_same_semantic_shape = bool(valid) and len(valid) == len(source_texts) and len(unique_signatures) == 1
    probable = len(source_texts) > 1 and all_same_semantic_shape and len(raw_hashes) > 1
    return {
        "version": SOURCE_FINGERPRINT_VERSION,
        "status": "probable_template_reuse" if probable else "pass",
        "probable_template_reuse": probable,
        "case_count": len(source_texts),
        "valid_case_count": len(valid),
        "unique_semantic_signature_count": len(unique_signatures),
        "unique_raw_source_count": len(raw_hashes),
        "semantic_fingerprints": {
            case_id: {field: fingerprint.get(field) for field in semantic_fields}
            for case_id, fingerprint in fingerprints.items()
        },
    }


def evaluate_fingerprint_pairs(
    pairs: list[dict[str, Any]],
    *,
    minimum_pairs: int = 100,
) -> dict[str, Any]:
    """Evaluate the semantic-reuse detector against an explicit labeled audit.

    The labels are supplied by the audit fixture/reviewer, never inferred from
    the detector itself.  A pair is positive when the two sources are the
    same parameter-only template.  Invalid or undersized audits fail closed;
    they cannot be reported as a successful precision/recall measurement.
    """

    if isinstance(minimum_pairs, bool) or not isinstance(minimum_pairs, int) or minimum_pairs < 1:
        raise ValueError("minimum_pairs must be a positive integer")
    if not isinstance(pairs, list):
        raise ValueError("pairs must be a list")

    invalid: list[str] = []
    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
    }
    pair_reports: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        pair_id = str(pair.get("pair_id") or f"pair-{index:04d}") if isinstance(pair, dict) else f"pair-{index:04d}"
        if not isinstance(pair, dict):
            invalid.append(f"{pair_id}:not_an_object")
            continue
        first = pair.get("first")
        second = pair.get("second")
        expected = pair.get("expected_template_reuse")
        if not isinstance(first, str) or not first.strip() or not isinstance(second, str) or not second.strip():
            invalid.append(f"{pair_id}:source_missing")
            continue
        if not isinstance(expected, bool):
            invalid.append(f"{pair_id}:expected_template_reuse_must_be_boolean")
            continue
        comparison = compare_source_fingerprints(first, second)
        predicted = bool(comparison["probable_template_reuse"])
        if predicted and expected:
            key = "true_positive"
        elif predicted and not expected:
            key = "false_positive"
        elif not predicted and not expected:
            key = "true_negative"
        else:
            key = "false_negative"
        counts[key] += 1
        pair_reports.append(
            {
                "pair_id": pair_id,
                "expected_template_reuse": expected,
                "predicted_template_reuse": predicted,
                "comparison": {
                    "normalized_ast_equal": comparison["normalized_ast_equal"],
                    "library_call_sequence_equal": comparison["library_call_sequence_equal"],
                    "control_flow_equal": comparison["control_flow_equal"],
                    "scene_graph_signature_equal": comparison["scene_graph_signature_equal"],
                    "animation_signature_equal": comparison["animation_signature_equal"],
                },
            }
        )

    observed = len(pair_reports)
    predicted_positive = counts["true_positive"] + counts["false_positive"]
    actual_positive = counts["true_positive"] + counts["false_negative"]
    precision = counts["true_positive"] / predicted_positive if predicted_positive else None
    recall = counts["true_positive"] / actual_positive if actual_positive else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    status = "pass" if observed >= minimum_pairs and not invalid else "fail"
    reasons = []
    if observed < minimum_pairs:
        reasons.append(f"minimum_pairs_not_reached:{observed}<{minimum_pairs}")
    if invalid:
        reasons.append("invalid_pairs_present")
    return {
        "version": SOURCE_FINGERPRINT_VERSION,
        "status": status,
        "reason": "labeled_fingerprint_audit_passed" if status == "pass" else ";".join(reasons),
        "minimum_pairs": minimum_pairs,
        "pair_count": len(pairs),
        "valid_pair_count": observed,
        "invalid_pair_ids": invalid,
        "confusion_matrix": counts,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pairs": pair_reports,
    }


__all__ = [
    "SOURCE_FINGERPRINT_VERSION",
    "audit_source_reuse",
    "compare_source_fingerprints",
    "evaluate_fingerprint_pairs",
    "source_fingerprint",
]
