"""Anti-saturation realism evaluation for Blender proxy artifacts.

The geometry audit is a hard eligibility check, not a realism oracle.  This
module combines continuous geometry evidence with measurements from real
rendered PNGs.  Without an independent VLM or human review, the result is
explicitly an artifact-only proxy score and is capped below 100.
"""

from __future__ import annotations

import math
from typing import Any


REALISM_EVALUATOR_VERSION = "realism-v4-shared-visual-review"
ARTIFACT_ONLY_CEILING = 80.0
GEOMETRY_WEIGHTS = {
    "entity_coverage": 0.20,
    "topology_detail": 0.20,
    "non_primitive_representation": 0.20,
    "semantic_integrity": 0.10,
    "structural_detail": 0.30,
}
INDEPENDENT_REVIEW_WEIGHTS = {
    "appearance_detail": 0.20,
    "physical_realism": 0.20,
    "spatial_consistency": 0.20,
    "motion_naturalness": 0.20,
    "visual_presentation": 0.20,
}
_STRUCTURAL_TARGETS = {"character": 8, "support": 8, "occluder": 6, "prop": 3}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _continuous_topology(mesh: dict[str, Any], min_vertices: int, min_faces: int) -> float:
    vertices = max(0, int(mesh.get("vertex_count", 0) or 0))
    faces = max(0, int(mesh.get("face_count", 0) or 0))
    vertex_target = max(512, min_vertices * 2)
    face_target = max(256, min_faces * 2)
    vertex_ratio = math.log1p(vertices) / math.log1p(vertex_target) if vertex_target else 0.0
    face_ratio = math.log1p(faces) / math.log1p(face_target) if face_target else 0.0
    return _clamp(50.0 * vertex_ratio + 50.0 * face_ratio)


def _geometry_quality(report: dict[str, Any]) -> tuple[dict[str, float], float]:
    meshes = list(report.get("mesh_evidence") or [])
    required_count = int(report.get("required_entity_count", 0) or 0)
    if meshes:
        min_vertices = int(report.get("min_vertices", 0) or 0)
        min_faces = int(report.get("min_faces", 0) or 0)
        topology = sum(_continuous_topology(mesh, min_vertices, min_faces) for mesh in meshes) / len(meshes)
        structural_values = []
        for mesh in meshes:
            kind = str(mesh.get("entity_kind") or "prop")
            target = _STRUCTURAL_TARGETS.get(kind, 3)
            components = max(0, int(mesh.get("connected_component_count", 0) or 0))
            style_bonus = 1.0 if mesh.get("geometry_style") == "detailed_parametric_v1" else 0.0
            structural_values.append(100.0 * (0.75 * min(components / target, 1.0) + 0.25 * style_bonus))
        structural = sum(structural_values) / len(structural_values)
    else:
        topology = float(report.get("topology_score", 0.0))
        structural = float(report.get("structural_score", 0.0))
    required = max(1, required_count)
    components = {
        "entity_coverage": _clamp(report.get("coverage_score", 0.0)),
        "topology_detail": _clamp(topology),
        "non_primitive_representation": _clamp(report.get("primitive_score", 0.0)),
        "semantic_integrity": _clamp(report.get("semantic_score", 0.0)),
        "structural_detail": _clamp(structural),
    }
    score = sum(components[name] * GEOMETRY_WEIGHTS[name] for name in components)
    if bool(report.get("hard_gate_failed", True)):
        score = min(score, 20.0)
    return components, score


def _review_score(independent_review: dict[str, Any] | None) -> tuple[float | None, str | None]:
    review = independent_review or {}
    if review.get("status") != "complete" or float(review.get("confidence", 0.0) or 0.0) < 0.6:
        return None, None
    source = str(review.get("source") or "")
    if source not in {"gpt-5.6-luna", "gpt-5.6-terra", "assistant_local_review", "human"}:
        return None, None
    scores = review.get("scores") or {}
    if not all(name in scores for name in INDEPENDENT_REVIEW_WEIGHTS):
        return None, None
    values = [_clamp(scores[name]) for name in INDEPENDENT_REVIEW_WEIGHTS]
    score = math.prod(values) ** (1.0 / len(values)) if values else 0.0
    return score, source


def score_realism(
    geometry_report: dict[str, Any] | None,
    visual_report: dict[str, Any] | None = None,
    independent_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return artifact evidence and, only when available, independent review.

    The compatibility field ``score`` is always numeric.  It means
    ``artifact_only_proxy`` until a complete independent review is supplied;
    it must not be described as photorealism or VLM quality in that state.
    """
    report = geometry_report or {}
    geometry_components, geometry_score = _geometry_quality(report)
    visual = visual_report or {}
    visual_score = _clamp(visual.get("score", 0.0)) if visual.get("status") in {"complete", "partial"} else 0.0
    artifact_unbounded = 0.60 * geometry_score + 0.40 * visual_score
    # Reserve 20% for semantics, physical plausibility, and motion quality
    # that low-level geometry/PNG measurements cannot observe.  This is a
    # smooth uncertainty penalty, not a candidate-fitted threshold, and keeps
    # case-to-case differences instead of turning all compliant meshes into
    # the same ceiling value.
    artifact_score = min(ARTIFACT_ONLY_CEILING, 0.80 * artifact_unbounded)
    reviewed_score, review_source = _review_score(independent_review)
    if reviewed_score is None:
        score = artifact_score
        score_kind = "artifact_only_proxy"
        band = "artifact_only_strong" if score >= 70.0 else "artifact_only_partial" if score >= 45.0 else "artifact_only_weak"
        realism_claim = "not_established"
        requires_independent_review = True
    else:
        # VLM/local visual review is the main realism signal.  Geometry and
        # low-level frame evidence remain independent safeguards, not a
        # substitute for judging visible realism.
        score = 0.15 * geometry_score + 0.15 * visual_score + 0.70 * reviewed_score
        score_kind = "independent_review_fused"
        band = "realism_pass" if score >= 85.0 else "realism_partial" if score >= 60.0 else "realism_fail"
        realism_claim = "independent_review_supported"
        requires_independent_review = False
    return {
        "evaluator_version": REALISM_EVALUATOR_VERSION,
        "score": round(score, 4),
        "score_kind": score_kind,
        "band": band,
        "realism_claim": realism_claim,
        "requires_independent_review": requires_independent_review,
        "hard_gate_failed": bool(report.get("hard_gate_failed", True)),
        "artifact_only_ceiling": ARTIFACT_ONLY_CEILING,
        "geometry_score": round(geometry_score, 4),
        "geometry_components": {name: round(value, 4) for name, value in geometry_components.items()},
        "visual_evidence_score": round(visual_score, 4),
        "artifact_only_unbounded_score": round(artifact_unbounded, 4),
        "weights": {
            "artifact_only": {"geometry": 0.60, "rendered_frame_evidence": 0.40, "unobserved_quality_reserve": 0.20},
            "geometry": GEOMETRY_WEIGHTS,
        "review_fusion": {"geometry": 0.15, "rendered_frame_evidence": 0.15, "independent_review": 0.70},
            "independent_review": INDEPENDENT_REVIEW_WEIGHTS,
        },
        "independent_review_source": review_source,
        "source": "actual_blender_geometry_report_and_sampled_pngs",
    }
