"""Artifact-level geometry realism checks for detailed proxy evaluations.

This evaluator is deliberately independent of the scene planner.  A valid
prompt contract and a playable video are insufficient when the Blender file
uses a few primitive stand-ins for an actor, prop, or scene element.
"""

from __future__ import annotations

from typing import Any


AUDIT_VERSION = "geometry-realism-v3"
DEFAULT_FORBIDDEN = {"uv_sphere", "sphere", "cylinder", "cube"}
MIN_COMPONENTS = {"character": 6, "support": 2, "occluder": 3, "prop": 1}


def _finding(failure_id: str, severity: str, message: str, evidence: list[str], *, root_cause_id: str | None = None) -> dict[str, Any]:
    return {
        "failure_id": failure_id,
        "owner": "proxy_renderer",
        "category": "geometry_realism",
        "severity": severity,
        "root_cause_id": root_cause_id or failure_id,
        "message": message,
        "evidence": evidence,
        "repair_route": "runtime_repair",
    }


def evaluate_geometry_report(raw: dict[str, Any] | None, proxy_scene: dict[str, Any] | None) -> dict[str, Any]:
    """Score Blender mesh evidence and return serializable findings.

    For ordinary proxy cases the audit is informational.  For a scene whose
    dataset declares ``detail_required``, missing evidence, missing entities,
    forbidden primitive stand-ins, or insufficient topology are hard failures.
    """
    raw = raw or {}
    proxy_scene = proxy_scene or {}
    spec = proxy_scene.get("geometry") or {}
    detail_required = bool(spec.get("detail_required", False))
    required_ids = list(spec.get("required_entity_ids") or [])
    required_kinds = dict(spec.get("required_entity_kinds") or {})
    min_vertices = int(spec.get("min_vertices", 0))
    min_faces = int(spec.get("min_faces", 0))
    forbidden = {str(item).lower() for item in spec.get("forbid_primitive_hints", DEFAULT_FORBIDDEN)}
    meshes = list(raw.get("meshes") or [])
    by_id = {str(mesh.get("entity_id") or mesh.get("name")): mesh for mesh in meshes}
    findings: list[dict[str, Any]] = []

    if detail_required and not raw.get("audit_available", False):
        findings.append(_finding(
            "geometry_audit_unavailable",
            "hard",
            "detailed geometry was required but Blender mesh inspection evidence is unavailable",
            ["geometry_raw.json missing or audit_available=false"],
            root_cause_id="geometry_evidence_missing",
        ))

    missing = [entity_id for entity_id in required_ids if entity_id not in by_id]
    if missing:
        findings.append(_finding(
            "proxy_entity_missing_from_blend",
            "hard" if detail_required else "error",
            f"required scene entities are absent from the Blender mesh audit: {missing}",
            missing,
            root_cause_id="geometry_required_entity_coverage",
        ))

    wrong_kinds = []
    for entity_id, expected_kind in required_kinds.items():
        observed = by_id.get(entity_id, {}).get("entity_kind")
        if observed is not None and observed != expected_kind:
            wrong_kinds.append(f"{entity_id}: observed={observed}, expected={expected_kind}")
    if wrong_kinds:
        findings.append(_finding(
            "proxy_entity_kind_mismatch_in_blend",
            "hard" if detail_required else "error",
            "Blender semantic entity kinds do not match the authored proxy scene",
            wrong_kinds,
            root_cause_id="geometry_entity_semantics",
        ))

    low_topology = []
    coarse = []
    structural = []
    style_missing = []
    for entity_id in required_ids or sorted(by_id):
        mesh = by_id.get(entity_id)
        if not mesh:
            continue
        primitive_hint = str(mesh.get("primitive_hint") or "").lower()
        vertices = int(mesh.get("vertex_count", 0) or 0)
        faces = int(mesh.get("face_count", mesh.get("polygon_count", 0)) or 0)
        entity_kind = str(mesh.get("entity_kind") or required_kinds.get(entity_id) or "prop")
        component_count = int(mesh.get("connected_component_count", 0) or 0)
        if detail_required and primitive_hint in forbidden:
            coarse.append(f"{entity_id}: primitive_hint={primitive_hint}")
        if vertices < min_vertices or faces < min_faces:
            low_topology.append(f"{entity_id}: vertices={vertices} (<{min_vertices}), faces={faces} (<{min_faces})")
        geometry_style = mesh.get("geometry_style")
        infrastructure_style = entity_kind == "environment" and geometry_style == "ground_contact_surface_v1"
        if detail_required and geometry_style != "detailed_parametric_v1" and not infrastructure_style:
            style_missing.append(f"{entity_id}: geometry_style={mesh.get('geometry_style')!r}")
        minimum_components = MIN_COMPONENTS.get(entity_kind, 1)
        if detail_required and component_count < minimum_components:
            structural.append(f"{entity_id}: connected_components={component_count} (<{minimum_components})")
    if coarse:
        findings.append(_finding(
            "proxy_coarse_primitive",
            "hard" if detail_required else "warning",
            "detailed realism is violated by primitive stand-ins for required entities",
            coarse,
            root_cause_id="geometry_primitive_standin",
        ))
    if low_topology:
        findings.append(_finding(
            "proxy_geometry_low_complexity",
            "hard" if detail_required else "warning",
            "required meshes do not meet the minimum topology budget for detailed rendering",
            low_topology,
            root_cause_id="geometry_topology_budget",
        ))
    if structural:
        findings.append(_finding(
            "proxy_structural_detail_insufficient",
            "hard" if detail_required else "warning",
            "required meshes do not contain enough disconnected structural components for a detailed representation",
            structural,
            root_cause_id="geometry_structural_components",
        ))
    if style_missing:
        findings.append(_finding(
            "proxy_geometry_style_missing",
            "hard" if detail_required else "warning",
            "detailed geometry style metadata is absent from one or more required Blender meshes",
            style_missing,
            root_cause_id="geometry_style_contract",
        ))

    required_count = len(required_ids) or len(meshes)
    present_count = required_count - len(missing)
    coverage_score = 100.0 if required_count == 0 else 100.0 * present_count / required_count
    topology_fail_count = len(low_topology)
    primitive_fail_count = len(coarse)
    structural_fail_count = len(structural) + len(style_missing)
    topology_score = 100.0 if required_count == 0 else max(0.0, 100.0 * (required_count - topology_fail_count) / required_count)
    primitive_score = 100.0 if required_count == 0 else max(0.0, 100.0 * (required_count - primitive_fail_count) / required_count)
    semantic_score = 100.0 if required_count == 0 else max(0.0, 100.0 * (required_count - len(wrong_kinds)) / required_count)
    structural_score = 100.0 if required_count == 0 else max(0.0, 100.0 * (required_count - structural_fail_count) / required_count)
    score = round(0.20 * coverage_score + 0.25 * topology_score + 0.25 * primitive_score + 0.10 * semantic_score + 0.20 * structural_score, 4)
    if detail_required and any(finding["severity"] == "hard" for finding in findings):
        score = min(score, 20.0)
    mesh_evidence = [
        {
            "entity_id": str(mesh.get("entity_id") or mesh.get("name")),
            "entity_kind": mesh.get("entity_kind"),
            "vertex_count": int(mesh.get("vertex_count", 0) or 0),
            "face_count": int(mesh.get("face_count", mesh.get("polygon_count", 0)) or 0),
            "connected_component_count": int(mesh.get("connected_component_count", 0) or 0),
            "primitive_hint": mesh.get("primitive_hint"),
            "geometry_style": mesh.get("geometry_style"),
        }
        for mesh in meshes
    ]
    return {
        "audit_version": AUDIT_VERSION,
        "detail_required": detail_required,
        "required_entity_count": required_count,
        "mesh_count": len(meshes),
        "min_vertices": min_vertices,
        "min_faces": min_faces,
        "mesh_evidence": mesh_evidence,
        "coverage_score": round(coverage_score, 4),
        "topology_score": round(topology_score, 4),
        "primitive_score": round(primitive_score, 4),
        "semantic_score": round(semantic_score, 4),
        "structural_score": round(structural_score, 4),
        "score": score,
        "hard_gate_failed": any(finding["severity"] == "hard" for finding in findings),
        "findings": findings,
    }
