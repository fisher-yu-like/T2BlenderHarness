from evaluator.geometry_realism import evaluate_geometry_report


def _scene():
    return {
        "geometry": {
            "detail_required": True,
            "min_vertices": 256,
            "min_faces": 128,
            "required_entity_ids": ["character", "red_cup"],
            "required_entity_kinds": {"character": "character", "red_cup": "prop"},
            "forbid_primitive_hints": ["uv_sphere", "sphere", "cylinder", "cube"],
        }
    }


def test_detailed_mesh_evidence_passes():
    report = evaluate_geometry_report({
        "audit_available": True,
        "meshes": [
            {"entity_id": "character", "entity_kind": "character", "geometry_style": "detailed_parametric_v1", "connected_component_count": 8, "vertex_count": 800, "face_count": 1200, "primitive_hint": None},
            {"entity_id": "red_cup", "entity_kind": "prop", "geometry_style": "detailed_parametric_v1", "connected_component_count": 2, "vertex_count": 500, "face_count": 700, "primitive_hint": None},
        ],
    }, _scene())
    assert report["hard_gate_failed"] is False
    assert report["score"] == 100.0


def test_primitive_standins_are_hard_failures():
    report = evaluate_geometry_report({
        "audit_available": True,
        "meshes": [
            {"entity_id": "character", "entity_kind": "character", "geometry_style": "detailed_parametric_v1", "connected_component_count": 1, "vertex_count": 500, "face_count": 700, "primitive_hint": "uv_sphere"},
            {"entity_id": "red_cup", "entity_kind": "prop", "geometry_style": "detailed_parametric_v1", "connected_component_count": 1, "vertex_count": 500, "face_count": 700, "primitive_hint": "cylinder"},
        ],
    }, _scene())
    assert report["hard_gate_failed"] is True
    assert any(finding["failure_id"] == "proxy_coarse_primitive" for finding in report["findings"])
    assert report["score"] <= 20


def test_missing_required_entity_is_a_hard_failure():
    report = evaluate_geometry_report({
        "audit_available": True,
        "meshes": [{"entity_id": "character", "entity_kind": "character", "geometry_style": "detailed_parametric_v1", "connected_component_count": 8, "vertex_count": 800, "face_count": 1200}],
    }, _scene())
    assert report["hard_gate_failed"] is True
    assert any(finding["failure_id"] == "proxy_entity_missing_from_blend" for finding in report["findings"])


def test_ground_environment_mesh_is_not_a_missing_detailed_entity():
    report = evaluate_geometry_report({
        "audit_available": True,
        "meshes": [
            {"entity_id": "character", "entity_kind": "character", "geometry_style": "detailed_parametric_v1", "connected_component_count": 8, "vertex_count": 800, "face_count": 1200, "primitive_hint": None},
            {"entity_id": "red_cup", "entity_kind": "prop", "geometry_style": "detailed_parametric_v1", "connected_component_count": 2, "vertex_count": 500, "face_count": 700, "primitive_hint": None},
            {"entity_id": "ground_plane", "entity_kind": "environment", "geometry_style": "ground_contact_surface_v1", "connected_component_count": 1, "vertex_count": 4, "face_count": 1, "primitive_hint": None},
        ],
    }, {"geometry": {"detail_required": True}})

    assert report["hard_gate_failed"] is False
    assert not any(finding["failure_id"] == "proxy_geometry_style_missing" for finding in report["findings"])
