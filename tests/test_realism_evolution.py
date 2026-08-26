from scripts.evaluate_realism_evolution import legacy_score


def test_legacy_score_ignores_independent_geometry_findings():
    report = {
        "score": 0,
        "findings": [
            {"failure_id": "geometry", "owner": "proxy_renderer", "category": "geometry_realism", "severity": "hard", "message": "geometry", "repair_route": "runtime_repair"},
            {"failure_id": "camera", "owner": "camera_planner", "category": "camera_coverage", "severity": "error", "message": "camera", "repair_route": "camera_repair"},
        ],
    }
    assert legacy_score(report) == 82.0
