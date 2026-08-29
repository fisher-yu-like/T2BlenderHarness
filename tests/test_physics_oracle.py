from __future__ import annotations


def _box(center, half=(0.25, 0.25, 0.25)):
    return {
        "min": [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
        "max": [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
    }


def _record(*, negative=None):
    return {
        "case_id": "physics-fixture",
        "fps": 1,
        "event_graph": [
            {"id": "attach_01", "action": "attach", "start": 1.0, "end": 2.0, "target_ids": ["prop_01"]},
            {"id": "handoff_01", "action": "handoff", "start": 2.0, "end": 4.0, "target_ids": ["prop_01"]},
            {"id": "detach_01", "action": "detach", "start": 4.0, "end": 5.0, "target_ids": ["prop_01"]},
        ],
        "interactions": [
            {
                "id": "life-01",
                "prop_id": "prop_01",
                "giver_id": "actor_a",
                "receiver_id": "actor_b",
                "attach_event_id": "attach_01",
                "transfer_event_id": "handoff_01",
                "detach_event_id": "detach_01",
                "final_owner_id": "actor_b",
                "final_support_id": "support_surface",
            }
        ],
        "oracle_expectations": {
            "required_negative_constraints": negative or ["no_prop_penetration"],
        },
        "entities": [
            {"id": "actor_a", "kind": "actor"},
            {"id": "actor_b", "kind": "actor"},
            {"id": "prop_01", "kind": "prop"},
            {"id": "support_surface", "kind": "support"},
        ],
    }


def _telemetry(*, embedded=False, receiver_close=True, teleport=False):
    observations = []
    for frame in range(1, 7):
        prop_center = [1.0, 0.0, 1.0]
        if teleport and frame == 2:
            prop_center = [10.0, 0.0, 1.0]
        receiver_hand = [1.0, 0.0, 1.0] if receiver_close else [5.0, 0.0, 1.0]
        actor_bounds = _box([0.0, 0.0, 1.0], (0.3, 0.3, 1.0))
        prop_bounds = _box(prop_center, (0.2, 0.2, 0.2))
        if embedded and frame == 6:
            prop_bounds = _box([0.0, 0.0, 1.0], (0.3, 0.3, 0.3))
        observations.append(
            {
                "frame": frame,
                "entities": {
                    "actor_a": {
                        "location": [0.0, 0.0, 1.0],
                        "world_bounds": actor_bounds,
                        "pose_bones": {"hand.R": {"head": [1.0, 0.0, 1.0], "tail": [1.1, 0.0, 1.0]}},
                    },
                    "actor_b": {
                        "location": [2.0, 0.0, 1.0],
                        "world_bounds": _box([2.0, 0.0, 1.0], (0.3, 0.3, 1.0)),
                        "pose_bones": {"hand.L": {"head": receiver_hand, "tail": [receiver_hand[0] + 0.1, receiver_hand[1], receiver_hand[2]]}},
                    },
                    "prop_01": {"location": prop_center, "world_bounds": prop_bounds},
                },
            }
        )
    return {"schema_version": "trusted-observer-v1", "fps": 1, "observations": observations}


def test_physics_oracle_accepts_raw_contact_and_ownership_evidence() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    report = evaluate_physics_oracle(_record(), _telemetry())

    assert report["status"] == "pass"
    assert report["metrics"]["interaction_contact_rate"] == 1.0
    assert not any(item["failure_id"] == "physics_penetration" for item in report["findings"])


def test_physics_oracle_detects_penetration_from_raw_bounds_not_claimed_flag() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    telemetry = _telemetry(embedded=True)
    telemetry["attachment_penetration"] = []
    report = evaluate_physics_oracle(_record(), telemetry)

    assert report["status"] == "fail"
    assert any(item["failure_id"] == "physics_penetration" for item in report["findings"])


def test_physics_oracle_detects_wrong_receiver_and_missing_contact() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    report = evaluate_physics_oracle(_record(), _telemetry(receiver_close=False))

    assert report["status"] == "fail"
    ids = {item["failure_id"] for item in report["findings"]}
    assert "interaction_receiver_contact_missing" in ids
    assert "interaction_final_owner_unobserved" in ids


def test_physics_oracle_detects_teleport_and_never_uses_plan_only_to_pass() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    report = evaluate_physics_oracle(_record(), _telemetry(teleport=True))

    assert report["status"] == "fail"
    assert any(item["failure_id"] == "physics_teleport" for item in report["findings"])


def test_physics_oracle_fails_closed_when_raw_observations_are_missing() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    report = evaluate_physics_oracle(_record(), {"schema_version": "trusted-observer-v1", "observations": []})

    assert report["status"] == "unavailable"
    assert any(item["failure_id"] == "physics_evidence_missing" for item in report["findings"])


def _mesh_telemetry(prop_triangle):
    broad_bounds = {"min": [0.0, 0.0, -0.1], "max": [1.8, 1.8, 0.1]}
    return {
        "schema_version": "trusted-observer-v1",
        "fps": 1,
        "observations": [
            {
                "frame": 1,
                "entities": {
                    "actor_a": {
                        "world_bounds": broad_bounds,
                        "mesh_triangles": [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
                    },
                    "prop_01": {
                        "world_bounds": broad_bounds,
                        "mesh_triangles": [prop_triangle],
                    },
                },
            }
        ],
    }


def test_mesh_bvh_avoids_false_positive_from_overlapping_broad_bounds() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    record = {
        "entities": [{"id": "actor_a", "kind": "actor"}, {"id": "prop_01", "kind": "prop"}],
        "negative_constraints": ["no_prop_penetration"],
    }
    disjoint_triangle = [[0.8, 0.8, 0.0], [1.8, 0.8, 0.0], [0.8, 1.8, 0.0]]

    report = evaluate_physics_oracle(record, _mesh_telemetry(disjoint_triangle))

    assert report["status"] == "pass"
    assert report["metrics"]["mesh_bvh_queries"] == 1
    assert report["metrics"]["mesh_bvh_intersections"] == 0


def test_mesh_bvh_catches_actual_triangle_intersection() -> None:
    from evaluator.physics_oracle import evaluate_physics_oracle

    record = {
        "entities": [{"id": "actor_a", "kind": "actor"}, {"id": "prop_01", "kind": "prop"}],
        "negative_constraints": ["no_prop_penetration"],
    }
    intersecting_triangle = [[0.1, 0.1, 0.0], [0.7, 0.1, 0.0], [0.1, 0.7, 0.0]]

    report = evaluate_physics_oracle(record, _mesh_telemetry(intersecting_triangle))

    assert report["status"] == "fail"
    assert report["metrics"]["mesh_bvh_queries"] == 1
    assert report["metrics"]["mesh_bvh_intersections"] == 1
    assert any(item["failure_id"] == "physics_penetration" for item in report["findings"])
