"""Simple deterministic spatial and temporal checks for trajectory plans."""

from __future__ import annotations

import math

from videoact.contracts import Finding, SceneContract, TrajectoryPlan


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _trajectory_owner(plan: TrajectoryPlan) -> str:
    return "director_trajectory" if "multi_entity_collision_free_lanes" in plan.validation_intents else "trajectory_planner"


def _state_at_frame(plan: TrajectoryPlan, entity_id: str, frame: int):
    trajectory = plan.entities.get(entity_id)
    if trajectory is None or not trajectory.states:
        return None
    return min(trajectory.states, key=lambda state: abs(state.frame - frame))


def check_support_before_grasp(contract: SceneContract, plan: TrajectoryPlan) -> list[Finding]:
    if "support_before_grasp" not in contract.physics_constraints:
        return []
    grasp = next((event for event in contract.events if event.id == "grasp"), None)
    target = next((entity for entity in contract.entities if entity.role == "target_object"), None)
    if grasp is None or target is None:
        return []
    support_relation = next(
        (
            relation
            for relation in contract.relations
            if relation.type == "on" and relation.subject == target.id
        ),
        None,
    )
    attachments = [
        attachment
        for trajectory in plan.entities.values()
        for attachment in trajectory.attachment_events
        if attachment.subject_id == target.id and attachment.action == "attach"
    ]
    expected_frame = round(grasp.start * contract.fps) + 1
    if support_relation is None or not attachments or attachments[0].frame < expected_frame:
        return [
            Finding(
                failure_id="support_before_grasp",
                owner=_trajectory_owner(plan),
                category="support_relation",
                severity="hard",
                root_cause_id="support_grasp_causality",
                message="target must remain supported until the grasp begins",
                repair_route="trajectory_repair",
            )
        ]
    return []


def check_attachment_contact(plan: TrajectoryPlan) -> list[Finding]:
    for owner_id, trajectory in plan.entities.items():
        for attachment in trajectory.attachment_events:
            if attachment.action != "attach":
                continue
            subject = _state_at_frame(plan, attachment.subject_id, attachment.frame)
            owner = _state_at_frame(plan, attachment.object_id, attachment.frame)
            if (
                attachment.constraint_type == "child_of"
                and attachment.subtarget in {"hand.L", "hand.R"}
                and owner is not None
            ):
                # New Director plans bind the prop to an articulated hand. The
                # executable Blender constraint, not a guessed root offset, is
                # the source of truth for contact.
                continue
            contact = _distance(subject.position, owner.position) if subject is not None and owner is not None else float("inf")
            if (
                contact > 0.5
                and "multi_entity_collision_free_lanes" in plan.validation_intents
                and owner is not None
                and subject is not None
            ):
                # Director trajectories attach props to an explicit hand lane,
                # while the legacy evaluator observes the actor root. Accept
                # the declared proxy hand offset only for Director plans.
                hand_position = tuple(
                    coordinate + offset
                    for coordinate, offset in zip(owner.position, (0.65, -0.05, 1.35))
                )
                contact = _distance(subject.position, hand_position)
            if subject is None or owner is None or contact > 0.5:
                return [
                    Finding(
                        failure_id="attachment_without_contact",
                        owner=_trajectory_owner(plan),
                        category="attachment_relation",
                        severity="hard",
                        root_cause_id=f"attachment_contact:{attachment.subject_id}:{attachment.object_id}",
                        message=f"attachment {attachment.subject_id}->{attachment.object_id} has no contact",
                        repair_route="trajectory_repair",
                    )
                ]
    return []


def check_velocity_continuity(plan: TrajectoryPlan, *, max_velocity: float = 10.0) -> list[Finding]:
    for entity_id, trajectory in plan.entities.items():
        for previous, current in zip(trajectory.states, trajectory.states[1:]):
            seconds = (current.frame - previous.frame) / plan.timebase.fps
            if seconds <= 0:
                continue
            velocity = _distance(previous.position, current.position) / seconds
            if velocity > max_velocity:
                return [
                    Finding(
                        failure_id="velocity_spike",
                        owner="trajectory_planner",
                        category="temporal_smoothness",
                        severity="error",
                        root_cause_id=f"velocity_continuity:{entity_id}",
                        message=f"entity {entity_id} velocity {velocity:.2f} exceeds {max_velocity:.2f}",
                        evidence=[entity_id],
                        repair_route="trajectory_repair",
                    )
                ]
    return []
