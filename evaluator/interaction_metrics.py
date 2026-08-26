"""Independent interaction and attachment lifecycle checks."""

from __future__ import annotations

import math
from typing import Any

from videoact.contracts import Finding, TrajectoryPlan
from videoact.director_contracts import DirectorPlan


def _distance(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _state_at(trajectory_plan: TrajectoryPlan, entity_id: str, frame: int):
    trajectory = trajectory_plan.entities.get(entity_id)
    if trajectory is None or not trajectory.states:
        return None
    return min(trajectory.states, key=lambda state: abs(state.frame - frame))


def _finding(
    failure_id: str,
    message: str,
    *,
    root_cause_id: str,
    evidence: list[str],
) -> Finding:
    return Finding(
        failure_id=failure_id,
        owner="director_trajectory",
        category="interaction_lifecycle",
        severity="hard",
        message=message,
        root_cause_id=root_cause_id,
        evidence=evidence,
        repair_route="trajectory_repair",
    )


def evaluate_interactions(
    director_plan: DirectorPlan,
    trajectory_plan: TrajectoryPlan,
    *,
    telemetry: dict[str, Any] | None = None,
) -> list[Finding]:
    findings: list[Finding] = []
    events = {event.id: event for event in director_plan.events}
    telemetry = telemetry or {}
    owner_by_event = telemetry.get("current_owner_by_event") or {}

    for lifecycle in director_plan.interactions:
        transfer_id = lifecycle.transfer_event_id or lifecycle.detach_event_id
        root = f"attachment_lifecycle:{lifecycle.receiver_id or lifecycle.giver_id}:{lifecycle.prop_id}:{transfer_id}"
        trajectory = trajectory_plan.entities.get(lifecycle.prop_id)
        if trajectory is None:
            findings.append(
                _finding(
                    "interaction_trajectory_missing",
                    f"interaction prop {lifecycle.prop_id} has no trajectory",
                    root_cause_id=root,
                    evidence=[lifecycle.prop_id],
                )
            )
            continue
        event = events.get(lifecycle.transfer_event_id) if lifecycle.transfer_event_id else None
        attachments = list(trajectory.attachment_events)
        attach = next((item for item in attachments if item.action == "attach"), None)
        transfer = next((item for item in attachments if item.action == "transfer"), None)
        detach = next((item for item in attachments if item.action == "detach"), None)
        carry_only = (
            lifecycle.transfer_event_id is None
            and lifecycle.receiver_id is None
        )
        if carry_only:
            if attach is None:
                findings.append(
                    _finding(
                        "interaction_attach_missing",
                        f"independent carry {lifecycle.id} has no attach evidence",
                        root_cause_id=root,
                        evidence=[lifecycle.id, "attach"],
                    )
                )
            continue
        if attach is None or transfer is None or detach is None or event is None:
            findings.append(
                _finding(
                    "interaction_handoff_incomplete",
                    f"interaction {lifecycle.id} lacks a complete attach/transfer/detach lifecycle",
                    root_cause_id=root,
                    evidence=[lifecycle.id, "attach", "transfer", "detach"],
                )
            )
            continue

        start_frame = round(event.start * trajectory_plan.timebase.fps) + 1
        end_frame = round(event.end * trajectory_plan.timebase.fps) + 1
        if not start_frame <= transfer.frame <= end_frame:
            findings.append(
                _finding(
                    "interaction_transfer_outside_window",
                    f"transfer for {lifecycle.prop_id} is outside the handoff window",
                    root_cause_id=root,
                    evidence=[lifecycle.prop_id, str(transfer.frame), str(start_frame), str(end_frame)],
                )
            )

        receiver_id = lifecycle.receiver_id or lifecycle.final_owner_id
        if receiver_id and transfer.object_id != receiver_id:
            findings.append(
                _finding(
                    "interaction_final_owner_mismatch",
                    f"handoff records {transfer.object_id} but DirectorPlan expects {receiver_id}",
                    root_cause_id=root,
                    evidence=[receiver_id, transfer.object_id],
                )
            )

        telemetry_key = f"{lifecycle.transfer_event_id}:{lifecycle.prop_id}"
        observed_owner = owner_by_event.get(telemetry_key)
        if observed_owner and lifecycle.final_owner_id and observed_owner != lifecycle.final_owner_id:
            findings.append(
                _finding(
                    "interaction_final_owner_mismatch",
                    f"telemetry records final owner {observed_owner}, expected {lifecycle.final_owner_id}",
                    root_cause_id=root,
                    evidence=[lifecycle.final_owner_id, observed_owner],
                )
            )

        if attach.object_id not in trajectory_plan.entities:
            findings.append(
                _finding(
                    "interaction_giver_missing",
                    f"attach event references unknown giver {attach.object_id}",
                    root_cause_id=root,
                    evidence=[attach.object_id],
                )
            )
            continue
        subject = _state_at(trajectory_plan, lifecycle.prop_id, attach.frame)
        giver = _state_at(trajectory_plan, attach.object_id, attach.frame)
        if subject is None or giver is None:
            contact = float("inf")
        else:
            hand = tuple(
                coordinate + offset
                for coordinate, offset in zip(giver.position, (0.65, -0.05, 1.35))
            )
            contact = min(_distance(subject.position, giver.position), _distance(subject.position, hand))
        if contact > 0.5:
            findings.append(
                _finding(
                    "interaction_contact_missing",
                    f"interaction {lifecycle.id} has no contact evidence at attach",
                    root_cause_id=root,
                    evidence=[lifecycle.prop_id, attach.object_id, str(attach.frame)],
                )
            )

        if lifecycle.final_support_id and detach.object_id != lifecycle.final_support_id:
            findings.append(
                _finding(
                    "interaction_final_support_mismatch",
                    f"detach records support {detach.object_id}, expected {lifecycle.final_support_id}",
                    root_cause_id=root,
                    evidence=[lifecycle.final_support_id, detach.object_id],
                )
            )
    return findings
