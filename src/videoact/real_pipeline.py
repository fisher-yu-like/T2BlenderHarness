"""Host-side state machine for real Blender MCP runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field

from .real_artifacts import RealArtifactGate, RealArtifactReport


RealState = Literal["prepared", "executing", "rendered", "artifact_valid", "evaluated", "failed"]
VALID_TRANSITIONS: dict[str, set[str]] = {
    "prepared": {"executing", "failed"},
    "executing": {"rendered", "failed"},
    "rendered": {"artifact_valid", "failed"},
    "artifact_valid": {"evaluated", "failed"},
    # An explicit rerender may reopen a terminal artifact state.  The history
    # is append-only, so this cannot erase the prior evaluation/failure; the
    # executor still records a fresh render attempt and hashes it separately.
    "evaluated": {"executing"},
    "failed": {"executing"},
}


class RealStateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    state: RealState
    history: list[dict[str, Any]] = Field(default_factory=list)


class RealRunStateMachine:
    def __init__(self, run_dir: str | Path, *, case_id: str):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "state.json"
        if self.state_path.exists():
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.record = RealStateRecord.model_validate(payload)
        else:
            self.record = RealStateRecord(
                case_id=case_id,
                state="prepared",
                history=[{"state": "prepared", "metadata": {}}],
            )
            self._save()

    @property
    def state(self) -> RealState:
        return self.record.state

    def transition(self, next_state: RealState, metadata: dict[str, Any] | None = None) -> None:
        if next_state not in VALID_TRANSITIONS[self.state]:
            raise ValueError(f"invalid transition: {self.state} -> {next_state}")
        self.record.state = next_state
        self.record.history.append({"state": next_state, "metadata": metadata or {}})
        self._save()

    def record_mcp_response(self, response: dict[str, Any]) -> None:
        (self.run_dir / "mcp_response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True), encoding="utf-8"
        )
        if response.get("isError") or response.get("status") in {"failed", "timeout"}:
            if self.state != "failed":
                self.transition("failed", {"mcp_response": response})
        elif self.state == "executing":
            self.transition("rendered", {"mcp_response": response})

    def validate_artifacts(self, gate: RealArtifactGate | None = None) -> RealArtifactReport:
        report = (gate or RealArtifactGate()).validate(self.run_dir)
        if report.artifact_status == "complete":
            if self.state == "rendered":
                self.transition("artifact_valid", {"readable_frames": report.readable_frame_count})
        elif self.state not in {"failed", "evaluated"}:
            self.transition("failed", {"hard_failures": report.hard_failures})
        return report

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self.record.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
