from __future__ import annotations

from pathlib import Path


def test_two_plan_audit_tracks_scope_statuses_and_human_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "two-plan-convergence-audit-v1.md").read_text(encoding="utf-8")

    for marker in (
        "E:/2026-08-27-harness-quality-remediation.md",
        "E:/2026-08-27-agent-codegen-layered-evolution.md",
        "complete",
        "partial",
        "pending_human",
        "pending_external",
        "not_applicable",
        "60 train",
        "60 dev",
        "20 frozen",
        "1320",
        "golden review",
        "provider",
        "template",
    ):
        assert marker in text
