"""Strict schemas for optional VLM judge responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VLMJudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_compliance: float = Field(ge=0, le=100)
    physical_plausibility: float = Field(ge=0, le=100)
    camera_coverage: float = Field(ge=0, le=100)
    camera_innovation: float = Field(ge=0, le=100)
    character_trajectory: float = Field(ge=0, le=100)
    object_trajectory: float = Field(ge=0, le=100)
    event_timing: float = Field(ge=0, le=100)
    temporal_smoothness: float = Field(ge=0, le=100)
    visual_clarity: float = Field(ge=0, le=100)
    # Optional in the compatibility schema so old review artifacts remain
    # readable; the unified review request asks for all five explicitly.
    appearance_detail: float | None = Field(default=None, ge=0, le=100)
    physical_realism: float | None = Field(default=None, ge=0, le=100)
    spatial_consistency: float | None = Field(default=None, ge=0, le=100)
    motion_naturalness: float | None = Field(default=None, ge=0, le=100)
    visual_presentation: float | None = Field(default=None, ge=0, le=100)
    visible_evidence: list[str] = Field(min_length=1)
    weaknesses: list[str]
    confidence: float = Field(ge=0, le=1)
