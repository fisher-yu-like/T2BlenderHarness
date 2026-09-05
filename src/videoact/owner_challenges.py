"""Train-only owner challenge fixtures and validator."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing_extensions import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


OWNER_CHALLENGE_SCHEMA_VERSION = "owner-challenge-v1"
CHALLENGE_OWNERS = (
    "director_prompt_interpreter",
    "director_event_scheduler",
    "director_trajectory",
    "director_camera",
    "blender_code_agent",
    "blender_executor",
    "interaction_library",
)
ChallengePolarity = Literal["positive", "negative"]


class OwnerChallengeFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = OWNER_CHALLENGE_SCHEMA_VERSION
    fixture_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    polarity: ChallengePolarity
    case_id: str = Field(min_length=1)
    split: Literal["train"] = "train"
    source_family: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    expected_first_divergence_owner: str = Field(min_length=1)
    expected_first_divergence_stage: str = Field(min_length=1)
    injected_fault: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identity(self) -> "OwnerChallengeFixture":
        if self.owner not in CHALLENGE_OWNERS:
            raise ValueError(f"unknown owner challenge owner: {self.owner}")
        if self.expected_first_divergence_owner != self.owner:
            raise ValueError("fixture expected owner must equal challenge owner")
        return self


class OwnerChallengeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = OWNER_CHALLENGE_SCHEMA_VERSION
    status: Literal["pass", "failed"]
    owner_count: int = Field(ge=0)
    fixture_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)
    pairs_per_owner: dict[str, int] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)


def _normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.casefold()).strip()


def _pair_prompt(owner: str, index: int, polarity: str) -> str:
    fault = {
        "director_prompt_interpreter": "preserve the named participants and explicit action order",
        "director_event_scheduler": "preserve the before-after dependency and timing window",
        "director_trajectory": "keep the object trajectory and support contact continuous",
        "director_camera": "keep the required event and all named targets visible",
        "blender_code_agent": "implement every required event with the declared primitive",
        "blender_executor": "produce a fresh artifact with matching run provenance",
        "interaction_library": "transfer ownership and end on the declared support",
    }[owner]
    action = "satisfies" if polarity == "positive" else "violates"
    return f"Owner challenge {owner} variant {index}: the scene {action} the contract to {fault}."


def build_default_challenge_set(*, pairs_per_owner: int = 5) -> list[OwnerChallengeFixture]:
    if pairs_per_owner <= 0:
        raise ValueError("pairs_per_owner must be positive")
    stages = {
        "director_prompt_interpreter": "planned",
        "director_event_scheduler": "planned",
        "director_trajectory": "planned",
        "director_camera": "visible",
        "blender_code_agent": "implemented",
        "blender_executor": "runtime_execution",
        "interaction_library": "runtime_execution",
    }
    result: list[OwnerChallengeFixture] = []
    for owner in CHALLENGE_OWNERS:
        for index in range(1, pairs_per_owner + 1):
            pair_id = f"{owner}:pair:{index:02d}"
            for polarity in ("positive", "negative"):
                fixture_id = f"{pair_id}:{polarity}"
                result.append(
                    OwnerChallengeFixture(
                        fixture_id=fixture_id,
                        pair_id=pair_id,
                        owner=owner,
                        polarity=polarity,
                        case_id=f"owner-challenge-{hashlib.sha256(fixture_id.encode()).hexdigest()[:12]}",
                        source_family=f"owner:{owner}",
                        prompt=_pair_prompt(owner, index, polarity),
                        expected_first_divergence_owner=owner,
                        expected_first_divergence_stage=stages[owner],
                        injected_fault=("none" if polarity == "positive" else f"inject_{owner}_{index:02d}"),
                    )
                )
    return result


def validate_owner_challenges(
    fixtures: Iterable[OwnerChallengeFixture | dict],
    *,
    min_pairs_per_owner: int = 5,
) -> OwnerChallengeReport:
    values = [item if isinstance(item, OwnerChallengeFixture) else OwnerChallengeFixture.model_validate(item) for item in fixtures]
    failures: list[str] = []
    ids = [item.fixture_id for item in values]
    if len(ids) != len(set(ids)):
        failures.append("duplicate fixture IDs")
    if any(item.split != "train" for item in values):
        failures.append("owner challenge fixtures must be train-only")
    prompts: dict[str, str] = {}
    for item in values:
        normalized = _normalize_prompt(item.prompt)
        prior = prompts.get(normalized)
        if prior is not None and prior != item.fixture_id:
            failures.append(f"prompt collision: {prior} vs {item.fixture_id}")
        prompts[normalized] = item.fixture_id
    pairs: dict[str, set[str]] = defaultdict(set)
    owner_pairs: dict[str, set[str]] = defaultdict(set)
    for item in values:
        pairs[item.pair_id].add(item.polarity)
        owner_pairs[item.owner].add(item.pair_id)
    for pair_id, polarities in pairs.items():
        if polarities != {"positive", "negative"}:
            failures.append(f"positive/negative pair incomplete: {pair_id}")
    for owner in CHALLENGE_OWNERS:
        count = len(owner_pairs.get(owner, set()))
        if count < min_pairs_per_owner:
            failures.append(f"owner {owner} has {count} pairs; requires {min_pairs_per_owner}")
    report = OwnerChallengeReport(
        status="pass" if not failures else "failed",
        owner_count=len({item.owner for item in values}),
        fixture_count=len(values),
        pair_count=len(pairs),
        pairs_per_owner={owner: len(owner_pairs.get(owner, set())) for owner in CHALLENGE_OWNERS},
        failures=failures,
    )
    if failures:
        raise ValueError("; ".join(failures))
    return report


def write_owner_challenges(path: str | Path, fixtures: Iterable[OwnerChallengeFixture | dict]) -> Path:
    values = [item if isinstance(item, OwnerChallengeFixture) else OwnerChallengeFixture.model_validate(item) for item in fixtures]
    validate_owner_challenges(values)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(item.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n" for item in values),
        encoding="utf-8",
    )
    return destination


def load_owner_challenges(path: str | Path) -> list[OwnerChallengeFixture]:
    values = [
        OwnerChallengeFixture.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_owner_challenges(values)
    return values


__all__ = [
    "CHALLENGE_OWNERS",
    "OWNER_CHALLENGE_SCHEMA_VERSION",
    "OwnerChallengeFixture",
    "OwnerChallengeReport",
    "build_default_challenge_set",
    "load_owner_challenges",
    "validate_owner_challenges",
    "write_owner_challenges",
]
