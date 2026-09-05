"""Host-side validation and fingerprinting for real Blender proxy runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from typing_extensions import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .experiment_fingerprint import hash_value as _contract_hash


EXPERIMENT_CONTRACT_VERSION = "experiment-contract-v1"
CONTRACT_ARTIFACT_NAMES = (
    "experiment_contract.json",
    "baseline_manifest.json",
    "split_access_policy.json",
    "frozen_component_hashes.json",
)


class ExperimentCaseIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    seed: int


class ExperimentContract(BaseModel):
    """Versioned, immutable identity and policy for one experiment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: str = EXPERIMENT_CONTRACT_VERSION
    experiment_id: str = Field(min_length=1)
    parent_harness_version: str = Field(min_length=1)
    split_cases: dict[str, list[ExperimentCaseIdentity]]
    dataset_fingerprint: str = Field(min_length=1)
    evaluator_fingerprint: Any
    observer_fingerprint: Any
    provider_fingerprint: Any
    blender_binary_fingerprint: str = Field(min_length=1)
    render_settings: dict[str, Any]
    scoring_policy: dict[str, Any]
    judge: dict[str, Any]
    frame_sampler: dict[str, Any]
    acceptance_margin: float
    test_unlock_milestones: list[str] = Field(min_length=1)
    experiment_fingerprint: str = Field(min_length=1)
    frozen_component_hashes: dict[str, str]


def _binary_fingerprint(value: str | Path | bytes) -> str:
    if isinstance(value, bytes):
        return hashlib.sha256(value).hexdigest()
    path = Path(value)
    if path.is_file():
        return _sha256(path)
    return _contract_hash(str(value))


def build_experiment_contract(
    *,
    experiment_id: str,
    parent_harness_version: str,
    split_cases: dict[str, list[dict[str, Any]]],
    dataset_fingerprint: str,
    evaluator_fingerprint: Any,
    observer_fingerprint: Any,
    provider_fingerprint: Any,
    blender_binary_fingerprint: str | Path | bytes,
    render_settings: dict[str, Any],
    scoring_policy: dict[str, Any],
    judge: dict[str, Any],
    frame_sampler: dict[str, Any],
    acceptance_margin: float,
    test_unlock_milestones: list[str],
) -> ExperimentContract:
    """Build all immutable experiment identity from declared inputs only."""

    normalized_cases = {
        split: [ExperimentCaseIdentity.model_validate(item) for item in cases]
        for split, cases in split_cases.items()
    }
    if set(normalized_cases) != {"train", "dev", "test"}:
        raise ValueError("experiment contract requires train, dev, and test case identities")
    all_case_ids = [item.case_id for cases in normalized_cases.values() for item in cases]
    if len(all_case_ids) != len(set(all_case_ids)):
        raise ValueError("experiment contract split case IDs must be disjoint")
    component_values = {
        "dataset": dataset_fingerprint,
        "evaluator": evaluator_fingerprint,
        "observer": observer_fingerprint,
        "provider": provider_fingerprint,
        "blender_binary": _binary_fingerprint(blender_binary_fingerprint),
        "render_settings": render_settings,
        "scoring_policy": scoring_policy,
        "judge": judge,
        "frame_sampler": frame_sampler,
        "acceptance_margin": acceptance_margin,
        "test_unlock_milestones": list(test_unlock_milestones),
        "split_cases": {
            split: [item.model_dump(mode="json") for item in cases]
            for split, cases in normalized_cases.items()
        },
    }
    frozen_hashes = {name: _contract_hash(value) for name, value in component_values.items()}
    identity_payload = {
        "contract_version": EXPERIMENT_CONTRACT_VERSION,
        "experiment_id": experiment_id,
        "parent_harness_version": parent_harness_version,
        "component_hashes": frozen_hashes,
    }
    contract = ExperimentContract(
        experiment_id=experiment_id,
        parent_harness_version=parent_harness_version,
        split_cases=normalized_cases,
        dataset_fingerprint=dataset_fingerprint,
        evaluator_fingerprint=evaluator_fingerprint,
        observer_fingerprint=observer_fingerprint,
        provider_fingerprint=provider_fingerprint,
        blender_binary_fingerprint=_binary_fingerprint(blender_binary_fingerprint),
        render_settings=render_settings,
        scoring_policy=scoring_policy,
        judge=judge,
        frame_sampler=frame_sampler,
        acceptance_margin=acceptance_margin,
        test_unlock_milestones=test_unlock_milestones,
        experiment_fingerprint=_contract_hash(identity_payload),
        frozen_component_hashes=frozen_hashes,
    )
    return contract


def _contract_payload(contract: ExperimentContract | dict[str, Any]) -> dict[str, Any]:
    return contract.model_dump(mode="json") if isinstance(contract, ExperimentContract) else ExperimentContract.model_validate(contract).model_dump(mode="json")


def _component_values(contract: ExperimentContract) -> dict[str, Any]:
    """Return the declared values covered by the frozen component hashes."""

    return {
        "dataset": contract.dataset_fingerprint,
        "evaluator": contract.evaluator_fingerprint,
        "observer": contract.observer_fingerprint,
        "provider": contract.provider_fingerprint,
        "blender_binary": contract.blender_binary_fingerprint,
        "render_settings": contract.render_settings,
        "scoring_policy": contract.scoring_policy,
        "judge": contract.judge,
        "frame_sampler": contract.frame_sampler,
        "acceptance_margin": contract.acceptance_margin,
        "test_unlock_milestones": list(contract.test_unlock_milestones),
        "split_cases": {
            split: [item.model_dump(mode="json") for item in cases]
            for split, cases in contract.split_cases.items()
        },
    }


def validate_experiment_contract(contract: ExperimentContract | dict[str, Any]) -> ExperimentContract:
    """Validate schema, component hashes, and the overall identity digest."""

    model = contract if isinstance(contract, ExperimentContract) else ExperimentContract.model_validate(contract)
    if model.contract_version != EXPERIMENT_CONTRACT_VERSION:
        raise ValueError(f"unsupported experiment contract version: {model.contract_version}")
    expected_components = {
        name: _contract_hash(value) for name, value in _component_values(model).items()
    }
    if expected_components != model.frozen_component_hashes:
        raise ValueError("experiment contract component hash mismatch")
    expected_identity = _contract_hash(
        {
            "contract_version": model.contract_version,
            "experiment_id": model.experiment_id,
            "parent_harness_version": model.parent_harness_version,
            "component_hashes": model.frozen_component_hashes,
        }
    )
    if expected_identity != model.experiment_fingerprint:
        raise ValueError("experiment contract fingerprint mismatch")
    return model


def validate_contract_runtime_inputs(
    contract: ExperimentContract | dict[str, Any],
    *,
    dataset_fingerprint: str | None = None,
    blender_binary: str | Path | bytes | None = None,
    evaluator_fingerprint: Any | None = None,
    observer_fingerprint: Any | None = None,
    provider_fingerprint: Any | None = None,
    render_settings: dict[str, Any] | None = None,
    scoring_policy: dict[str, Any] | None = None,
    judge: dict[str, Any] | None = None,
    frame_sampler: dict[str, Any] | None = None,
    acceptance_margin: float | None = None,
    observed_split_cases: dict[str, list[dict[str, Any]]] | None = None,
) -> ExperimentContract:
    """Bind a loaded contract to the inputs available at formal-run startup."""

    model = validate_experiment_contract(contract)
    mismatches: list[str] = []
    if dataset_fingerprint is not None and str(dataset_fingerprint) != model.dataset_fingerprint:
        mismatches.append("dataset")
    if blender_binary is not None:
        if isinstance(blender_binary, (str, Path)) and not Path(blender_binary).is_file():
            raise ValueError(f"formal contract runtime input is missing: Blender binary {blender_binary}")
        if _binary_fingerprint(blender_binary) != model.blender_binary_fingerprint:
            mismatches.append("blender_binary")
    direct_values = {
        "evaluator": evaluator_fingerprint,
        "observer": observer_fingerprint,
        "provider": provider_fingerprint,
        "render_settings": render_settings,
        "scoring_policy": scoring_policy,
        "judge": judge,
        "frame_sampler": frame_sampler,
        "acceptance_margin": acceptance_margin,
    }
    for name, value in direct_values.items():
        if value is not None and _contract_hash(value) != model.frozen_component_hashes[name]:
            mismatches.append(name)
    if observed_split_cases is not None:
        expected = {
            split: [(item.case_id, item.prompt_hash) for item in cases]
            for split, cases in model.split_cases.items()
        }
        observed: dict[str, list[tuple[str, str]]] = {}
        for split, cases in observed_split_cases.items():
            observed[split] = [
                (
                    str(item.get("case_id") or ""),
                    str(item.get("prompt_hash") or ""),
                )
                for item in cases
                if isinstance(item, dict)
            ]
        for split, values in observed.items():
            if sorted(values) != sorted(expected.get(split, [])):
                mismatches.append("split_cases")
    if mismatches:
        raise ValueError(
            "experiment contract runtime identity mismatch: " + ", ".join(sorted(set(mismatches)))
        )
    return model


def _contract_bundle_documents(model: ExperimentContract) -> dict[str, dict[str, Any]]:
    """Derive the complete immutable bundle from one validated contract."""

    baseline = {
        "contract_version": model.contract_version,
        "experiment_id": model.experiment_id,
        "train": {
            "case_ids": [item.case_id for item in model.split_cases["train"]],
            "prompt_hashes": [item.prompt_hash for item in model.split_cases["train"]],
            "seeds": [item.seed for item in model.split_cases["train"]],
        },
        "dev": {
            "case_ids": [item.case_id for item in model.split_cases["dev"]],
            "prompt_hashes": [item.prompt_hash for item in model.split_cases["dev"]],
            "seeds": [item.seed for item in model.split_cases["dev"]],
        },
        "final_test_evaluation": {
            "case_ids": [item.case_id for item in model.split_cases["test"]],
            "prompt_hashes": [item.prompt_hash for item in model.split_cases["test"]],
            "seeds": [item.seed for item in model.split_cases["test"]],
            "selection_excluded": True,
            "patch_selection_excluded": True,
        },
    }
    policy = {
        "contract_version": model.contract_version,
        "test": {
            "allowed": False,
            "unlock_milestones": model.test_unlock_milestones,
            "selection_excluded": True,
            "patch_selection_excluded": True,
        },
        "train": {"proposal_source_allowed": True, "evidence_allowed": True},
        "dev": {"proposal_source_allowed": False, "evidence_allowed": True},
    }
    frozen = {
        "contract_version": model.contract_version,
        "experiment_fingerprint": model.experiment_fingerprint,
        "component_hashes": model.frozen_component_hashes,
    }
    return {
        "experiment_contract.json": model.model_dump(mode="json"),
        "baseline_manifest.json": baseline,
        "split_access_policy.json": policy,
        "frozen_component_hashes.json": frozen,
    }


def write_experiment_contract_bundle(contract: ExperimentContract | dict[str, Any], destination: str | Path) -> dict[str, Path]:
    """Write the contract and its derived manifests without adding new identity."""

    model = validate_experiment_contract(contract)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    documents = _contract_bundle_documents(model)
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = root / name
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"existing contract artifact is unreadable and cannot be replaced: {path}") from exc
            if existing != document:
                raise ValueError(f"experiment contract artifacts are immutable: {path.name}")
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[name.removesuffix(".json")] = path
    return paths


def load_experiment_contract(path: str | Path) -> ExperimentContract:
    candidate = Path(path)
    root = Path(path)
    if root.is_dir():
        candidate = root / "experiment_contract.json"
    else:
        candidate = root
        root = candidate.parent
    if not candidate.is_file():
        raise ValueError(f"experiment contract is missing: {candidate}")
    try:
        contract = validate_experiment_contract(json.loads(candidate.read_text(encoding="utf-8")))
        expected_documents = _contract_bundle_documents(contract)
        for name, expected in expected_documents.items():
            artifact = root / name
            if not artifact.is_file():
                raise ValueError(f"experiment contract bundle artifact missing: {name}")
            actual = json.loads(artifact.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(f"experiment contract bundle artifact mismatch: {name}")
        return contract
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"experiment contract is unreadable or invalid: {candidate}: {exc}") from exc


def compare_experiment_contracts(first: ExperimentContract | dict[str, Any], second: ExperimentContract | dict[str, Any], *, require_compatible: bool = False) -> dict[str, Any]:
    left = validate_experiment_contract(_contract_payload(first))
    right = validate_experiment_contract(_contract_payload(second))
    mismatches = sorted(
        name
        for name in set(left.frozen_component_hashes) | set(right.frozen_component_hashes)
        if left.frozen_component_hashes.get(name) != right.frozen_component_hashes.get(name)
    )
    if left.contract_version != right.contract_version:
        mismatches.append("contract_version")
    if left.experiment_id != right.experiment_id:
        mismatches.append("experiment_id")
    if left.parent_harness_version != right.parent_harness_version:
        mismatches.append("parent_harness_version")
    mismatches = sorted(set(mismatches))
    result = {"compatible": not mismatches, "mismatches": mismatches, "first_fingerprint": left.experiment_fingerprint, "second_fingerprint": right.experiment_fingerprint}
    if require_compatible and mismatches:
        raise ValueError(f"incompatible experiment contracts: {', '.join(mismatches)}")
    return result


def validate_proposal_split_access(proposal: dict[str, Any], contract: ExperimentContract | dict[str, Any]) -> None:
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be an object")
    model = validate_experiment_contract(_contract_payload(contract))
    if str(proposal.get("source_split") or "").lower() != "train":
        raise ValueError("proposal source split must be train")
    test_ids = {item.case_id for item in model.split_cases["test"]}
    raw_ids = proposal.get("source_case_ids", [])
    if not isinstance(raw_ids, list):
        raise ValueError("proposal source_case_ids must be a list")
    if test_ids & {str(item) for item in raw_ids}:
        raise ValueError("test cases cannot enter proposal or controller inputs")
    serialized = json.dumps(proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    test_hashes = {item.prompt_hash for item in model.split_cases["test"]}
    if any(prompt_hash in serialized for prompt_hash in test_hashes):
        raise ValueError("test prompt hashes cannot enter proposal or controller inputs")
    if any(case_id in serialized for case_id in test_ids):
        raise ValueError("test case IDs cannot enter proposal or controller inputs")


def validate_contract_revision(previous: ExperimentContract | dict[str, Any], revised: ExperimentContract | dict[str, Any], *, test_unlocked: bool) -> dict[str, Any]:
    left = validate_experiment_contract(_contract_payload(previous))
    right = validate_experiment_contract(_contract_payload(revised))
    changed = sorted(name for name in set(left.frozen_component_hashes) | set(right.frozen_component_hashes) if left.frozen_component_hashes.get(name) != right.frozen_component_hashes.get(name))
    threshold_changes = [name for name in changed if name in {"scoring_policy", "judge", "frame_sampler", "acceptance_margin"}]
    if threshold_changes and not test_unlocked:
        raise ValueError(f"frozen threshold or evaluation policy cannot change before test unlock: {threshold_changes}")
    return {"compatible": not changed, "changed_fields": changed, "changed_threshold_fields": threshold_changes}


class RealRunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    split: Literal["calibration", "train", "dev", "test"]
    prompt_hash: str
    plan_hash: str
    director_plan_hash: str | None = None
    code_hash: str | None = None
    rollout_seed: int | None = None
    harness_version: str
    evaluator_version: str
    blender_version: str
    fps: int = Field(gt=0)
    frame_start: int = Field(ge=1)
    frame_end: int = Field(ge=1)
    render_settings: dict[str, Any]
    fingerprint: str
    state: Literal["prepared", "executing", "rendered", "artifact_valid", "evaluated", "failed"]
    trusted_observer_required: bool = False
    observer_version: str | None = None
    observer_source_hash: str | None = None
    experiment_fingerprint: dict[str, Any] | None = None
    obligation_ids: list[str] = Field(default_factory=list)


class RealArtifactReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_status: Literal["complete", "incomplete"]
    hard_failures: list[str] = Field(default_factory=list)
    readable_frame_count: int = Field(ge=0)
    video_frame_count: int = Field(default=0, ge=0)
    video_fps: float = Field(default=0.0, ge=0)
    video_duration_s: float = Field(default=0.0, ge=0)
    artifact_hashes: dict[str, str] = Field(default_factory=dict)
    artifact_hash: str | None = None
    manifest: RealRunManifest | None = None


REQUIRED_ARTIFACTS = (
    "run_manifest.json",
    "scene_contract.json",
    "trajectory.json",
    "camera_plan.json",
    "blender_job.py",
    "proxy.blend",
    "proxy.mp4",
    "telemetry.json",
    "frames/index.json",
)


def sample_frame_paths(run_dir: str | Path, *, max_frames: int | None = None) -> list[Path]:
    frames_dir = Path(run_dir) / "frames"
    index_path = frames_dir / "index.json"
    indexed: list[Path] = []
    if index_path.exists():
        try:
            entries = json.loads(index_path.read_text(encoding="utf-8")).get("frames", [])
            indexed = [frames_dir / str(entry["path"]) for entry in entries]
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            indexed = []
    if max_frames is None:
        return indexed or sorted(frames_dir.glob("*.png"))
    animation = sorted((frames_dir / "animation").glob("frame_*.png"))
    by_name: dict[str, Path] = {}
    direct = sorted(frames_dir.glob("frame_*.png"))
    for path in indexed + animation + direct:
        by_name.setdefault(path.name, path)
    candidates = sorted(
        by_name.values(),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]) if "_" in path.stem else path.name,
    )
    if len(candidates) <= max_frames:
        return candidates
    indices = {
        round(index * (len(candidates) - 1) / max(1, max_frames - 1))
        for index in range(max_frames)
    }
    return [candidates[index] for index in sorted(indices)]


def sample_event_aligned_frame_paths(
    run_dir: str | Path,
    scene_contract: Any,
    *,
    max_frames: int = 8,
) -> list[Path]:
    """Prefer event midpoints, then add endpoints and uniform timeline coverage."""
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    root = Path(run_dir)
    frames_dir = root / "frames"
    animation = sorted((frames_dir / "animation").glob("frame_*.png"))
    candidates = animation or sample_frame_paths(root, max_frames=None)
    if not candidates:
        return []

    def frame_number(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[-1])

    by_frame = {frame_number(path): path for path in candidates}
    frame_numbers = sorted(by_frame)
    if hasattr(scene_contract, "model_dump"):
        scene_contract = scene_contract.model_dump(mode="json")
    contract = scene_contract if isinstance(scene_contract, dict) else {}
    fps = int(contract.get("fps") or 24)
    events = contract.get("events") or []
    must_show = set(contract.get("must_show") or [event.get("id") for event in events])

    event_targets: list[int] = []
    for event in events:
        if event.get("id") not in must_show:
            continue
        midpoint_s = (float(event.get("start", 0.0)) + float(event.get("end", 0.0))) / 2.0
        event_targets.append(max(1, round(midpoint_s * fps) + 1))

    selected: list[int] = []
    for target in event_targets:
        nearest = min(frame_numbers, key=lambda value: abs(value - target))
        if nearest not in selected:
            selected.append(nearest)
        if len(selected) >= max(0, max_frames - 2):
            break
    for endpoint in (frame_numbers[0], frame_numbers[-1]):
        if endpoint not in selected:
            selected.append(endpoint)
    if len(selected) < max_frames:
        uniform_indices = {
            round(index * (len(frame_numbers) - 1) / max(1, max_frames - 1))
            for index in range(max_frames)
        }
        for index in sorted(uniform_indices):
            number = frame_numbers[index]
            if number not in selected:
                selected.append(number)
            if len(selected) >= max_frames:
                break
    return [by_frame[number] for number in sorted(selected[:max_frames])]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate_artifact_hash(artifact_hashes: dict[str, str]) -> str | None:
    """Hash the canonical path-to-content hash map for one real run."""

    if not artifact_hashes:
        return None
    encoded = json.dumps(
        sorted(artifact_hashes.items()),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probe_mp4(path: str | Path, *, minimum_frames: int = 3) -> dict[str, Any]:
    """Decode enough of an MP4 to prove it is playable, not merely non-empty."""
    video = Path(path)
    if not video.is_file() or video.stat().st_size == 0:
        return {"playable": False, "frame_count": 0, "fps": 0.0, "duration_s": 0.0, "error": "missing_or_empty"}
    try:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
        metadata = next(reader)
        frame_count = 0
        try:
            for _frame in reader:
                frame_count += 1
                if frame_count >= minimum_frames:
                    break
        finally:
            close = getattr(reader, "close", None)
            if close:
                close()
        return {
            "playable": frame_count >= minimum_frames,
            "frame_count": frame_count,
            "fps": float(metadata.get("fps") or 0.0),
            "duration_s": float(metadata.get("duration") or 0.0),
            "error": None if frame_count >= minimum_frames else "insufficient_decoded_frames",
        }
    except (ImportError, OSError, RuntimeError, ValueError, StopIteration) as exc:
        return {
            "playable": False,
            "frame_count": 0,
            "fps": 0.0,
            "duration_s": 0.0,
            "error": f"{type(exc).__name__}:{exc}",
        }


class RealArtifactGate:
    def __init__(self, *, minimum_readable_frames: int = 3):
        self.minimum_readable_frames = minimum_readable_frames

    def validate(self, run_dir: str | Path) -> RealArtifactReport:
        root = Path(run_dir)
        failures: list[str] = []
        hashes: dict[str, str] = {}
        manifest = None
        for relative in REQUIRED_ARTIFACTS:
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing_artifact:{relative}")
            else:
                hashes[relative] = _sha256(path)

        manifest_path = root / "run_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = RealRunManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                failures.append(f"invalid_manifest:{type(exc).__name__}")

        if manifest is not None and manifest.trusted_observer_required:
            for relative in ("candidate.blend", "observer_request.json", "telemetry_manifest.json", "experiment_fingerprint.json"):
                path = root / relative
                if not path.is_file() or path.stat().st_size == 0:
                    failures.append(f"missing_trusted_observer_artifact:{relative}")
                else:
                    hashes[relative] = _sha256(path)
            candidate = root / "candidate.blend"
            proxy = root / "proxy.blend"
            if candidate.is_file() and proxy.is_file() and _sha256(candidate) != _sha256(proxy):
                failures.append("candidate_proxy_blend_hash_mismatch")
            try:
                from .observer_contract import read_trusted_observer_output

                observer_report = read_trusted_observer_output(
                    root,
                    observer_source_path=Path(__file__).resolve().parents[2] / "blender" / "trusted_observer.py",
                )
                if observer_report.get("status") != "pass":
                    failures.extend(
                        f"trusted_observer:{item}"
                        for item in observer_report.get("failures", ["observer_validation_failed"])
                    )
                elif manifest.observer_source_hash and (
                    observer_report.get("manifest") or {}
                ).get("observer_source_hash") != manifest.observer_source_hash:
                    failures.append("trusted_observer:run_manifest_source_hash_mismatch")
            except (OSError, RuntimeError, ValueError, TypeError):
                failures.append("trusted_observer:validation_error")
            fingerprint_payload = manifest.experiment_fingerprint
            if not isinstance(fingerprint_payload, dict):
                failures.append("missing_experiment_fingerprint")
            else:
                try:
                    from .experiment_fingerprint import ExperimentFingerprint

                    fingerprint = ExperimentFingerprint.model_validate(fingerprint_payload)
                    if fingerprint.with_digest().digest != fingerprint.digest:
                        failures.append("experiment_fingerprint_digest_mismatch")
                    fingerprint_path = root / "experiment_fingerprint.json"
                    if fingerprint_path.is_file():
                        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
                        if stored != fingerprint.model_dump(mode="json"):
                            failures.append("experiment_fingerprint_file_mismatch")
                except (TypeError, ValueError):
                    failures.append("invalid_experiment_fingerprint")
                except (OSError, json.JSONDecodeError):
                    failures.append("experiment_fingerprint_file_unreadable")

        director_plan_path = root / "director_plan.json"
        if director_plan_path.is_file():
            hashes["director_plan.json"] = _sha256(director_plan_path)
            if manifest is not None and manifest.director_plan_hash:
                try:
                    director_payload = json.loads(director_plan_path.read_text(encoding="utf-8"))
                    # Canonicalize exactly like DirectorPlan.content_hash()
                    # (ensure_ascii=False); the default escaping diverged on
                    # non-ASCII prompts and broke the hash chain.
                    encoded = json.dumps(
                        director_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    actual_hash = hashlib.sha256(encoded).hexdigest()
                    if actual_hash != manifest.director_plan_hash:
                        failures.append("director_plan_hash_mismatch")
                except (OSError, ValueError, json.JSONDecodeError):
                    failures.append("invalid_director_plan")
        elif manifest is not None and manifest.director_plan_hash:
            failures.append("missing_artifact:director_plan.json")

        if manifest is not None and manifest.code_hash:
            actual_code_hash = hashes.get("blender_job.py")
            if actual_code_hash != manifest.code_hash:
                failures.append("job_source_hash_mismatch")

        readable = 0
        for frame_path in sample_frame_paths(root):
            try:
                with Image.open(frame_path) as image:
                    image.verify()
                readable += 1
                relative = str(frame_path.relative_to(root)).replace("\\", "/")
                hashes[relative] = _sha256(frame_path)
            except (OSError, ValueError):
                failures.append(f"unreadable_frame:{frame_path.name}")
        if readable < self.minimum_readable_frames:
            failures.append(f"insufficient_readable_frames:{readable}")

        video_probe = probe_mp4(root / "proxy.mp4", minimum_frames=self.minimum_readable_frames)
        if not video_probe["playable"]:
            failures.append("unplayable_video:proxy.mp4")

        return RealArtifactReport(
            artifact_status="complete" if not failures else "incomplete",
            hard_failures=sorted(set(failures)),
            readable_frame_count=readable,
            video_frame_count=int(video_probe["frame_count"]),
            video_fps=float(video_probe["fps"]),
            video_duration_s=float(video_probe["duration_s"]),
            artifact_hashes=hashes,
            artifact_hash=aggregate_artifact_hash(hashes),
            manifest=manifest,
        )


def fingerprint_real_run(
    *,
    prompt_hash: str,
    plan_hash: str,
    director_plan_hash: str | None = None,
    harness_version: str,
    evaluator_version: str,
    blender_version: str,
    render_settings: dict[str, Any],
    rollout_seed: int | None = None,
) -> str:
    payload = {
        "prompt_hash": prompt_hash,
        "plan_hash": plan_hash,
        "director_plan_hash": director_plan_hash,
        "harness_version": harness_version,
        "evaluator_version": evaluator_version,
        "blender_version": blender_version,
        "render_settings": render_settings,
        "rollout_seed": rollout_seed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resume_matches(manifest: RealRunManifest, expected: dict[str, Any]) -> bool:
    return manifest.fingerprint == expected.get("fingerprint")
