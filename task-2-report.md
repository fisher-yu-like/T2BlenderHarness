# T02 report: repository portability and reproducibility

## Scope completed

- Updated `scripts/check_repo_portability.py` to detect both literal and JSON-escaped Windows drive paths (for example, `C:\\Users\\...`).
- Restricted the portability gate to active executable contract inputs: the checked-in `dataset/vbench2-agent-training-index-v1` and project execution configuration. Historical experiment manifests, golden-review evidence, and training baselines were intentionally not rewritten.
- Confirmed the active benchmark metadata already uses repository-relative POSIX paths; no dataset rewrite was needed, preserving its validated fingerprint.
- Added cross-platform unit and separate Blender integration test entry points. Both launch pytest through `sys.executable`; Blender integration requires an explicit `--blender-bin` and sets `VIDEOACT_BLENDER_BIN` only for that subprocess.
- Added the `blender_integration` pytest marker, split the GitHub Actions workflow into a Windows/Linux unit matrix and an Ubuntu Blender integration job, and documented both local commands in README.

## Changed files

- `.github/workflows/ci.yml`
- `pyproject.toml`
- `README.md`
- `scripts/check_repo_portability.py`
- `scripts/run_unit_tests.py` (new)
- `scripts/run_blender_integration.py` (new)
- `tests/test_repo_portability.py` (new)
- `tests/test_blender_integration.py` (new)

## Verification

| Command | Result |
|---|---|
| `uv run pytest -q tests/test_repo_portability.py --basetemp .pytest-t02-portability` | 6 passed |
| `uv run python scripts/run_unit_tests.py -q tests/test_repo_portability.py tests/test_benchmark_reproducibility.py --basetemp .pytest-t02-focused` | 9 passed |
| `uv run python scripts/run_blender_integration.py --blender-bin blender --collect-only -q tests/test_blender_integration.py --basetemp .pytest-t02-integration-collect` | 1 Blender integration test collected |
| `uv run python scripts/check_repo_portability.py` | pass; no failures |
| `uv run python scripts/validate_benchmark_prompt_index.py --root dataset/vbench2-agent-training-index-v1` | pass; `training_eligible: true`; fingerprint `196f71f7bf9deed54b2c4cd0b0f943e3122fff01114b3a84b2c1b400e0b43332` |
| `uv run python -m compileall -q src evaluator blender scripts training` | pass |
| `uv run --with pyyaml python -c "...yaml.safe_load(.github/workflows/ci.yml)..."` | `ci_yaml=valid` |

## Limitations and non-actions

- GitHub Actions was not run from this worktree; the workflow is checked in and YAML-validated only.
- A real Blender process was not started locally. The isolated integration entry was collection-tested only because no local Blender binary was supplied for execution.
- The initial pytest invocation using the system temporary directory hit `WinError 5` under `C:\\Users\\sy\\AppData\\Local\\Temp\\pytest-of-sy`; all T02 pytest commands use worktree-local `--basetemp` directories instead.
- `task-2-brief.md` was not present at the requested worktree root or found under the supplied repository/plan locations. The T02 section of `D:\\sy\\T2BlenderHarness-closed-loop-weakness-discovery-design-zh.md` was used as the task specification.
- Existing dirty changes were preserved. T00/T01-owned source files were not modified. Historical artifacts that retain author-machine paths remain immutable by design and are outside the active portability scan.
