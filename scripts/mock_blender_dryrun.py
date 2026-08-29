"""Dry-run a generated Blender job against a stubbed bpy.

Executes the generated source with ``bpy``/``mathutils`` replaced by
``MagicMock`` and the job directory pre-seeded with a run manifest.  Any
undefined name, frozen-dataclass mutation, or shape error raised before real
rendering surfaces here -- without spending a Blender render on it.
"""

from __future__ import annotations

import json
import shutil
import sys
import unittest.mock
from pathlib import Path
from types import ModuleType


def dry_run(source_path: Path, template_dir: Path | None) -> str | None:
    work = Path(source_path).parent / (source_path.stem + "_dryrun")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if template_dir is not None:
        for name in ("run_manifest.json", "director_plan.json"):
            source = template_dir / name
            if source.is_file():
                shutil.copy2(source, work / name)
    if not (work / "run_manifest.json").is_file():
        return "dryrun missing run_manifest.json template"

    bpy = unittest.mock.MagicMock(name="bpy")
    # object.mode_set and friends are absorbed by the mock; make a few
    # attribute reads return sane scalars so arithmetic in the job survives.
    bpy.app.version_string = "4.0-mock"
    bpy.context.scene.render.fps = 12
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 120
    mathutils = ModuleType("mathutils")
    mathutils.Vector = lambda value: type("V", (), {"to_track_quat": lambda self, a, b: type("Q", (), {"to_euler": lambda self2: (0.0, 0.0, 0.0)})()})()
    # Tolerant json: artifact writes under the mock may contain mock values;
    # only real runtime errors should fail the dry-run.
    real_json = json
    tolerant_json = ModuleType("json")
    tolerant_json.dumps = lambda obj, **kw: real_json.dumps(obj, default=str, **kw)
    tolerant_json.loads = real_json.loads
    source = source_path.read_text(encoding="utf-8")
    namespace = {"__name__": "blender_job_dryrun", "__file__": str(work / "blender_job.py")}
    repo_root = str(Path(__file__).resolve().parents[1])
    stdout = sys.stdout
    stderr = sys.stderr
    try:
        import io

        buffer = io.StringIO()
        sys.stdout = buffer
        sys.stderr = buffer
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        # Only bpy and mathutils are stubbed; the verified blender.lib
        # primitives are Blender-free and import for real.
        with unittest.mock.patch.dict(sys.modules, {"bpy": bpy, "mathutils": mathutils, "json": tolerant_json}):
            exec(compile(source, str(source_path), "exec"), namespace)  # noqa: S102 - deliberate dry-run
        return None
    except Exception as exc:  # noqa: BLE001 - report any dry-run failure
        return f"{type(exc).__name__}: {exc}"
    finally:
        sys.stdout = stdout
        sys.stderr = stderr


def main() -> int:
    import json

    source_path = Path(sys.argv[1])
    template_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    error = dry_run(source_path, template_dir)
    print(json.dumps({"source": str(source_path), "dryrun_error": error}))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
