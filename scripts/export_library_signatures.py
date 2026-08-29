"""Export verified Blender primitive signatures for BlenderCodeAgent context."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blender.lib.__meta__ import collect_library_signatures  # noqa: E402


def export_library_signatures(output: str | Path) -> Path:
    """Write the current library metadata and return its absolute path."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(collect_library_signatures(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="blender/lib/signatures.json")
    args = parser.parse_args()
    print(export_library_signatures(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

