"""Validate the train-only owner challenge registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from videoact.owner_challenges import (  # noqa: E402
    build_default_challenge_set,
    load_owner_challenges,
    validate_owner_challenges,
    write_owner_challenges,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="dataset/owner-challenge-v1/manifest.jsonl")
    parser.add_argument("--write-default", action="store_true")
    args = parser.parse_args()
    path = Path(args.manifest)
    if args.write_default:
        write_owner_challenges(path, build_default_challenge_set())
    fixtures = load_owner_challenges(path)
    print(json.dumps(validate_owner_challenges(fixtures).model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
