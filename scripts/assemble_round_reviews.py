"""Assemble assistant review payloads for every reviewed case of one split."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.assistant_author_review import main as author_main  # noqa: E402

base = Path(sys.argv[1])
scores_dir = Path(sys.argv[2])
failures = []
for req_path in sorted(base.glob("vbench2-*/assistant_review_request.json")):
    request = json.loads(req_path.read_text(encoding="utf-8"))
    case_id = request["case_id"]
    scores_file = scores_dir / f"{case_id}-scores.json"
    if not scores_file.is_file():
        failures.append(case_id)
        continue
    sys.argv = [
        "assistant_author_review.py",
        "--run-dir", str(req_path.parent),
        "--scores-file", str(scores_file),
        "--out-dir", str(base / "assistant-reviews"),
    ]
    author_main()
print(json.dumps({"assembled_for": str(base), "missing_scores": failures}))
