from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_review import apply_hash_bound_review  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply hash-bound visual decisions to a synthetic full-image manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-pack", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    document = apply_hash_bound_review(args.manifest, args.review_pack, args.decisions)
    counts = {status: sum(item["review_status"] == status for item in document["records"])
              for status in ("APPROVED", "REJECTED", "UNCERTAIN")}
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
