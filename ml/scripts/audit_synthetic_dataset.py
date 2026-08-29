from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_audit import audit_full_dataset, audit_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit train-only synthetic marked-point records")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--review-pack", type=Path,
                        help="Hash-bound review-pack manifest; required for full-image audit")
    parser.add_argument(
        "--formal-truth",
        type=Path,
        default=Path("E:/crrc_vision_data/annotations/fastener-v2/instances.json"),
    )
    args = parser.parse_args()
    manifest_path = args.manifest or args.root / "approved-locals.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = document["records"] if isinstance(document, dict) else document
    if isinstance(document, dict) and document.get("schema_version") == "synthetic-marked-point-full-v1":
        result = audit_full_dataset(
            document,
            args.root,
            args.root / "instances.synthetic-train.json",
            args.formal_truth,
            review_pack_manifest_path=args.review_pack,
        )
    else:
        result = audit_manifest(records, args.formal_truth)
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    return 0 if result.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
