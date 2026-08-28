from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import FROZEN_FORMAL_TRUTH_SHA256  # noqa: E402
from crrc_vision.synthetic_pipeline import ingest_local_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest reviewed ImageGen local marked-point images")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing PNG/JPEG plus .ext.json sidecars")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formal-truth",
        type=Path,
        default=Path("E:/crrc_vision_data/annotations/fastener-v2/instances.json"),
    )
    args = parser.parse_args()
    paths = sorted(
        path for path in args.input.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    document = ingest_local_candidates(
        paths,
        args.output,
        REPOSITORY_ROOT,
        args.formal_truth,
        FROZEN_FORMAL_TRUTH_SHA256,
    )
    print(json.dumps({"output": str(args.output), "records": len(document["records"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
