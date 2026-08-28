from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import (  # noqa: E402
    FROZEN_FORMAL_TRUTH_SHA256,
    assert_external_output,
)
from crrc_vision.synthetic_pipeline import build_full_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build seeded train-only synthetic full images")
    parser.add_argument("--input", type=Path, required=True, help="approved-locals.json")
    parser.add_argument("--background-coco", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--formal-truth",
        type=Path,
        default=Path("E:/crrc_vision_data/annotations/fastener-v2/instances.json"),
    )
    args = parser.parse_args()
    output = assert_external_output(args.output, REPOSITORY_ROOT)
    background_coco = args.background_coco or Path(
        "E:/crrc_vision_data/annotations/marked-point-v1.4/instances.train.json"
    )
    source_dir = args.source_dir or Path("E:/crrc_vision_data/source/20240529-luosi")
    manifest = build_full_images(
        args.input,
        background_coco,
        source_dir,
        output,
        args.seed,
        args.formal_truth,
        FROZEN_FORMAL_TRUTH_SHA256,
    )
    print(json.dumps({"output": str(output), "records": len(manifest["records"]), "content_sha256": manifest["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
