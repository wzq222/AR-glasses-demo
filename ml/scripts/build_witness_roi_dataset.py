from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.assets import asset_root  # noqa: E402
from crrc_vision.witness_roi_dataset import build_witness_roi_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a hash-bound train-only witness ROI geometry manifest"
    )
    parser.add_argument(
        "--source",
        default="synthetic/marked-point-v1/repositioned-approved-v2/approved-locals.json",
    )
    parser.add_argument(
        "--formal-truth", default="annotations/fastener-v2/instances.json"
    )
    parser.add_argument(
        "--output", default="runs/witness-roi-v1/dataset/manifest.json"
    )
    args = parser.parse_args()

    root = asset_root().resolve()
    output = (root / args.output).resolve()
    document = build_witness_roi_manifest(
        source_manifest=(root / args.source).resolve(),
        formal_truth=(root / args.formal_truth).resolve(),
        output_manifest=output,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "counts": document["counts"],
                "governance": document["governance"],
                "input_hashes": document["input_hashes"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
