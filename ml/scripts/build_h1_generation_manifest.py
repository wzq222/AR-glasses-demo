from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.hard_sample_generation import build_generation_manifest  # noqa: E402
from crrc_vision.synthetic_contract import (  # noqa: E402
    assert_external_output,
    assert_formal_truth_unchanged,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hash-bind H1 ImageGen results")
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--formal-truth", type=Path)
    args = parser.parse_args()

    assert_external_output(args.output, REPOSITORY_ROOT)
    root = os.environ.get("CRRC_VISION_DATA_ROOT", "")
    if args.formal_truth is None and not root:
        raise RuntimeError("set CRRC_VISION_DATA_ROOT or pass --formal-truth")
    formal_truth = (
        args.formal_truth
        if args.formal_truth is not None
        else Path(root) / "annotations/fastener-v2/instances.json"
    ).resolve()
    formal_hash = assert_formal_truth_unchanged(formal_truth)
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
    if jobs.get("formal_truth_sha256") != formal_hash:
        raise RuntimeError("jobs formal truth hash mismatch")
    manifest = build_generation_manifest(jobs, args.generated, args.attempt)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite generation manifest: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    assert_formal_truth_unchanged(formal_truth, formal_hash)
    print(
        json.dumps(
            {"output": str(args.output), "count": manifest["count"], "formal_truth_sha256": formal_hash},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
