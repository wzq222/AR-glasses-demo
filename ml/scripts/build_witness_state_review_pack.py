from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import (  # noqa: E402
    assert_external_output,
    assert_formal_truth_unchanged,
)
from crrc_vision.witness_state_review_pack import build_state_review_pack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a real witness-mark topology and state review pack"
    )
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-truth", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    output = assert_external_output(args.output, REPOSITORY_ROOT)
    formal_hash = assert_formal_truth_unchanged(args.formal_truth.resolve())
    summary = build_state_review_pack(
        args.references.resolve(),
        args.reference_root.resolve(),
        output,
        batch_size=args.batch_size,
        expected_formal_truth_sha256=formal_hash,
        formal_truth_path=args.formal_truth.resolve(),
    )
    print(json.dumps({"output": str(output), **summary.__dict__}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
