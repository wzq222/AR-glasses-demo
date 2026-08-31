from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import assert_external_output  # noqa: E402
from crrc_vision.witness_state_review_pack import (  # noqa: E402
    build_state_second_pass_pack,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a blind endpoint second-pass pack for real witness marks"
    )
    parser.add_argument("--source-pack", type=Path, required=True)
    parser.add_argument("--reference-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    output = assert_external_output(args.output, REPOSITORY_ROOT)
    summary = build_state_second_pass_pack(
        args.source_pack.resolve(),
        args.reference_id,
        output,
        batch_size=args.batch_size,
    )
    print(json.dumps({"output": str(output), **summary.__dict__}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
