from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_contract import assert_external_output  # noqa: E402
from crrc_vision.witness_state_second_review import (  # noqa: E402
    audit_second_pass_reviews,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit blind witness-state second-pass decisions and training eligibility"
    )
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = assert_external_output(args.output, REPOSITORY_ROOT)
    if output.exists():
        raise FileExistsError(output)
    result = audit_second_pass_reviews(
        args.pack_manifest.resolve(),
        args.decisions.resolve(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(output), **result["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
