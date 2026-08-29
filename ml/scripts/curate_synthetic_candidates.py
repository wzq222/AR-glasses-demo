from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.synthetic_curation import curate_candidates  # noqa: E402
from crrc_vision.synthetic_contract import assert_external_output  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze reviewed ImageGen marked-point candidates")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert_external_output(args.output, REPOSITORY_ROOT)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = curate_candidates(
        selection,
        args.references,
        args.candidates,
        args.baselines,
        args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
