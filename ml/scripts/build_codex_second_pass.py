"""Create blind second-pass tasks for adjusted or newly added boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.codex_review_pack import build_second_pass_tasks


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-reviews", required=True)
    parser.add_argument("--output", default="review-packs/safe-auto-v1/second-pass")
    args = parser.parse_args()

    root = asset_root()
    reviews_path = _below(root, args.first_reviews)
    output_root = _below(root, args.output)
    if not reviews_path.is_file():
        raise FileNotFoundError(reviews_path)
    document = json.loads(reviews_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("first review document must be an object")
    images = build_second_pass_tasks(document, output_root)
    print(json.dumps({"second_pass_images": images}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
