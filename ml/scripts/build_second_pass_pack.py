"""Render blind second-pass geometry review tasks from first-pass proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.auto_labeling import verify_truth_unchanged
from crrc_vision.codex_review_pack import build_second_pass_tasks


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = asset_root()
    reviews_path = _below(root, args.reviews)
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    if not reviews_path.is_file() or not truth_path.is_file():
        raise FileNotFoundError("review or truth input is missing")
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    truth_before = _sha256(truth_path)
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    count = build_second_pass_tasks(
        reviews,
        output_root,
        source_root=source_root,
    )
    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    print(json.dumps({"images": count, "truth_sha256": truth_after}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
