"""Assemble first- and second-pass decisions into an isolated reviewed COCO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.auto_labeling import verify_truth_unchanged
from crrc_vision.reviewed_coco import assemble_reviewed_coco, write_reviewed_coco


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="runs/safe-auto-candidates-v1/candidates.json")
    parser.add_argument("--first-reviews", required=True)
    parser.add_argument("--second-reviews")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = asset_root()
    candidates_path = _below(root, args.candidates)
    first_path = _below(root, args.first_reviews)
    second_path = _below(root, args.second_reviews) if args.second_reviews else None
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    required = [candidates_path, first_path, truth_path]
    if second_path is not None:
        required.append(second_path)
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("assembly input is missing")

    truth_before = _sha256(truth_path)
    result = assemble_reviewed_coco(
        _load(candidates_path),
        _load(first_path),
        _load(second_path) if second_path is not None else None,
    )
    write_reviewed_coco(result, output_root)
    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    print(
        json.dumps(
            {
                "complete_images": len(result.document["images"]),
                "annotations": len(result.document["annotations"]),
                "uncertain_images": len(result.uncertain_image_ids),
                "truth_sha256": truth_after,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
