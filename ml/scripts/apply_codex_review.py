"""Validate and freeze Codex decisions without touching formal truth."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.auto_labeling import verify_truth_unchanged
from crrc_vision.codex_review import merge_reviews


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidates",
        default="runs/safe-auto-candidates-v1/candidates.json",
    )
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/safe-auto-decisions-v1.json")
    args = parser.parse_args()

    root = asset_root()
    candidates_path = _below(root, args.candidates)
    reviews_path = _below(root, args.reviews)
    truth_path = _below(root, args.truth)
    output_path = _below(root, args.output)
    for required in (candidates_path, reviews_path, truth_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_path.exists():
        raise FileExistsError(f"decision manifest already exists: {output_path}")

    truth_before = _sha256(truth_path)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, dict) or not isinstance(reviews, dict):
        raise ValueError("candidate and review documents must be objects")
    merged = merge_reviews(candidates, reviews)
    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    merged["integrity"] = {
        "candidates_sha256": _sha256(candidates_path),
        "reviews_sha256": _sha256(reviews_path),
        "truth_sha256_before": truth_before,
        "truth_sha256_after": truth_after,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "candidate_decisions": len(merged["candidate_decisions"]),
                "added_boxes": len(merged["added_boxes"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
