"""Build a Git-external Codex visual review pack from fused candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.auto_labeling import verify_truth_unchanged
from crrc_vision.codex_review_pack import build_pack


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
        default="runs/safe-auto-candidates-v2.2/candidates.json",
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--selection", default="selections/selection-v2.json")
    parser.add_argument("--partition", choices=("train", "val", "sealed_test"))
    parser.add_argument(
        "--exclude-reviewed",
        help="Optional reviewed COCO whose relative paths must not enter this pack.",
    )
    parser.add_argument("--output", default="review-packs/safe-auto-v2")
    args = parser.parse_args()

    root = asset_root()
    candidates_path = _below(root, args.candidates)
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    selection_path = _below(root, args.selection)
    output_root = _below(root, args.output)
    for required in (candidates_path, truth_path, selection_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    truth_before = _sha256(truth_path)
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, dict):
        raise ValueError("candidate document must be an object")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    items = (
        selection.get(args.partition)
        if isinstance(selection, dict) and args.partition
        else selection.get("items") if isinstance(selection, dict) else None
    )
    if not isinstance(items, list) or any(not isinstance(row, dict) for row in items):
        raise ValueError("selection must contain an items list")
    selected_paths = [str(row.get("relative_path") or "") for row in items]
    if any(not path for path in selected_paths):
        raise ValueError("selection contains an empty relative_path")
    excluded_reviewed_sha256 = None
    if args.exclude_reviewed:
        excluded_path = _below(root, args.exclude_reviewed)
        if not excluded_path.is_file():
            raise FileNotFoundError(excluded_path)
        excluded_document = json.loads(excluded_path.read_text(encoding="utf-8"))
        excluded_images = (
            excluded_document.get("images")
            if isinstance(excluded_document, dict)
            else None
        )
        if not isinstance(excluded_images, list) or any(
            not isinstance(row, dict) for row in excluded_images
        ):
            raise ValueError("excluded reviewed COCO must contain an images list")
        excluded_paths = {
            str(row.get("relative_path") or row.get("file_name") or "")
            for row in excluded_images
        }
        selected_paths = [
            relative_path
            for relative_path in selected_paths
            if relative_path not in excluded_paths
        ]
        excluded_reviewed_sha256 = _sha256(excluded_path)
    summary = build_pack(
        candidates,
        source_root,
        output_root,
        selected_relative_paths=selected_paths,
        partition=args.partition,
        partition_manifest_sha256=_sha256(selection_path),
        include_existing_decisions=False,
    )
    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    integrity = {
        "schema_version": "safe-auto-review-integrity-v1",
        "partition": args.partition,
        "partition_manifest_sha256": _sha256(selection_path),
        "candidates_sha256": _sha256(candidates_path),
        "selection_sha256": _sha256(selection_path),
        "excluded_reviewed_sha256": excluded_reviewed_sha256,
        "source_image_hashes": {
            relative_path: _sha256(source_root / relative_path)
            for relative_path in selected_paths
        },
        "truth_sha256_before": truth_before,
        "truth_sha256_after": truth_after,
    }
    (output_root / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary.__dict__, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
