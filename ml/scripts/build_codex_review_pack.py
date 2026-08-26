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
        default="runs/safe-auto-candidates-v1/candidates.json",
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--selection", default="selections/selection-v2.json")
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
    items = selection.get("items") if isinstance(selection, dict) else None
    if not isinstance(items, list) or any(not isinstance(row, dict) for row in items):
        raise ValueError("selection must contain an items list")
    selected_paths = [str(row.get("relative_path") or "") for row in items]
    if any(not path for path in selected_paths):
        raise ValueError("selection contains an empty relative_path")
    summary = build_pack(
        candidates,
        source_root,
        output_root,
        selected_relative_paths=selected_paths,
    )
    truth_after = _sha256(truth_path)
    verify_truth_unchanged(truth_before, truth_after)
    integrity = {
        "schema_version": "safe-auto-review-integrity-v1",
        "candidates_sha256": _sha256(candidates_path),
        "selection_sha256": _sha256(selection_path),
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
