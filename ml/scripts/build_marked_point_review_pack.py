"""Build the Git-external marked-point full-image review pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_review_pack import build_review_pack


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{relative}")
    return path


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", default="selections/marked-point-v1/selection.json"
    )
    parser.add_argument(
        "--candidates", default="runs/marked-point-proposals-v1/union/candidates.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="review-packs/marked-point-v1")
    args = parser.parse_args()

    root = asset_root()
    selection_path = _below(root, args.selection)
    candidates_path = _below(root, args.candidates)
    source_root = _below(root, args.source)
    truth_path = _below(root, args.truth)
    output_root = _below(root, args.output)
    for required in (selection_path, candidates_path, truth_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)

    truth_before = _sha256(truth_path)
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")
    summary = build_review_pack(
        _load(selection_path),
        _load(candidates_path),
        source_root,
        output_root,
    )
    truth_after = _sha256(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_REVIEW_PACK")
    integrity = {
        "schema_version": "marked-point-review-integrity-v1",
        "selection_sha256": _sha256(selection_path),
        "candidates_sha256": _sha256(candidates_path),
        "source_image_hashes": {
            str(row["relative_path"]): str(row["sha256"]).upper()
            for partition in ("train", "val")
            for row in _load(selection_path)[partition]  # type: ignore[index]
        },
        "formal_truth_sha256_before": truth_before,
        "formal_truth_sha256_after": truth_after,
        "old_sealed_test_opened": False,
    }
    (output_root / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {**summary.__dict__, "formal_truth_sha256": truth_after, "output": str(output_root)},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
