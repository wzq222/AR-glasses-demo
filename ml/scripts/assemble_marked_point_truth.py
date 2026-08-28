"""Assemble guarded train/val marked-point truth outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from crrc_vision.assets import asset_root
from crrc_vision.marked_point import assemble_partition


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


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection", default="selections/marked-point-v1/selection.json"
    )
    parser.add_argument(
        "--review", default="review-packs/marked-point-v1/review-complete.json"
    )
    parser.add_argument(
        "--candidates", default="runs/marked-point-proposals-v1/union/candidates.json"
    )
    parser.add_argument(
        "--pack-manifest", default="review-packs/marked-point-v1/pack-manifest.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="annotations/marked-point-v1")
    args = parser.parse_args()

    root = asset_root()
    paths = {
        name: _below(root, relative)
        for name, relative in {
            "selection": args.selection,
            "review": args.review,
            "candidates": args.candidates,
            "pack_manifest": args.pack_manifest,
            "source": args.source,
            "truth": args.truth,
            "output": args.output,
        }.items()
    }
    for name in ("selection", "review", "candidates", "pack_manifest", "truth"):
        if not paths[name].is_file():
            raise FileNotFoundError(paths[name])
    if not paths["source"].is_dir():
        raise FileNotFoundError(paths["source"])
    output_root = paths["output"]
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"OUTPUT_DIRECTORY_NOT_EMPTY:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    truth_before = _sha256(paths["truth"])
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")
    selection = _load(paths["selection"])
    review_set = _load(paths["review"])
    if review_set.get("schema_version") != "marked-point-review-set-v1":
        raise ValueError("INVALID_REVIEW_SET_SCHEMA")
    reviews = review_set.get("reviews")
    if not isinstance(reviews, dict):
        raise ValueError("REVIEW_SET_PARTITIONS_MISSING")
    expected_hashes = {
        "selection_sha256": _sha256(paths["selection"]),
        "candidates_sha256": _sha256(paths["candidates"]),
        "pack_manifest_sha256": _sha256(paths["pack_manifest"]),
    }
    review_hashes = review_set.get("input_hashes")
    if not isinstance(review_hashes, dict):
        raise ValueError("REVIEW_INPUT_HASHES_MISSING")
    for key, expected in expected_hashes.items():
        if str(review_hashes.get(key) or "").upper() != expected:
            raise ValueError(f"REVIEW_INPUT_HASH_MISMATCH:{key}")

    selected_by_path: dict[str, dict[str, object]] = {}
    selected_partition: dict[str, str] = {}
    image_sizes: dict[str, tuple[int, int]] = {}
    for partition in ("train", "val"):
        rows = selection.get(partition)
        if not isinstance(rows, list):
            raise ValueError(f"SELECTION_INVALID:{partition}")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("SELECTION_ROW_INVALID")
            relative = str(row.get("relative_path") or "")
            if not relative or relative in selected_by_path:
                raise ValueError(f"SELECTION_DUPLICATE_PATH:{relative}")
            source_path = paths["source"] / relative
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            if _sha256(source_path).lower() != str(row.get("sha256") or "").lower():
                raise RuntimeError(f"SOURCE_HASH_MISMATCH:{relative}")
            with Image.open(source_path) as opened:
                oriented = ImageOps.exif_transpose(opened)
                image_sizes[relative] = oriented.size
            selected_by_path[relative] = row
            selected_partition[relative] = partition

    exclusions = review_set.get("uncertain_exclusions", [])
    if not isinstance(exclusions, list) or any(not isinstance(row, dict) for row in exclusions):
        raise ValueError("INVALID_UNCERTAIN_EXCLUSIONS")
    exclusion_paths = {str(row.get("relative_path") or "") for row in exclusions}
    if "" in exclusion_paths or len(exclusion_paths) != len(exclusions):
        raise ValueError("INVALID_UNCERTAIN_EXCLUSION_IDENTITY")
    for row in exclusions:
        relative = str(row["relative_path"])
        selected = selected_by_path.get(relative)
        if selected is None:
            raise ValueError(f"EXCLUSION_OUTSIDE_SELECTION:{relative}")
        if str(row.get("source_sha256") or "").upper() != str(selected["sha256"]).upper():
            raise ValueError(f"EXCLUSION_IDENTITY_MISMATCH:{relative}")

    outputs: dict[str, dict[str, object]] = {}
    reviewed_paths: set[str] = set()
    for partition in ("train", "val"):
        review = reviews.get(partition)
        if not isinstance(review, dict):
            raise ValueError(f"REVIEW_PARTITION_MISSING:{partition}")
        images = review.get("images")
        if not isinstance(images, list):
            raise ValueError(f"REVIEW_IMAGES_INVALID:{partition}")
        for row in images:
            if not isinstance(row, dict):
                raise ValueError("REVIEW_IMAGE_INVALID")
            relative = str(row.get("relative_path") or "")
            if relative in reviewed_paths or relative in exclusion_paths:
                raise ValueError(f"REVIEW_COVERAGE_DUPLICATE:{relative}")
            reviewed_paths.add(relative)
        outputs[partition] = assemble_partition(
            review,
            selection=selection,
            partition=partition,
            image_sizes=image_sizes,
        )
    if reviewed_paths | exclusion_paths != set(selected_by_path):
        missing = sorted(set(selected_by_path) - reviewed_paths - exclusion_paths)
        raise ValueError(f"REVIEW_SELECTION_COVERAGE_MISMATCH:{missing[0]}")

    negatives = {
        "schema_version": "marked-point-negatives-v1",
        "partitions": {
            partition: outputs[partition]["info"]["negative_counts"]
            for partition in ("train", "val")
        },
        "uncertain_exclusions": exclusions,
    }
    _atomic_json(output_root / "instances.train.json", outputs["train"])
    _atomic_json(output_root / "instances.val.json", outputs["val"])
    _atomic_json(output_root / "negatives.json", negatives)

    truth_after = _sha256(paths["truth"])
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_ASSEMBLY")
    negative_totals = Counter()
    for partition in ("train", "val"):
        negative_totals.update(outputs[partition]["info"]["negative_counts"])
    manifest = {
        "schema_version": "marked-point-dataset-manifest-v1",
        "complete_scenes": {
            partition: len(outputs[partition]["images"])
            for partition in ("train", "val")
        },
        "positive_boxes": {
            partition: len(outputs[partition]["annotations"])
            for partition in ("train", "val")
        },
        "negative_counts": dict(sorted(negative_totals.items())),
        "uncertain_exclusions": len(exclusions),
        "input_hashes": {**expected_hashes, "review_sha256": _sha256(paths["review"])},
        "output_hashes": {
            name: _sha256(output_root / name)
            for name in ("instances.train.json", "instances.val.json", "negatives.json")
        },
        "formal_truth_sha256_before": truth_before,
        "formal_truth_sha256_after": truth_after,
        "old_sealed_test_opened": False,
    }
    _atomic_json(output_root / "manifest.json", manifest)
    print(json.dumps({**manifest, "output": str(output_root)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
