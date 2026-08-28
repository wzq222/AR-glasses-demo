"""Repair the high-accuracy sealed split from completed blind-reviewed scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from crrc_vision.assets import asset_root
from crrc_vision.high_accuracy_dataset import repartition_complete_reviews
from crrc_vision.high_accuracy_split import (
    PARTITIONS,
    repair_partition_with_reviewed_sealed_test,
)


EXPECTED_FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _below(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"asset path escapes CRRC_VISION_DATA_ROOT: {relative}")
    return path


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
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


def _image_box_counts(document: dict[str, Any]) -> dict[str, int]:
    image_by_id = {
        row.get("id", row.get("image_id")): str(row.get("scene_group") or "")
        for row in document.get("images", [])
        if isinstance(row, dict)
    }
    counts = Counter(
        image_by_id.get(row.get("image_id"), "")
        for row in document.get("annotations", [])
        if isinstance(row, dict)
    )
    return {scene: count for scene, count in counts.items() if scene}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition", default="selections/high-accuracy-v1/partition.json"
    )
    parser.add_argument(
        "--exclusions", default="selections/high-accuracy-v1/exclusions.json"
    )
    parser.add_argument(
        "--existing-reviewed",
        default="annotations/silver-gate-cumulative-013/instances.silver.json",
    )
    parser.add_argument(
        "--train-reviewed",
        default=(
            "review-packs/high-accuracy-v1/train/decisions/"
            "assembly-original-complete/instances.reviewed.json"
        ),
    )
    parser.add_argument(
        "--val-reviewed",
        default=(
            "review-packs/high-accuracy-v1/val/decisions/"
            "assembly-original-complete/instances.reviewed.json"
        ),
    )
    parser.add_argument("--truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output-selection", default="selections/high-accuracy-v2")
    parser.add_argument(
        "--output-reviews", default="review-packs/high-accuracy-v2/decisions"
    )
    parser.add_argument("--sealed-test-count", type=int, default=30)
    parser.add_argument("--minimum-sealed-test-boxes", type=int, default=200)
    args = parser.parse_args()

    root = asset_root()
    partition_path = _below(root, args.partition)
    exclusions_path = _below(root, args.exclusions)
    existing_path = _below(root, args.existing_reviewed)
    train_path = _below(root, args.train_reviewed)
    val_path = _below(root, args.val_reviewed)
    truth_path = _below(root, args.truth)
    selection_root = _below(root, args.output_selection)
    review_root = _below(root, args.output_reviews)
    inputs = (
        partition_path,
        exclusions_path,
        existing_path,
        train_path,
        val_path,
        truth_path,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for output in (selection_root, review_root):
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"OUTPUT_DIRECTORY_NOT_EMPTY:{output}")

    truth_before = _sha256(truth_path)
    if truth_before != EXPECTED_FORMAL_TRUTH_SHA256:
        raise RuntimeError(f"FORMAL_TRUTH_HASH_MISMATCH:{truth_before}")
    partition = _load(partition_path)
    exclusions = _load(exclusions_path)
    existing = _load(existing_path)
    train_review = _load(train_path)
    val_review = _load(val_path)
    completed_counts = _image_box_counts(train_review)
    completed_counts.update(_image_box_counts(val_review))
    existing_scenes = {
        str(row.get("scene_group") or "")
        for row in existing.get("images", [])
        if isinstance(row, dict)
    }
    uncertain_scenes = {
        str(row.get("scene_group") or "")
        for split in ("train", "val")
        for row in exclusions.get(split, [])
        if isinstance(row, dict) and row.get("reason") == "review_uncertain"
    }

    repaired, quarantined = repair_partition_with_reviewed_sealed_test(
        partition,
        completed_box_counts=completed_counts,
        existing_reviewed_scenes=existing_scenes,
        excluded_scenes=uncertain_scenes,
        sealed_test_count=args.sealed_test_count,
        minimum_sealed_test_boxes=args.minimum_sealed_test_boxes,
    )
    owner_by_scene = {
        str(row["scene_group"]): split
        for split in PARTITIONS
        for row in repaired[split]
    }
    repaired_exclusions: dict[str, object] = {
        "schema_version": "high-accuracy-exclusions-v1",
        "train": [dict(row) for row in exclusions.get("train", [])],
        "val": [dict(row) for row in exclusions.get("val", [])],
        "sealed_test": [],
    }
    for row in quarantined:
        split = owner_by_scene[str(row["scene_group"])]
        if split == "sealed_test":
            raise AssertionError("quarantined scene entered sealed test")
        repaired_exclusions[split].append(row)  # type: ignore[index, union-attr]
    for split in ("train", "val"):
        repaired_exclusions[split] = sorted(  # type: ignore[index]
            repaired_exclusions[split], key=lambda row: str(row["scene_group"])
        )

    repartitioned = repartition_complete_reviews(
        repaired, [train_review, val_review]
    )
    partition_out = selection_root / "partition.json"
    exclusions_out = selection_root / "exclusions.json"
    _atomic_json(partition_out, repaired)
    _atomic_json(exclusions_out, repaired_exclusions)
    review_paths: dict[str, Path] = {}
    for split in PARTITIONS:
        directory = review_root / split.replace("_", "-")
        reviewed_out = directory / "instances.reviewed.json"
        review_paths[split] = reviewed_out
        _atomic_json(reviewed_out, repartitioned[split])
        uncertain_ids = sorted(
            row["image_id"]
            for row in repaired_exclusions.get(split, [])
            if isinstance(row, dict) and row.get("reason") == "review_uncertain"
        )
        _atomic_json(directory / "uncertain-images.json", uncertain_ids)

    truth_after = _sha256(truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED_DURING_REPAIR")
    manifest = {
        "schema_version": "high-accuracy-partition-repair-manifest-v1",
        "partition_counts": {
            split: len(repaired[split]) for split in PARTITIONS
        },
        "reviewed_counts": {
            split: {
                "scenes": len(repartitioned[split]["images"]),
                "boxes": len(repartitioned[split]["annotations"]),
            }
            for split in PARTITIONS
        },
        "exclusion_counts": {
            split: len(repaired_exclusions[split]) for split in PARTITIONS
        },
        "formal_truth_sha256": truth_after,
        "input_hashes": {path.name: _sha256(path) for path in inputs},
        "output_hashes": {
            "partition.json": _sha256(partition_out),
            "exclusions.json": _sha256(exclusions_out),
            **{
                f"{split}.reviewed.json": _sha256(path)
                for split, path in review_paths.items()
            },
        },
    }
    _atomic_json(selection_root / "repair-manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
