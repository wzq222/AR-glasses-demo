"""Materialize reviewed marked/lookalike/unmarked crops for verifier training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_verifier import select_semantic_review_examples


FORMAL_TRUTH_SHA256 = "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
REVIEW_SHA256 = "E1AC3794EC80D55C54537BC8CCA9A65A73C35122744DC717F4636C4EB3AE7FBE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _crop_box(bbox, *, width: int, height: int, context: float):
    x, y, box_width, box_height = (float(value) for value in bbox)
    center_x, center_y = x + box_width / 2, y + box_height / 2
    side = min(max(max(box_width, box_height) * context, 64.0), width, height)
    side_int = int(round(side))
    x1 = min(max(int(round(center_x - side / 2)), 0), width - side_int)
    y1 = min(max(int(round(center_y - side / 2)), 0), height - side_int)
    return x1, y1, x1 + side_int, y1 + side_int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="annotations/marked-point-v1.4")
    parser.add_argument(
        "--review", default="review-packs/marked-point-v1/review-complete-v1.4.json"
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/marked-point-verifier-e4/semantic-dataset")
    parser.add_argument("--train-negative-cap", type=int, default=10)
    parser.add_argument("--val-negative-cap", type=int, default=10000)
    parser.add_argument("--context", type=float, default=1.6)
    parser.add_argument("--saved-size", type=int, default=512)
    args = parser.parse_args()

    root = asset_root().resolve()
    dataset_root = (root / args.dataset).resolve()
    review_path = (root / args.review).resolve()
    source_root = (root / args.source).resolve()
    formal_truth = (root / args.formal_truth).resolve()
    output = (root / args.output).resolve()
    if _sha256(formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    if _sha256(review_path) != REVIEW_SHA256:
        raise RuntimeError("SEMANTIC_REVIEW_HASH_MISMATCH")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output}")
    output.mkdir(parents=True, exist_ok=True)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    all_rows = []
    scene_sets = {}
    input_hashes = {
        "formal_truth_sha256": _sha256(formal_truth),
        "review_sha256": _sha256(review_path),
    }
    for split in ("train", "val"):
        truth_path = dataset_root / f"instances.{split}.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        examples = select_semantic_review_examples(
            truth,
            review["reviews"][split],
            max_negative_per_scene_per_class=(
                args.train_negative_cap if split == "train" else args.val_negative_cap
            ),
        )
        images = {row["id"]: row for row in truth["images"]}
        scene_sets[split] = {row["scene_group"] for row in truth["images"]}
        input_hashes[f"{split}_truth_sha256"] = _sha256(truth_path)
        for index, row in enumerate(examples):
            image_row = images[row["image_id"]]
            source = source_root / str(image_row["file_name"])
            crop_box = _crop_box(
                row["candidate_bbox"],
                width=int(image_row["width"]),
                height=int(image_row["height"]),
                context=args.context,
            )
            crop_root = output / split / str(row["label"])
            crop_root.mkdir(parents=True, exist_ok=True)
            crop_path = crop_root / f"{split}-{index:05d}.jpg"
            with Image.open(source) as image:
                cropped = image.convert("RGB").crop(crop_box)
                cropped.thumbnail(
                    (args.saved_size, args.saved_size), Image.Resampling.LANCZOS
                )
                cropped.save(crop_path, quality=95, subsampling=0)
            all_rows.append(
                {
                    **row,
                    "split": split,
                    "crop_xyxy": list(crop_box),
                    "crop_path": str(crop_path.resolve()),
                    "crop_sha256": _sha256(crop_path),
                }
            )
    if scene_sets["train"] & scene_sets["val"]:
        raise RuntimeError("VERIFIER_SCENE_LEAKAGE")
    labels = ["marked_point", "unmarked_fastener", "lookalike"]
    counts = {
        f"{split}_{label}": sum(
            row["split"] == split and row["label"] == label for row in all_rows
        )
        for split in ("train", "val")
        for label in labels
    }
    manifest = {
        "schema_version": "marked-point-semantic-verifier-dataset-v1",
        "task": "reviewed_three_class_candidate_semantics",
        "labels": labels,
        "positive_label": "marked_point",
        "counts": counts,
        "scene_counts": {key: len(value) for key, value in scene_sets.items()},
        "context": args.context,
        "saved_size": args.saved_size,
        "input_hashes": input_hashes,
        "sealed_test_opened": False,
        "examples": all_rows,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": counts, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
