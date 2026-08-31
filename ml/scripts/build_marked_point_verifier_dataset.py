"""Build scene-isolated E1 proposal crops for a binary mobile verifier pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_verifier import select_verifier_examples


FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _below(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"ASSET_PATH_ESCAPE:{value}")
    return path


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_ROOT_NOT_OBJECT:{path}")
    return value


def _crop_box(
    bbox: list[float], *, width: int, height: int, context: float
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = bbox
    center_x = x + box_width / 2.0
    center_y = y + box_height / 2.0
    side = min(max(max(box_width, box_height) * context, 64.0), float(width), float(height))
    x1 = min(max(int(round(center_x - side / 2.0)), 0), width - int(round(side)))
    y1 = min(max(int(round(center_y - side / 2.0)), 0), height - int(round(side)))
    side_int = int(round(side))
    return x1, y1, x1 + side_int, y1 + side_int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="annotations/marked-point-v1.4")
    parser.add_argument(
        "--train-predictions",
        default="runs/marked-point-p2-e1-pilot/predictions-train-fused.json",
    )
    parser.add_argument(
        "--val-predictions",
        default="runs/marked-point-p2-e1-pilot/predictions-fused.json",
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/marked-point-verifier-e3/dataset-v2")
    parser.add_argument("--proposal-threshold", type=float, default=0.0025412512477487326)
    parser.add_argument("--max-train-positive-per-truth", type=int, default=2)
    parser.add_argument("--max-train-negative-per-scene", type=int, default=10)
    parser.add_argument("--max-val-positive-per-truth", type=int, default=10000)
    parser.add_argument("--max-val-negative-per-scene", type=int, default=10000)
    parser.add_argument("--context", type=float, default=1.6)
    parser.add_argument("--saved-size", type=int, default=0)
    args = parser.parse_args()

    root = asset_root().resolve()
    dataset_root = _below(root, args.dataset)
    source_root = _below(root, args.source)
    formal_truth_path = _below(root, args.formal_truth)
    output_root = _below(root, args.output)
    prediction_paths = {
        "train": _below(root, args.train_predictions),
        "val": _below(root, args.val_predictions),
    }
    if _sha256(formal_truth_path) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    all_examples: list[dict[str, object]] = []
    input_hashes = {"formal_truth_sha256": _sha256(formal_truth_path)}
    scenes: dict[str, set[str]] = {}
    for split in ("train", "val"):
        truth_path = dataset_root / f"instances.{split}.json"
        predictions_path = prediction_paths[split]
        truth = _object(truth_path)
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        if not isinstance(predictions, list) or any(
            not isinstance(row, dict) for row in predictions
        ):
            raise ValueError(f"PREDICTIONS_INVALID:{split}")
        examples = select_verifier_examples(
            predictions,
            truth,
            score_threshold=args.proposal_threshold,
            max_positive_per_truth=(
                args.max_train_positive_per_truth
                if split == "train"
                else args.max_val_positive_per_truth
            ),
            max_negative_per_scene=(
                args.max_train_negative_per_scene
                if split == "train"
                else args.max_val_negative_per_scene
            ),
        )
        images = truth.get("images")
        if not isinstance(images, list):
            raise ValueError("VERIFIER_IMAGES_INVALID")
        image_by_id = {row["id"]: row for row in images if isinstance(row, dict)}
        scenes[split] = {str(row["scene_group"]) for row in images if isinstance(row, dict)}
        input_hashes[f"{split}_truth_sha256"] = _sha256(truth_path)
        input_hashes[f"{split}_predictions_sha256"] = _sha256(predictions_path)
        for index, example in enumerate(examples):
            image_row = image_by_id[example["image_id"]]
            source = Path(str(image_row["file_name"]))
            if not source.is_absolute():
                source = source_root / source
            if not source.is_file():
                raise FileNotFoundError(source)
            crop = _crop_box(
                example["candidate_bbox"],
                width=int(image_row["width"]),
                height=int(image_row["height"]),
                context=args.context,
            )
            label = str(example["label"])
            label_root = output_root / split / label
            label_root.mkdir(parents=True, exist_ok=True)
            crop_path = label_root / f"{split}-{index:05d}.jpg"
            with Image.open(source) as image:
                cropped = image.convert("RGB").crop(crop)
                if args.saved_size > 0:
                    cropped.thumbnail(
                        (args.saved_size, args.saved_size), Image.Resampling.LANCZOS
                    )
                cropped.save(crop_path, quality=95, subsampling=0)
            all_examples.append(
                {
                    **example,
                    "split": split,
                    "crop_xyxy": list(crop),
                    "crop_path": str(crop_path.resolve()),
                    "crop_sha256": _sha256(crop_path),
                }
            )
    if scenes["train"] & scenes["val"]:
        raise RuntimeError("VERIFIER_SCENE_LEAKAGE")
    counts = {
        f"{split}_{label}": sum(
            row["split"] == split and row["label"] == label for row in all_examples
        )
        for split in ("train", "val")
        for label in ("marked_point", "not_marked_point")
    }
    manifest = {
        "schema_version": "marked-point-verifier-dataset-v1",
        "task": "binary_candidate_relevance_pilot",
        "labels": ["marked_point", "not_marked_point"],
        "proposal_threshold": args.proposal_threshold,
        "context": args.context,
        "saved_size": args.saved_size,
        "sampling": {
            "train_max_positive_per_truth": args.max_train_positive_per_truth,
            "train_max_negative_per_scene": args.max_train_negative_per_scene,
            "val_max_positive_per_truth": args.max_val_positive_per_truth,
            "val_max_negative_per_scene": args.max_val_negative_per_scene,
        },
        "counts": counts,
        "scene_counts": {split: len(values) for split, values in scenes.items()},
        "input_hashes": input_hashes,
        "sealed_test_opened": False,
        "examples": all_examples,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"counts": counts, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
