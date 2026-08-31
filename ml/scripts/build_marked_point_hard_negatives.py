"""Materialize hash-bound empty train crops from marked-point false positives."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_hard_negatives import select_hard_negative_crops


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-truth", default="annotations/marked-point-v1.4/instances.train.json"
    )
    parser.add_argument(
        "--val-truth", default="annotations/marked-point-v1.4/instances.val.json"
    )
    parser.add_argument(
        "--predictions",
        default="runs/marked-point-p2-e1-pilot/predictions-train-fused.json",
    )
    parser.add_argument("--source", default="source/20240529-luosi")
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/marked-point-p2-e2/hard-negatives")
    parser.add_argument("--score-threshold", type=float, default=0.0025)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--max-per-scene", type=int, default=1)
    parser.add_argument("--maximum-crops", type=int)
    args = parser.parse_args()

    root = asset_root().resolve()
    train_path = _below(root, args.train_truth)
    val_path = _below(root, args.val_truth)
    predictions_path = _below(root, args.predictions)
    source_root = _below(root, args.source)
    formal_truth_path = _below(root, args.formal_truth)
    output_root = _below(root, args.output)
    for path in (train_path, val_path, predictions_path, formal_truth_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(formal_truth_path) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output_root}")

    train = _object(train_path)
    val = _object(val_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list) or any(
        not isinstance(row, dict) for row in predictions
    ):
        raise ValueError("PREDICTIONS_INVALID")
    val_images = val.get("images")
    if not isinstance(val_images, list):
        raise ValueError("VAL_IMAGES_INVALID")
    forbidden_hashes = {
        str(row.get("sha256") or "").upper()
        for row in val_images
        if isinstance(row, dict)
    }
    selected = select_hard_negative_crops(
        predictions,
        train,
        score_threshold=args.score_threshold,
        crop_size=args.crop_size,
        max_per_scene=args.max_per_scene,
        maximum_crops=args.maximum_crops,
        forbidden_sha256=forbidden_hashes,
    )
    if not selected:
        raise RuntimeError("NO_HARD_NEGATIVE_CROPS")

    output_root.mkdir(parents=True, exist_ok=True)
    image_root = output_root / "images"
    image_root.mkdir()
    train_images = train.get("images")
    if not isinstance(train_images, list):
        raise ValueError("TRAIN_IMAGES_INVALID")
    next_image_id = max(int(row["id"]) for row in train_images if isinstance(row, dict)) + 1
    derived_images: list[dict[str, object]] = []
    materialized: list[dict[str, object]] = []
    for index, row in enumerate(selected):
        source = Path(str(row["source_file_name"]))
        if not source.is_absolute():
            source = source_root / source
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        x1, y1, x2, y2 = (int(value) for value in row["crop_xyxy"])
        crop_path = image_root / f"hn-{int(row['image_id']):06d}-{index:03d}.jpg"
        with Image.open(source) as image:
            image.convert("RGB").crop((x1, y1, x2, y2)).save(
                crop_path, quality=95, subsampling=0
            )
        crop_sha256 = _sha256(crop_path)
        derived_images.append(
            {
                "id": next_image_id + index,
                "file_name": str(crop_path.resolve()),
                "width": x2 - x1,
                "height": y2 - y1,
                "scene_group": row["scene_group"],
                "sha256": crop_sha256,
                "synthetic": False,
                "derived": True,
                "origin": "e2_hard_negative_crop",
                "source_image_id": row["image_id"],
            }
        )
        materialized.append({**row, "crop_path": str(crop_path), "crop_sha256": crop_sha256})

    extended = {
        **train,
        "info": {
            **(train.get("info") if isinstance(train.get("info"), dict) else {}),
            "partition": "train",
            "hard_negative_images": len(derived_images),
            "hard_negative_policy": "empty_640_crop_no_truth_center",
        },
        "images": [*train_images, *derived_images],
    }
    extended_path = output_root / "instances.train.e2.json"
    extended_path.write_text(
        json.dumps(extended, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "marked-point-hard-negatives-v1",
        "score_threshold": args.score_threshold,
        "crop_size": args.crop_size,
        "max_per_scene": args.max_per_scene,
        "maximum_crops": args.maximum_crops,
        "selected_crops": len(materialized),
        "sealed_test_opened": False,
        "input_hashes": {
            "train_truth_sha256": _sha256(train_path),
            "val_truth_sha256": _sha256(val_path),
            "predictions_sha256": _sha256(predictions_path),
            "formal_truth_sha256": _sha256(formal_truth_path),
        },
        "extended_train_sha256": _sha256(extended_path),
        "crops": materialized,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected_crops": len(materialized),
                "extended_train": str(extended_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
