"""Deterministic COCO-to-YOLO materialization for the P2 accuracy challenger."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from crrc_vision.tiles import build_tiles


def _rows(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _source_path(image: dict[str, Any], source_root: Path | None) -> Path:
    source = Path(str(image["file_name"]))
    if not source.is_absolute() and source_root is not None:
        source = source_root / source
    return source.resolve()


def _split_fingerprints(
    document: dict[str, Any], source_root: Path | None
) -> dict[str, set[object]]:
    images = _rows(document, "images")
    ids = [int(image["id"]) for image in images]
    if len(ids) != len(set(ids)):
        raise ValueError("YOLO_DUPLICATE_IMAGE_ID")
    scenes = [str(image.get("scene_group", "")).strip() for image in images]
    if any(not scene for scene in scenes):
        raise ValueError("YOLO_MISSING_SCENE_GROUP")
    paths: list[Path] = []
    hashes: list[str] = []
    for image in images:
        source = _source_path(image, source_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        paths.append(source)
        hashes.append(hashlib.sha256(source.read_bytes()).hexdigest())
    if len(paths) != len(set(paths)):
        raise ValueError("YOLO_DUPLICATE_SOURCE_PATH")
    return {
        "ids": set(ids),
        "scenes": set(scenes),
        "paths": set(paths),
        "hashes": set(hashes),
    }


def _validate_split_isolation(
    train_document: dict[str, Any],
    val_document: dict[str, Any],
    source_root: Path | None,
) -> None:
    train = _split_fingerprints(train_document, source_root)
    val = _split_fingerprints(val_document, source_root)
    checks = (
        ("scenes", "YOLO_SPLIT_SCENE_LEAKAGE"),
        ("ids", "YOLO_SPLIT_IMAGE_ID_LEAKAGE"),
        ("paths", "YOLO_SPLIT_SOURCE_PATH_LEAKAGE"),
        ("hashes", "YOLO_SPLIT_HASH_LEAKAGE"),
    )
    for field, error in checks:
        if train[field] & val[field]:
            raise ValueError(error)


def _materialize_split(
    coco_path: Path,
    output_root: Path,
    split: str,
    *,
    merge_target_categories: bool = False,
    source_root: Path | None = None,
    include_tiles: bool = False,
) -> tuple[int, int]:
    document = json.loads(coco_path.read_text(encoding="utf-8"))
    images = _rows(document, "images")
    annotations = _rows(document, "annotations")
    categories = sorted(
        (int(row["id"]), str(row["name"])) for row in _rows(document, "categories")
    )
    if categories not in (
        [(1, "fastener"), (2, "pipe_joint")],
        [(1, "fastener_target")],
    ):
        raise ValueError("INVALID_CATEGORY")
    if categories == [(1, "fastener_target")] and not merge_target_categories:
        raise ValueError("SINGLE_TARGET_REQUIRES_MERGE_MODE")
    image_root = output_root / "images" / split
    label_root = output_root / "labels" / split
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    effective_images = 0
    effective_annotations = 0
    for image in sorted(images, key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        source = _source_path(image, source_root)
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower() or ".jpg"
        width = float(image["width"])
        height = float(image["height"])
        source_annotations = sorted(
            annotations_by_image.get(image_id, []), key=lambda row: int(row["id"])
        )

        def write_view(
            stem: str,
            *,
            origin_x: float,
            origin_y: float,
            view_width: float,
            view_height: float,
            selected_annotations: list[dict[str, Any]],
        ) -> None:
            nonlocal effective_images, effective_annotations
            lines: list[str] = []
            for annotation in selected_annotations:
                x, y, box_width, box_height = (
                    float(value) for value in annotation["bbox"]
                )
                clipped_x1 = max(x, origin_x)
                clipped_y1 = max(y, origin_y)
                clipped_x2 = min(x + box_width, origin_x + view_width)
                clipped_y2 = min(y + box_height, origin_y + view_height)
                clipped_width = clipped_x2 - clipped_x1
                clipped_height = clipped_y2 - clipped_y1
                if clipped_width <= 0 or clipped_height <= 0:
                    continue
                class_index = (
                    0
                    if merge_target_categories
                    else int(annotation["category_id"]) - 1
                )
                center_x = (clipped_x1 - origin_x + clipped_width / 2) / view_width
                center_y = (clipped_y1 - origin_y + clipped_height / 2) / view_height
                normalized_width = clipped_width / view_width
                normalized_height = clipped_height / view_height
                values = (center_x, center_y, normalized_width, normalized_height)
                valid_classes = (0,) if merge_target_categories else (0, 1)
                if class_index not in valid_classes or any(
                    not 0.0 <= value <= 1.0 for value in values
                ):
                    raise ValueError(f"INVALID_YOLO_BOX:{annotation.get('id')}")
                lines.append(
                    f"{class_index} {center_x:.6f} {center_y:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )
            (label_root / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="ascii"
            )
            effective_images += 1
            effective_annotations += len(lines)

        full_stem = f"{image_id:06d}_f" if include_tiles else f"{image_id:06d}"
        shutil.copy2(source, image_root / f"{full_stem}{suffix}")
        write_view(
            full_stem,
            origin_x=0,
            origin_y=0,
            view_width=width,
            view_height=height,
            selected_annotations=source_annotations,
        )
        if include_tiles:
            with Image.open(source) as source_image:
                for tile in build_tiles(int(width), int(height), overlap=0.12):
                    tile_stem = f"{image_id:06d}_t{tile.index}"
                    source_image.crop((tile.x1, tile.y1, tile.x2, tile.y2)).save(
                        image_root / f"{tile_stem}{suffix}"
                    )
                    selected = [
                        annotation
                        for annotation in source_annotations
                        if tile.x1
                        <= float(annotation["bbox"][0])
                        + float(annotation["bbox"][2]) / 2
                        <= tile.x2
                        and tile.y1
                        <= float(annotation["bbox"][1])
                        + float(annotation["bbox"][3]) / 2
                        <= tile.y2
                    ]
                    write_view(
                        tile_stem,
                        origin_x=tile.x1,
                        origin_y=tile.y1,
                        view_width=tile.width,
                        view_height=tile.height,
                        selected_annotations=selected,
                    )
    return effective_images, effective_annotations


def prepare_yolo_dataset(
    *,
    train_coco: Path,
    val_coco: Path,
    output_root: Path,
    runtime_output_root: Path | None = None,
    merge_target_categories: bool = False,
    source_root: Path | None = None,
    train_tiles: bool = False,
) -> dict[str, int]:
    """Materialize split-isolated YOLO images/labels and an ASCII runtime YAML."""

    train_document = json.loads(train_coco.read_text(encoding="utf-8"))
    val_document = json.loads(val_coco.read_text(encoding="utf-8"))
    _validate_split_isolation(train_document, val_document, source_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("YOLO_OUTPUT_NOT_EMPTY")
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_root = runtime_output_root.absolute() if runtime_output_root else output_root.absolute()
    if runtime_root.resolve() != output_root.resolve():
        raise ValueError("YOLO_RUNTIME_ROOT_MISMATCH")
    train_images, train_annotations = _materialize_split(
        train_coco,
        output_root,
        "train",
        merge_target_categories=merge_target_categories,
        source_root=source_root,
        include_tiles=train_tiles,
    )
    val_images, val_annotations = _materialize_split(
        val_coco,
        output_root,
        "val",
        merge_target_categories=merge_target_categories,
        source_root=source_root,
    )
    names = "  0: fastener_target\n" if merge_target_categories else (
        "  0: fastener\n  1: pipe_joint\n"
    )
    yaml = f"""path: {runtime_root.as_posix()}
train: images/train
val: images/val
names:
{names}"""
    (output_root / "dataset.yaml").write_text(yaml, encoding="ascii")
    return {
        "train_images": train_images,
        "val_images": val_images,
        "train_annotations": train_annotations,
        "val_annotations": val_annotations,
    }
