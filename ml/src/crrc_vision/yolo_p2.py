"""Deterministic COCO-to-YOLO materialization for the P2 accuracy challenger."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
from typing import Any


def _rows(document: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{key} must be a list of objects")
    return value


def _split_fingerprints(document: dict[str, Any]) -> dict[str, set[object]]:
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
        source = Path(str(image["file_name"])).resolve()
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
    train_document: dict[str, Any], val_document: dict[str, Any]
) -> None:
    train = _split_fingerprints(train_document)
    val = _split_fingerprints(val_document)
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
) -> tuple[int, int]:
    document = json.loads(coco_path.read_text(encoding="utf-8"))
    images = _rows(document, "images")
    annotations = _rows(document, "annotations")
    categories = sorted(
        (int(row["id"]), str(row["name"])) for row in _rows(document, "categories")
    )
    if categories != [(1, "fastener"), (2, "pipe_joint")]:
        raise ValueError("INVALID_CATEGORY")
    image_root = output_root / "images" / split
    label_root = output_root / "labels" / split
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    for image in sorted(images, key=lambda row: int(row["id"])):
        image_id = int(image["id"])
        source = Path(str(image["file_name"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        suffix = source.suffix.lower() or ".jpg"
        destination = image_root / f"{image_id:06d}{suffix}"
        shutil.copy2(source, destination)
        width = float(image["width"])
        height = float(image["height"])
        lines: list[str] = []
        for annotation in sorted(
            annotations_by_image.get(image_id, []), key=lambda row: int(row["id"])
        ):
            x, y, box_width, box_height = (
                float(value) for value in annotation["bbox"]
            )
            class_index = (
                0 if merge_target_categories else int(annotation["category_id"]) - 1
            )
            center_x = (x + box_width / 2) / width
            center_y = (y + box_height / 2) / height
            normalized_width = box_width / width
            normalized_height = box_height / height
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
        (label_root / f"{image_id:06d}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="ascii",
        )
    return len(images), len(annotations)


def prepare_yolo_dataset(
    *,
    train_coco: Path,
    val_coco: Path,
    output_root: Path,
    runtime_output_root: Path | None = None,
    merge_target_categories: bool = False,
) -> dict[str, int]:
    """Materialize split-isolated YOLO images/labels and an ASCII runtime YAML."""

    train_document = json.loads(train_coco.read_text(encoding="utf-8"))
    val_document = json.loads(val_coco.read_text(encoding="utf-8"))
    _validate_split_isolation(train_document, val_document)
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
    )
    val_images, val_annotations = _materialize_split(
        val_coco,
        output_root,
        "val",
        merge_target_categories=merge_target_categories,
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
