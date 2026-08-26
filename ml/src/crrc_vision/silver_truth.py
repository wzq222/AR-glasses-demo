"""Whole-image quality gates for isolated AI silver annotations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageReport:
    errors: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class SilverReport:
    errors: tuple[str, ...]
    train_groups: int
    val_groups: int
    train_images: int
    val_images: int
    synthetic_train_images: int

    @property
    def can_train(self) -> bool:
        return not self.errors


def _valid_bbox(value: object, width: object, height: object) -> bool:
    if not (
        isinstance(value, list)
        and len(value) == 4
        and all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            for coordinate in value
        )
        and isinstance(width, (int, float))
        and not isinstance(width, bool)
        and isinstance(height, (int, float))
        and not isinstance(height, bool)
        and width > 0
        and height > 0
    ):
        return False
    x, y, box_width, box_height = value
    return (
        x >= 0
        and y >= 0
        and box_width > 0
        and box_height > 0
        and x + box_width <= width
        and y + box_height <= height
    )


def evaluate_image(
    image: dict[str, object],
    annotations: list[dict[str, object]],
) -> ImageReport:
    """Require an explicit whole-image decision in addition to box decisions."""

    errors: set[str] = set()
    image_status = image.get("image_review_status")
    if image_status not in {"complete", "accept_empty"}:
        errors.add("IMAGE_NOT_COMPLETE")
    if any(row.get("review_status") != "accept" for row in annotations):
        errors.add("UNRESOLVED_CANDIDATE")
    if image_status == "accept_empty" and annotations:
        errors.add("BOX_ON_ACCEPTED_EMPTY_IMAGE")
    if any(
        not _valid_bbox(row.get("bbox"), image.get("width"), image.get("height"))
        for row in annotations
    ):
        errors.add("INVALID_ANNOTATION_BOX")
    return ImageReport(tuple(sorted(errors)))


def _dict_rows(value: object) -> list[dict[str, Any]] | None:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        return None
    return value


def evaluate_dataset(document: dict[str, object]) -> SilverReport:
    """Evaluate whether a reviewed silver document is safe to use for training."""

    errors: set[str] = set()
    images = _dict_rows(document.get("images"))
    annotations = _dict_rows(document.get("annotations"))
    categories = _dict_rows(document.get("categories"))
    if images is None:
        images = []
        errors.add("INVALID_IMAGES")
    if annotations is None:
        annotations = []
        errors.add("INVALID_ANNOTATIONS")
    if categories is None:
        categories = []
        errors.add("INVALID_CATEGORIES")

    category_ids = {
        row.get("id")
        for row in categories
        if row.get("name") in {"fastener", "pipe_joint"}
    }
    category_names = {row.get("name") for row in categories}
    if category_names != {"fastener", "pipe_joint"} or len(category_ids) != 2:
        errors.add("INVALID_CATEGORY_SCHEMA")

    image_ids = [row.get("id") for row in images]
    if any(image_id is None for image_id in image_ids) or any(
        count > 1 for count in Counter(image_ids).values()
    ):
        errors.add("INVALID_IMAGE_ID")
    by_image: dict[object, list[dict[str, object]]] = {
        image_id: [] for image_id in image_ids if image_id is not None
    }

    annotation_ids = [row.get("id") for row in annotations]
    if any(annotation_id is None for annotation_id in annotation_ids) or any(
        count > 1 for count in Counter(annotation_ids).values()
    ):
        errors.add("INVALID_ANNOTATION_ID")
    for annotation in annotations:
        image_id = annotation.get("image_id")
        if image_id not in by_image:
            errors.add("UNKNOWN_IMAGE_REFERENCE")
            continue
        if annotation.get("category_id") not in category_ids:
            errors.add("UNKNOWN_CATEGORY_REFERENCE")
        by_image[image_id].append(annotation)

    for image in images:
        image_id = image.get("id")
        errors.update(evaluate_image(image, by_image.get(image_id, [])).errors)
        if image.get("split") not in {"train", "val"}:
            errors.add("INVALID_SPLIT")
        if not isinstance(image.get("scene_group"), str) or not image.get(
            "scene_group"
        ):
            errors.add("INVALID_SCENE_GROUP")

    train_images = [row for row in images if row.get("split") == "train"]
    val_images = [row for row in images if row.get("split") == "val"]
    train_groups = {
        row["scene_group"]
        for row in train_images
        if isinstance(row.get("scene_group"), str) and row["scene_group"]
    }
    val_groups = {
        row["scene_group"]
        for row in val_images
        if isinstance(row.get("scene_group"), str) and row["scene_group"]
    }
    if len(train_groups) < 64:
        errors.add("INSUFFICIENT_TRAIN_GROUPS")
    if len(val_groups) < 16:
        errors.add("INSUFFICIENT_VAL_GROUPS")
    if train_groups & val_groups:
        errors.add("SCENE_GROUP_LEAKAGE")
    if any(bool(row.get("synthetic")) for row in val_images):
        errors.add("SYNTHETIC_VALIDATION_IMAGE")

    synthetic_train_images = sum(
        1 for row in train_images if bool(row.get("synthetic"))
    )
    if train_images and synthetic_train_images / len(train_images) > 0.20:
        errors.add("SYNTHETIC_TRAIN_RATIO_EXCEEDED")

    return SilverReport(
        errors=tuple(sorted(errors)),
        train_groups=len(train_groups),
        val_groups=len(val_groups),
        train_images=len(train_images),
        val_images=len(val_images),
        synthetic_train_images=synthetic_train_images,
    )


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _document_hash(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper()


def export_silver(
    document: dict[str, object],
    output_root: Path,
    *,
    integrity: dict[str, object] | None = None,
) -> int:
    """Write either an isolated silver dataset or a refusal, never both."""

    output_root.mkdir(parents=True, exist_ok=True)
    reserved = [
        output_root / "silver-refusal.json",
        output_root / "instances.silver.json",
        output_root / "silver-manifest.json",
        output_root / "accepted-images.json",
        output_root / "uncertain-images.json",
    ]
    existing = [path for path in reserved if path.exists()]
    if existing:
        raise FileExistsError(f"silver export already exists: {existing[0]}")

    report = evaluate_dataset(document)
    source_hash = _document_hash(document)
    if not report.can_train:
        _atomic_json(
            output_root / "silver-refusal.json",
            {
                "schema_version": "silver-refusal-v1",
                "source_document_sha256": source_hash,
                "errors": list(report.errors),
                "train_groups": report.train_groups,
                "val_groups": report.val_groups,
                "train_images": report.train_images,
                "val_images": report.val_images,
                "integrity": integrity or {},
            },
        )
        return 2

    silver = json.loads(json.dumps(document, ensure_ascii=False))
    info = silver.setdefault("info", {})
    if not isinstance(info, dict):
        raise ValueError("COCO info must be an object")
    info.update(
        {
            "schema_version": "ai-silver-truth-v1",
            "truth_tier": "silver",
            "production_metrics_allowed": False,
        }
    )
    instances_path = output_root / "instances.silver.json"
    _atomic_json(instances_path, silver)
    images = silver["images"]
    accepted = [row["id"] for row in images]
    _atomic_json(output_root / "accepted-images.json", accepted)
    _atomic_json(output_root / "uncertain-images.json", [])
    _atomic_json(
        output_root / "silver-manifest.json",
        {
            "schema_version": "silver-manifest-v1",
            "source_document_sha256": source_hash,
            "instances_sha256": hashlib.sha256(instances_path.read_bytes())
            .hexdigest()
            .upper(),
            "train_groups": report.train_groups,
            "val_groups": report.val_groups,
            "train_images": report.train_images,
            "val_images": report.val_images,
            "synthetic_train_images": report.synthetic_train_images,
            "production_metrics_allowed": False,
            "integrity": integrity or {},
        },
    )
    return 0
