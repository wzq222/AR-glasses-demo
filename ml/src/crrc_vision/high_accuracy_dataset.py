"""Guarded assembly of the scene-isolated high-accuracy COCO datasets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from crrc_vision.high_accuracy_split import assert_partition_isolated


TARGET_CATEGORY = {"id": 1, "name": "fastener_target"}
SOURCE_TARGET_NAMES = {"fastener", "pipe_joint", "fastener_target"}
PARTITIONS = ("train", "val", "sealed_test")


def repartition_complete_reviews(
    partition_document: Mapping[str, object],
    reviewed_documents: list[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Route complete reviewed images and their boxes to a repaired partition."""

    assert_partition_isolated(partition_document)
    owner_by_scene: dict[str, str] = {}
    identity_by_scene: dict[str, tuple[object, ...]] = {}
    for split in PARTITIONS:
        for row in _rows(partition_document.get(split), f"partition {split}"):
            scene = str(row.get("scene_group") or "")
            owner_by_scene[scene] = split
            identity_by_scene[scene] = _partition_identity(row)

    output_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    split_by_image_id: dict[object, str] = {}
    seen_scenes: set[str] = set()
    categories: list[dict[str, Any]] | None = None
    for document in reviewed_documents:
        current_categories = _rows(document.get("categories"), "categories")
        if categories is None:
            categories = [dict(row) for row in current_categories]
        elif _category_names(document) != {
            row.get("id"): str(row.get("name") or "") for row in categories
        }:
            raise ValueError("REVIEW_CATEGORY_MISMATCH")
        local_split: dict[object, str] = {}
        for image in _rows(document.get("images"), "reviewed images"):
            scene = str(image.get("scene_group") or "")
            image_id = image.get("id", image.get("image_id"))
            split = owner_by_scene.get(scene)
            if split is None:
                raise ValueError(f"REVIEW_SCENE_NOT_IN_PARTITION:{scene}")
            if scene in seen_scenes or image_id in split_by_image_id:
                raise ValueError(f"DUPLICATE_REVIEWED_IDENTITY:{scene}:{image_id}")
            if image.get("image_review_status", "complete") != "complete":
                raise ValueError(f"UNCERTAIN_IMAGE_IN_COMPLETE_DATASET:{image_id}")
            if _review_identity(image) != identity_by_scene[scene]:
                raise ValueError(f"REVIEW_IDENTITY_MISMATCH:{scene}")
            copied = dict(image)
            copied["split"] = split
            output_images[split].append(copied)
            local_split[image_id] = split
            split_by_image_id[image_id] = split
            seen_scenes.add(scene)
        for annotation in _rows(document.get("annotations"), "reviewed annotations"):
            split = local_split.get(annotation.get("image_id"))
            if split is None:
                raise ValueError(
                    f"ANNOTATION_REFERENCES_UNKNOWN_IMAGE:{annotation.get('image_id')}"
                )
            output_annotations[split].append(dict(annotation))

    result: dict[str, dict[str, object]] = {}
    for split in PARTITIONS:
        images = sorted(output_images[split], key=lambda row: int(row["id"]))
        annotations = sorted(
            output_annotations[split],
            key=lambda row: (int(row["image_id"]), int(row.get("id") or 0)),
        )
        for index, annotation in enumerate(annotations, 1):
            annotation["id"] = index
        result[split] = {
            "info": {
                "schema_version": "high-accuracy-repartitioned-review-v1",
                "partition": split,
            },
            "images": images,
            "annotations": annotations,
            "categories": categories or [],
        }
    return result


def assert_uncertain_matches_exclusions(
    uncertain_image_ids: list[object],
    exclusion_document: Mapping[str, object],
    split: str,
) -> None:
    """Require every unresolved review to have one explicit quality exclusion."""

    if split not in PARTITIONS:
        raise ValueError(f"INVALID_PARTITION:{split}")
    if exclusion_document.get("schema_version") != "high-accuracy-exclusions-v1":
        raise ValueError("INVALID_HIGH_ACCURACY_EXCLUSIONS")
    rows = _rows(exclusion_document.get(split), f"exclusions {split}")
    review_rows = [
        row for row in rows if str(row.get("reason") or "") == "review_uncertain"
    ]
    excluded_ids = {row.get("image_id") for row in review_rows}
    if len(excluded_ids) != len(review_rows) or set(uncertain_image_ids) != excluded_ids:
        raise ValueError(
            f"UNCERTAIN_EXCLUSION_MISMATCH:{split}:"
            f"uncertain={len(set(uncertain_image_ids))}:excluded={len(excluded_ids)}"
        )


def _rows(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _partition_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(row.get("scene_group") or ""),
        row.get("image_id", row.get("id")),
        str(row.get("relative_path") or row.get("file_name") or ""),
        str(row.get("sha256") or "").upper(),
    )


def _review_identity(row: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(row.get("scene_group") or ""),
        row.get("id", row.get("image_id")),
        str(row.get("relative_path") or row.get("file_name") or ""),
        str(row.get("sha256") or "").upper(),
    )


def _category_names(document: Mapping[str, object]) -> dict[object, str]:
    categories = _rows(document.get("categories"), "categories")
    result: dict[object, str] = {}
    for row in categories:
        category_id = row.get("id")
        name = str(row.get("name") or "")
        if category_id is None or category_id in result or not name:
            raise ValueError("INVALID_SOURCE_CATEGORIES")
        result[category_id] = name
    return result


def assemble_high_accuracy_dataset(
    partition_document: Mapping[str, object],
    existing_reviewed_coco: Mapping[str, object],
    new_reviewed_by_partition: Mapping[str, Mapping[str, object]],
    *,
    exclusion_document: Mapping[str, object] | None = None,
    minimum_sealed_test_boxes: int = 200,
    required_existing_scenes: int = 80,
) -> dict[str, object]:
    """Merge complete reviewed scenes into frozen train/val/sealed-test COCO."""

    if partition_document.get("schema_version") != "high-accuracy-partition-v1":
        raise ValueError("INVALID_HIGH_ACCURACY_PARTITION")
    assert_partition_isolated(partition_document)
    expected_by_split: dict[str, dict[str, tuple[object, ...]]] = {}
    owner_by_scene: dict[str, str] = {}
    for split in PARTITIONS:
        rows = _rows(partition_document.get(split), f"partition {split}")
        expected: dict[str, tuple[object, ...]] = {}
        for row in rows:
            identity = _partition_identity(row)
            scene = str(identity[0])
            if not scene or None in identity or "" in identity:
                raise ValueError("INVALID_PARTITION_IDENTITY")
            if scene in expected:
                raise ValueError(f"DUPLICATE_PARTITION_SCENE:{scene}")
            expected[scene] = identity
            owner_by_scene[scene] = split
        expected_by_split[split] = expected

    excluded_by_split: dict[str, dict[str, tuple[object, ...]]] = {
        split: {} for split in PARTITIONS
    }
    if exclusion_document is not None:
        if exclusion_document.get("schema_version") != "high-accuracy-exclusions-v1":
            raise ValueError("INVALID_HIGH_ACCURACY_EXCLUSIONS")
        for split in PARTITIONS:
            for row in _rows(exclusion_document.get(split), f"exclusions {split}"):
                if split == "sealed_test":
                    raise ValueError("SEALED_TEST_EXCLUSION_FORBIDDEN")
                if str(row.get("reason") or "") not in {
                    "review_uncertain",
                    "quality_quarantine",
                }:
                    raise ValueError("INVALID_EXCLUSION_REASON")
                identity = _partition_identity(row)
                scene = str(identity[0])
                expected = expected_by_split[split].get(scene)
                if expected is None or identity != expected:
                    raise ValueError(f"EXCLUSION_IDENTITY_MISMATCH:{scene}")
                if scene in excluded_by_split[split]:
                    raise ValueError(f"DUPLICATE_EXCLUSION_SCENE:{scene}")
                excluded_by_split[split][scene] = identity

    existing_images = _rows(existing_reviewed_coco.get("images"), "existing images")
    if len(existing_images) != required_existing_scenes:
        raise ValueError(
            f"EXISTING_REVIEW_COUNT_MISMATCH:{len(existing_images)}:"
            f"expected={required_existing_scenes}"
        )
    for image in existing_images:
        scene = str(image.get("scene_group") or "")
        if owner_by_scene.get(scene) not in {
            "train",
            "val",
        }:
            raise ValueError("EXISTING_REVIEW_OUTSIDE_TRAIN_VAL")
        owner = owner_by_scene[scene]
        if scene in excluded_by_split[owner]:
            raise ValueError("EXISTING_REVIEW_EXCLUSION_FORBIDDEN")

    collected_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    collected_annotations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_scenes: set[str] = set()
    seen_image_ids: set[object] = set()

    def ingest(document: Mapping[str, object], declared_split: str | None) -> None:
        category_names = _category_names(document)
        images = _rows(document.get("images"), "reviewed images")
        image_split_by_id: dict[object, str] = {}
        for image in images:
            scene = str(image.get("scene_group") or "")
            image_id = image.get("id", image.get("image_id"))
            target_split = owner_by_scene.get(scene)
            if target_split is None:
                raise ValueError(f"REVIEW_SCENE_NOT_IN_PARTITION:{scene}")
            if declared_split is not None and target_split != declared_split:
                raise ValueError(
                    f"REVIEW_SCENE_WRONG_PARTITION:{scene}:{declared_split}:"
                    f"expected={target_split}"
                )
            if image.get("image_review_status", "complete") != "complete":
                raise ValueError(f"UNCERTAIN_IMAGE_IN_COMPLETE_DATASET:{image_id}")
            if scene in seen_scenes or image_id in seen_image_ids:
                raise ValueError(f"DUPLICATE_REVIEWED_IDENTITY:{scene}:{image_id}")
            if _review_identity(image) != expected_by_split[target_split][scene]:
                raise ValueError(f"REVIEW_IDENTITY_MISMATCH:{scene}")
            if target_split == "sealed_test" and image.get("synthetic") is not False:
                raise ValueError(f"SYNTHETIC_SEALED_TEST_IMAGE:{scene}")
            output_image = dict(image)
            output_image["id"] = image_id
            output_image["file_name"] = str(
                image.get("relative_path") or image.get("file_name") or ""
            )
            output_image["split"] = target_split
            collected_images[target_split].append(output_image)
            image_split_by_id[image_id] = target_split
            seen_scenes.add(scene)
            seen_image_ids.add(image_id)

        for annotation in _rows(document.get("annotations"), "reviewed annotations"):
            image_id = annotation.get("image_id")
            target_split = image_split_by_id.get(image_id)
            if target_split is None:
                raise ValueError(f"ANNOTATION_REFERENCES_UNKNOWN_IMAGE:{image_id}")
            category_name = category_names.get(annotation.get("category_id"))
            if category_name not in SOURCE_TARGET_NAMES:
                raise ValueError(f"INVALID_TARGET_CATEGORY:{category_name}")
            output_annotation = dict(annotation)
            output_annotation["category_id"] = 1
            collected_annotations[target_split].append(output_annotation)

    ingest(existing_reviewed_coco, None)
    if set(new_reviewed_by_partition) != set(PARTITIONS):
        raise ValueError("NEW_REVIEW_PARTITIONS_MISSING")
    for split in PARTITIONS:
        ingest(new_reviewed_by_partition[split], split)

    output: dict[str, object] = {"categories": [TARGET_CATEGORY]}
    for split in PARTITIONS:
        actual_scenes = {str(row["scene_group"]) for row in collected_images[split]}
        expected_scenes = set(expected_by_split[split]) - set(
            excluded_by_split[split]
        )
        if actual_scenes != expected_scenes:
            missing = len(expected_scenes - actual_scenes)
            extra = len(actual_scenes - expected_scenes)
            raise ValueError(
                f"PARTITION_IMAGE_SET_MISMATCH:{split}:missing={missing}:extra={extra}"
            )
        images = sorted(collected_images[split], key=lambda row: int(row["id"]))
        annotations = sorted(
            collected_annotations[split],
            key=lambda row: (int(row["image_id"]), int(row.get("id") or 0)),
        )
        for annotation_id, annotation in enumerate(annotations, 1):
            annotation["id"] = annotation_id
        output[split] = {
            "info": {
                "schema_version": "high-accuracy-coco-v1",
                "partition": split,
                "truth_tier": "reviewed-ai-calibration",
            },
            "images": images,
            "annotations": annotations,
            "categories": [TARGET_CATEGORY],
        }

    sealed_images = collected_images["sealed_test"]
    sealed_annotations = collected_annotations["sealed_test"]
    if len(sealed_images) < 30:
        raise ValueError(f"SEALED_TEST_SCENE_COUNT_TOO_LOW:{len(sealed_images)}")
    if len(sealed_annotations) < minimum_sealed_test_boxes:
        raise ValueError(
            f"SEALED_TEST_BOX_COUNT_TOO_LOW:{len(sealed_annotations)}:"
            f"required={minimum_sealed_test_boxes}"
        )
    if not collected_images["val"]:
        raise ValueError("EMPTY_VALIDATION_PARTITION")
    output["counts"] = {
        "train_scenes": len(collected_images["train"]),
        "val_scenes": len(collected_images["val"]),
        "test_scenes": len(sealed_images),
    }
    output["exclusions"] = {
        split: len(excluded_by_split[split]) for split in PARTITIONS
    }
    return output
