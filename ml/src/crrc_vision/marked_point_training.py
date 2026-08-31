from __future__ import annotations

from collections.abc import Mapping


def experiment_manifest_fields() -> dict[str, object]:
    return {
        "experiment_kind": "marked_point_proposal",
        "business_target": "marked anti-loosening inspection point",
        "selection_metric": "proposal_recall_then_candidate_burden",
        "sealed_test_visible": False,
    }


def _rows(document: Mapping[str, object], key: str, prefix: str) -> list[dict]:
    value = document.get(key)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"MARKED_POINT_{prefix}_{key.upper()}_INVALID")
    return value


def _validate_document(
    document: Mapping[str, object],
    *,
    partition: str,
    minimum_images: int,
) -> tuple[list[dict], list[dict]]:
    prefix = partition.upper()
    info = document.get("info")
    if not isinstance(info, Mapping) or info.get("partition") != partition:
        raise ValueError(f"MARKED_POINT_{prefix}_PARTITION_INVALID")
    categories = _rows(document, "categories", prefix)
    if categories != [{"id": 1, "name": "marked_point"}]:
        raise ValueError("MARKED_POINT_CATEGORY_REQUIRED")
    images = _rows(document, "images", prefix)
    annotations = _rows(document, "annotations", prefix)
    if len(images) < minimum_images:
        raise ValueError(f"MARKED_POINT_{prefix}_IMAGES_INSUFFICIENT")
    if not annotations:
        raise ValueError(f"MARKED_POINT_{prefix}_ANNOTATIONS_EMPTY")

    image_ids = [row.get("id") for row in images]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"MARKED_POINT_{prefix}_IMAGE_ID_DUPLICATE")
    annotation_ids = [row.get("id") for row in annotations]
    if len(set(annotation_ids)) != len(annotation_ids):
        raise ValueError(f"MARKED_POINT_{prefix}_ANNOTATION_ID_DUPLICATE")
    image_id_set = set(image_ids)
    if any(row.get("image_id") not in image_id_set for row in annotations):
        raise ValueError(f"MARKED_POINT_{prefix}_ANNOTATION_IMAGE_MISSING")
    if any(row.get("category_id") != 1 for row in annotations):
        raise ValueError(f"MARKED_POINT_{prefix}_ANNOTATION_CATEGORY_INVALID")
    if any(not str(row.get("scene_group") or "").strip() for row in images):
        raise ValueError(f"MARKED_POINT_{prefix}_SCENE_GROUP_MISSING")
    return images, annotations


def validate_marked_point_training_documents(
    train: Mapping[str, object],
    val: Mapping[str, object],
) -> dict[str, int]:
    train_images, train_annotations = _validate_document(
        train, partition="train", minimum_images=30
    )
    val_images, val_annotations = _validate_document(
        val, partition="val", minimum_images=17
    )
    train_scenes = {str(row["scene_group"]) for row in train_images}
    val_scenes = {str(row["scene_group"]) for row in val_images}
    if train_scenes & val_scenes:
        raise ValueError("MARKED_POINT_SCENE_LEAKAGE")
    return {
        "train_images": len(train_images),
        "train_boxes": len(train_annotations),
        "val_images": len(val_images),
        "val_boxes": len(val_annotations),
    }


def training_contract_for_experiment(
    experiment_kind: str,
    train: Mapping[str, object],
    val: Mapping[str, object],
) -> dict[str, object]:
    if experiment_kind == "physical_target":
        return {"experiment_kind": experiment_kind}
    if experiment_kind != "marked_point_proposal":
        raise ValueError(f"INVALID_EXPERIMENT_KIND:{experiment_kind}")
    return {
        **experiment_manifest_fields(),
        "marked_point_counts": validate_marked_point_training_documents(train, val),
    }
