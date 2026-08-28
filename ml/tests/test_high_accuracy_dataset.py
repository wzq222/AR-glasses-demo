from __future__ import annotations

import copy

import pytest

from crrc_vision.high_accuracy_dataset import (
    assemble_high_accuracy_dataset,
    assert_uncertain_matches_exclusions,
    repartition_complete_reviews,
)


CATEGORIES = [
    {"id": 1, "name": "fastener"},
    {"id": 2, "name": "pipe_joint"},
]


def _image(image_id: int, partition: str) -> dict[str, object]:
    return {
        "id": image_id,
        "image_id": image_id,
        "relative_path": f"image-{image_id:03d}.jpg",
        "file_name": f"image-{image_id:03d}.jpg",
        "sha256": f"{image_id:04x}".ljust(64, "0"),
        "scene_group": f"scene-{image_id:04d}",
        "width": 200,
        "height": 100,
        "split": partition,
        "synthetic": False,
        "image_review_status": "complete",
    }


def _partition() -> dict[str, object]:
    train = [_image(index, "train") for index in range(1, 117)]
    val = [_image(index, "val") for index in range(117, 148)]
    test = [_image(index, "sealed_test") for index in range(148, 178)]
    return {
        "schema_version": "high-accuracy-partition-v1",
        "sealed_test_opened": False,
        "train": train,
        "val": val,
        "sealed_test": test,
    }


def _coco(images: list[dict[str, object]]) -> dict[str, object]:
    return {
        "images": [copy.deepcopy(image) for image in images],
        "annotations": [
            {
                "id": index,
                "image_id": image["id"],
                "category_id": 1 if index % 2 else 2,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "area": 400.0,
                "iscrowd": 0,
            }
            for index, image in enumerate(images, 1)
        ],
        "categories": copy.deepcopy(CATEGORIES),
    }


def _documents():
    partition = _partition()
    train = partition["train"]
    val = partition["val"]
    test = partition["sealed_test"]
    existing = _coco(train[:64] + val[:16])
    new = {
        "train": _coco(train[64:]),
        "val": _coco(val[16:]),
        "sealed_test": _coco(test),
    }
    return partition, existing, new


def test_assembly_merges_frozen_80_and_maps_both_classes_to_one_target() -> None:
    partition, existing, new = _documents()

    result = assemble_high_accuracy_dataset(
        partition,
        existing,
        new,
        minimum_sealed_test_boxes=0,
    )

    assert result["categories"] == [{"id": 1, "name": "fastener_target"}]
    assert result["counts"] == {
        "train_scenes": 116,
        "val_scenes": 31,
        "test_scenes": 30,
    }
    assert len(result["train"]["annotations"]) == 116
    assert len(result["val"]["annotations"]) == 31
    assert len(result["sealed_test"]["annotations"]) == 30
    assert all(
        annotation["category_id"] == 1
        for split in ("train", "val", "sealed_test")
        for annotation in result[split]["annotations"]
    )


def test_uncertain_image_never_enters_complete_dataset() -> None:
    partition, existing, new = _documents()
    new["train"]["images"][0]["image_review_status"] = "uncertain"

    with pytest.raises(ValueError, match="UNCERTAIN_IMAGE_IN_COMPLETE_DATASET"):
        assemble_high_accuracy_dataset(
            partition, existing, new, minimum_sealed_test_boxes=0
        )


def test_original_reviewed_scene_cannot_move_to_sealed_test() -> None:
    partition, existing, new = _documents()
    existing["images"][0] = copy.deepcopy(partition["sealed_test"][0])

    with pytest.raises(ValueError, match="EXISTING_REVIEW_OUTSIDE_TRAIN_VAL"):
        assemble_high_accuracy_dataset(
            partition, existing, new, minimum_sealed_test_boxes=0
        )


def test_synthetic_sealed_test_image_is_rejected() -> None:
    partition, existing, new = _documents()
    new["sealed_test"]["images"][0]["synthetic"] = True

    with pytest.raises(ValueError, match="SYNTHETIC_SEALED_TEST_IMAGE"):
        assemble_high_accuracy_dataset(
            partition, existing, new, minimum_sealed_test_boxes=0
        )


def test_sealed_test_requires_30_scenes_and_200_boxes() -> None:
    partition, existing, new = _documents()
    new["sealed_test"]["images"].pop()
    new["sealed_test"]["annotations"].pop()

    with pytest.raises(ValueError, match="PARTITION_IMAGE_SET_MISMATCH"):
        assemble_high_accuracy_dataset(partition, existing, new)

    partition, existing, new = _documents()
    with pytest.raises(ValueError, match="SEALED_TEST_BOX_COUNT_TOO_LOW"):
        assemble_high_accuracy_dataset(partition, existing, new)


def test_declared_train_exclusions_preserve_quality_without_forcing_uncertain_images() -> None:
    partition, existing, new = _documents()
    excluded = copy.deepcopy(new["train"]["images"][:2])
    excluded_scenes = {str(row["scene_group"]) for row in excluded}
    new["train"]["images"] = [
        row
        for row in new["train"]["images"]
        if str(row["scene_group"]) not in excluded_scenes
    ]
    new["train"]["annotations"] = [
        row
        for row in new["train"]["annotations"]
        if row["image_id"] not in {image["id"] for image in excluded}
    ]

    result = assemble_high_accuracy_dataset(
        partition,
        existing,
        new,
        exclusion_document={
            "schema_version": "high-accuracy-exclusions-v1",
            "train": [
                {
                    "scene_group": image["scene_group"],
                    "image_id": image["id"],
                    "relative_path": image["relative_path"],
                    "sha256": image["sha256"],
                    "reason": "review_uncertain",
                }
                for image in excluded
            ],
            "val": [],
            "sealed_test": [],
        },
        minimum_sealed_test_boxes=0,
    )

    assert result["counts"]["train_scenes"] == 114
    assert result["exclusions"]["train"] == 2


def test_exclusions_cannot_hide_sealed_or_existing_reviewed_scenes() -> None:
    partition, existing, new = _documents()
    sealed = partition["sealed_test"][0]
    with pytest.raises(ValueError, match="SEALED_TEST_EXCLUSION_FORBIDDEN"):
        assemble_high_accuracy_dataset(
            partition,
            existing,
            new,
            exclusion_document={
                "schema_version": "high-accuracy-exclusions-v1",
                "train": [],
                "val": [],
                "sealed_test": [
                    {
                        "scene_group": sealed["scene_group"],
                        "image_id": sealed["id"],
                        "relative_path": sealed["relative_path"],
                        "sha256": sealed["sha256"],
                        "reason": "review_uncertain",
                    }
                ],
            },
            minimum_sealed_test_boxes=0,
        )

    existing_image = partition["train"][0]
    with pytest.raises(ValueError, match="EXISTING_REVIEW_EXCLUSION_FORBIDDEN"):
        assemble_high_accuracy_dataset(
            partition,
            existing,
            new,
            exclusion_document={
                "schema_version": "high-accuracy-exclusions-v1",
                "train": [
                    {
                        "scene_group": existing_image["scene_group"],
                        "image_id": existing_image["id"],
                        "relative_path": existing_image["relative_path"],
                        "sha256": existing_image["sha256"],
                        "reason": "review_uncertain",
                    }
                ],
                "val": [],
                "sealed_test": [],
            },
            minimum_sealed_test_boxes=0,
        )


def test_uncertain_manifest_must_exactly_match_declared_exclusions() -> None:
    document = {
        "schema_version": "high-accuracy-exclusions-v1",
        "train": [
            {
                "scene_group": "scene-1",
                "image_id": 7,
                "relative_path": "a.jpg",
                "sha256": "a" * 64,
                "reason": "review_uncertain",
            }
        ],
        "val": [],
        "sealed_test": [],
    }
    assert_uncertain_matches_exclusions([7], document, "train")
    with pytest.raises(ValueError, match="UNCERTAIN_EXCLUSION_MISMATCH"):
        assert_uncertain_matches_exclusions([7, 8], document, "train")
    with pytest.raises(ValueError, match="UNCERTAIN_EXCLUSION_MISMATCH"):
        assert_uncertain_matches_exclusions([7], document, "val")


def test_quality_quarantine_is_not_required_in_uncertain_manifest() -> None:
    document = {
        "schema_version": "high-accuracy-exclusions-v1",
        "train": [
            {
                "scene_group": "scene-1",
                "image_id": 7,
                "relative_path": "a.jpg",
                "sha256": "a" * 64,
                "reason": "review_uncertain",
            },
            {
                "scene_group": "scene-2",
                "image_id": 8,
                "relative_path": "b.jpg",
                "sha256": "b" * 64,
                "reason": "quality_quarantine",
            },
        ],
        "val": [],
        "sealed_test": [],
    }

    assert_uncertain_matches_exclusions([7], document, "train")


def test_complete_reviews_can_be_repartitioned_without_changing_boxes() -> None:
    partition, _, new = _documents()
    moved = copy.deepcopy(new["train"]["images"][0])
    moved_scene = str(moved["scene_group"])
    original_test = copy.deepcopy(partition["sealed_test"][0])
    partition["train"] = [
        row for row in partition["train"] if row["scene_group"] != moved_scene
    ] + [original_test]
    partition["sealed_test"] = [
        row
        for row in partition["sealed_test"]
        if row["scene_group"] != original_test["scene_group"]
    ] + [moved]

    result = repartition_complete_reviews(
        partition, [new["train"], new["val"]]
    )

    assert moved_scene in {
        str(row["scene_group"]) for row in result["sealed_test"]["images"]
    }
    assert len(result["train"]["annotations"]) == 51
    assert len(result["val"]["annotations"]) == 15
    assert len(result["sealed_test"]["annotations"]) == 1
