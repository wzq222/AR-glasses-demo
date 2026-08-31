from __future__ import annotations

import pytest

from crrc_vision.marked_point_training import (
    experiment_manifest_fields,
    training_contract_for_experiment,
    validate_marked_point_training_documents,
)


def _doc(partition: str, start: int, count: int) -> dict:
    return {
        "info": {"partition": partition},
        "categories": [{"id": 1, "name": "marked_point"}],
        "images": [
            {
                "id": start + index,
                "file_name": f"{start + index}.jpg",
                "scene_group": f"scene-{start + index}",
            }
            for index in range(count)
        ],
        "annotations": [
            {
                "id": start + index,
                "image_id": start + index,
                "category_id": 1,
                "bbox": [1, 1, 10, 10],
            }
            for index in range(count)
        ],
    }


def test_marked_point_training_requires_one_business_class_and_scene_isolation() -> None:
    report = validate_marked_point_training_documents(
        _doc("train", 1, 30), _doc("val", 100, 17)
    )
    assert report == {
        "train_images": 30,
        "train_boxes": 30,
        "val_images": 17,
        "val_boxes": 17,
    }


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (lambda train, val: train["categories"].__setitem__(0, {"id": 1, "name": "fastener_target"}), "MARKED_POINT_CATEGORY_REQUIRED"),
        (lambda train, val: train.__setitem__("images", train["images"][:29]), "MARKED_POINT_TRAIN_IMAGES_INSUFFICIENT"),
        (lambda train, val: val.__setitem__("images", val["images"][:16]), "MARKED_POINT_VAL_IMAGES_INSUFFICIENT"),
        (lambda train, val: train.__setitem__("annotations", []), "MARKED_POINT_TRAIN_ANNOTATIONS_EMPTY"),
        (lambda train, val: train["images"][1].__setitem__("id", train["images"][0]["id"]), "MARKED_POINT_TRAIN_IMAGE_ID_DUPLICATE"),
        (lambda train, val: train["annotations"][1].__setitem__("id", train["annotations"][0]["id"]), "MARKED_POINT_TRAIN_ANNOTATION_ID_DUPLICATE"),
        (lambda train, val: train["annotations"][0].__setitem__("image_id", 999999), "MARKED_POINT_TRAIN_ANNOTATION_IMAGE_MISSING"),
        (lambda train, val: val["images"][0].__setitem__("scene_group", train["images"][0]["scene_group"]), "MARKED_POINT_SCENE_LEAKAGE"),
    ],
)
def test_marked_point_training_rejects_invalid_documents(mutation, error) -> None:
    train = _doc("train", 1, 30)
    val = _doc("val", 100, 17)
    mutation(train, val)
    with pytest.raises(ValueError, match=error):
        validate_marked_point_training_documents(train, val)


def test_marked_point_manifest_fields_freeze_business_target() -> None:
    assert experiment_manifest_fields() == {
        "experiment_kind": "marked_point_proposal",
        "business_target": "marked anti-loosening inspection point",
        "selection_metric": "proposal_recall_then_candidate_burden",
        "sealed_test_visible": False,
    }


def test_experiment_contract_validates_marked_point_documents() -> None:
    fields = training_contract_for_experiment(
        "marked_point_proposal", _doc("train", 1, 30), _doc("val", 100, 17)
    )
    assert fields["experiment_kind"] == "marked_point_proposal"
    assert fields["marked_point_counts"]["train_images"] == 30
    assert training_contract_for_experiment(
        "physical_target", {"images": []}, {"images": []}
    ) == {"experiment_kind": "physical_target"}
