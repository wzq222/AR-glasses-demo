import numpy as np
import pytest
from pathlib import Path

from crrc_vision.detection_fusion import (
    fuse_image_detections,
    merge_target_classes,
    merge_target_coco_document,
    merge_target_coco_predictions,
    nms_image_detections,
    runtime_path_text,
    select_detection_mode,
    to_coco_predictions,
)


def test_merge_target_classes_maps_every_detection_to_physical_target() -> None:
    detections = np.asarray(
        [[0, 0.7, 1, 2, 3, 4], [1, 0.6, 5, 6, 7, 8]], dtype=np.float32
    )

    merged = merge_target_classes(detections)

    assert merged[:, 0].tolist() == [0.0, 0.0]
    assert detections[:, 0].tolist() == [0.0, 1.0]


def test_cross_tile_nms_removes_duplicate_boundary_detection() -> None:
    tiled = np.asarray(
        [
            [0, 0.90, 100, 100, 140, 140],
            [0, 0.80, 102, 101, 141, 139],
            [0, 0.70, 200, 200, 230, 230],
        ],
        dtype=np.float32,
    )

    deduplicated = nms_image_detections(tiled, iou_threshold=0.5)

    assert deduplicated.shape == (2, 6)
    assert deduplicated[:, 1].tolist() == pytest.approx([0.9, 0.7])


def test_coco_category_merge_keeps_inputs_unchanged() -> None:
    ground_truth = {
        "images": [{"id": 1}],
        "annotations": [{"id": 2, "image_id": 1, "category_id": 2}],
        "categories": [{"id": 1, "name": "fastener"}, {"id": 2, "name": "pipe_joint"}],
    }
    predictions = [{"image_id": 1, "category_id": 2, "score": 0.9}]

    merged_truth = merge_target_coco_document(ground_truth)
    merged_predictions = merge_target_coco_predictions(predictions)

    assert merged_truth["categories"] == [{"id": 1, "name": "fastener_target"}]
    assert merged_truth["annotations"][0]["category_id"] == 1
    assert merged_predictions[0]["category_id"] == 1
    assert ground_truth["annotations"][0]["category_id"] == 2
    assert predictions[0]["category_id"] == 2


def test_fusion_removes_same_class_overlap_but_keeps_other_classes() -> None:
    full = np.asarray(
        [
            [0, 0.70, 10, 10, 30, 30],
            [1, 0.65, 10, 10, 30, 30],
        ],
        dtype=np.float32,
    )
    sliced = np.asarray(
        [
            [0, 0.90, 11, 11, 31, 31],
            [0, 0.80, 70, 70, 90, 90],
        ],
        dtype=np.float32,
    )

    fused = fuse_image_detections(full, sliced, iou_threshold=0.5)

    assert fused.shape == (3, 6)
    assert fused[:, 1].tolist() == pytest.approx([0.9, 0.8, 0.65])
    assert sorted(fused[:, 0].astype(int).tolist()) == [0, 0, 1]


def test_coco_conversion_splits_flat_results_and_clips_to_image() -> None:
    boxes = np.asarray(
        [
            [0, 0.9, -5, -6, 20, 30],
            [1, 0.8, 90, 70, 120, 100],
        ],
        dtype=np.float32,
    )

    predictions = to_coco_predictions(
        image_id=7,
        detections=boxes,
        image_width=100,
        image_height=80,
    )

    assert predictions == [
        {"image_id": 7, "category_id": 1, "bbox": [0.0, 0.0, 20.0, 30.0], "score": pytest.approx(0.9)},
        {"image_id": 7, "category_id": 2, "bbox": [90.0, 70.0, 10.0, 10.0], "score": pytest.approx(0.8)},
    ]


def test_runtime_path_preserves_windows_junction_spelling() -> None:
    path = Path("E:/crrc_vision_data/model")

    assert "crrc_vision_data" in runtime_path_text(path)


def test_sliced_mode_does_not_allow_full_boxes_to_suppress_tile_boxes() -> None:
    full = np.asarray([[0, 0.95, 0, 0, 20, 20]], dtype=np.float32)
    sliced = np.asarray([[0, 0.80, 2, 2, 18, 18]], dtype=np.float32)

    selected = select_detection_mode(full, sliced, mode="sliced", iou_threshold=0.5)

    assert selected.tolist() == sliced.tolist()
