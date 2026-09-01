import numpy as np

from crrc_vision.mobile_runtime_parity import (
    decode_yolo_predictions,
    letterbox_rgb,
    predict_image,
)


def test_letterbox_matches_android_odd_padding_geometry() -> None:
    image = np.zeros((480, 641, 3), dtype=np.uint8)

    tensor, transform = letterbox_rgb(image, target_size=640)

    assert tensor.shape == (1, 3, 640, 640)
    assert transform.resized_width == 640
    assert transform.resized_height == 479
    assert transform.pad_left == 0
    assert transform.pad_top == 80
    assert transform.pad_right == 0
    assert transform.pad_bottom == 81


def test_decode_matches_android_class_agnostic_nms() -> None:
    prediction = np.array(
        [
            [100.0, 105.0, 300.0],
            [100.0, 105.0, 300.0],
            [100.0, 100.0, 40.0],
            [100.0, 100.0, 40.0],
            [0.90, 0.10, 0.10],
            [0.10, 0.80, 0.70],
        ],
        dtype=np.float32,
    )

    detections = decode_yolo_predictions(
        prediction,
        image_id=7,
        original_width=640,
        original_height=640,
        scale=1.0,
        pad_x=0.0,
        pad_y=0.0,
        confidence_threshold=0.20,
        nms_iou_threshold=0.45,
    )

    assert len(detections) == 2
    assert detections[0]["image_id"] == 7
    assert detections[0]["category_id"] == 0
    assert detections[0]["score"] == np.float32(0.90)
    assert detections[1]["category_id"] == 1
    assert detections[1]["score"] == np.float32(0.70)


def test_decode_reverses_letterbox_and_clips() -> None:
    prediction = np.array(
        [[10.0], [130.0], [40.0], [80.0], [0.90], [0.10]],
        dtype=np.float32,
    )

    detections = decode_yolo_predictions(
        prediction,
        image_id=1,
        original_width=1280,
        original_height=720,
        scale=0.5,
        pad_x=0.0,
        pad_y=140.0,
    )

    assert detections[0]["bbox"] == [0.0, 0.0, 60.0, 60.0]


def test_predict_image_requires_frozen_output_contract() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    def wrong_shape(tensor: np.ndarray) -> np.ndarray:
        assert tensor.shape == (1, 3, 640, 640)
        return np.zeros((1, 6, 100), dtype=np.float32)

    try:
        predict_image(wrong_shape, image, image_id=3)
    except ValueError as error:
        assert str(error) == "YOLO_OUTPUT_SHAPE_MISMATCH:(1, 6, 100)"
    else:
        raise AssertionError("wrong runtime output shape was accepted")
