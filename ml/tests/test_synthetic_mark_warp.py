import cv2
import numpy as np

from crrc_vision.synthetic_mark_warp import warp_imagegen_moving_mark
from crrc_vision.synthetic_state import validate_state


def test_warp_uses_existing_imagegen_mark_pixels_and_hits_slight_band() -> None:
    image = np.full((120, 140, 3), 135, dtype=np.uint8)
    cv2.line(image, (25, 60), (115, 60), (20, 210, 230), 7, cv2.LINE_AA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mark_mask = cv2.inRange(hsv, np.array([14, 80, 60]), np.array([42, 255, 255]))
    fixed = ((25.0, 60.0), (70.0, 60.0))
    moving = ((70.0, 60.0), (115.0, 60.0))

    result = warp_imagegen_moving_mark(image, mark_mask, fixed, moving, 8.0)

    audit = validate_state("SLIGHT_LOOSE", result.fixed_segment_xyxy, result.moving_segment_xyxy)
    assert audit.accepted, audit.reason
    assert np.count_nonzero(result.image[:, :, 2] - result.image[:, :, 0] > 80) > 0
    assert result.image.shape == image.shape


def test_warp_rejects_empty_imagegen_mark() -> None:
    image = np.full((80, 80, 3), 120, dtype=np.uint8)
    mask = np.zeros((80, 80), dtype=np.uint8)

    try:
        warp_imagegen_moving_mark(
            image,
            mask,
            ((10.0, 40.0), (40.0, 40.0)),
            ((40.0, 40.0), (70.0, 40.0)),
            8.0,
        )
    except ValueError as exc:
        assert "ImageGen" in str(exc)
    else:
        raise AssertionError("empty mark must fail")
