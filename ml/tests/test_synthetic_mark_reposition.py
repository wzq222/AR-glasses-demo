import cv2
import numpy as np

from crrc_vision.synthetic_mark_reposition import reposition_imagegen_mark
from crrc_vision.synthetic_state import validate_state


def test_reposition_uses_existing_imagegen_pixels_and_exact_target_geometry() -> None:
    base = np.full((100, 120, 3), 110, dtype=np.uint8)
    donor = np.full_like(base, 110)
    donor_mask = np.zeros(base.shape[:2], dtype=np.uint8)
    cv2.line(donor, (10, 50), (90, 50), (10, 210, 245), 5, cv2.LINE_AA)
    cv2.line(donor_mask, (10, 50), (90, 50), 255, 5, cv2.LINE_AA)

    result = reposition_imagegen_mark(
        base,
        donor,
        donor_mask,
        donor_segment_xyxy=((10.0, 50.0), (90.0, 50.0)),
        fixed_target_xyxy=((30.0, 40.0), (60.0, 40.0)),
        moving_target_xyxy=((60.0, 40.0), (90.0, 46.0)),
    )

    audit = validate_state("SLIGHT_LOOSE", result.fixed_segment_xyxy, result.moving_segment_xyxy)
    assert audit.accepted
    assert np.count_nonzero(result.mark_mask) > 30
    assert np.count_nonzero(result.image != base) > 30
    assert tuple(result.image[40, 45]) != tuple(base[40, 45])
    assert tuple(result.image[10, 10]) == tuple(base[10, 10])


def test_reposition_rejects_empty_imagegen_donor() -> None:
    image = np.full((40, 40, 3), 100, dtype=np.uint8)
    try:
        reposition_imagegen_mark(
            image,
            image,
            np.zeros((40, 40), dtype=np.uint8),
            donor_segment_xyxy=((5.0, 20.0), (35.0, 20.0)),
            fixed_target_xyxy=((5.0, 10.0), (20.0, 10.0)),
            moving_target_xyxy=((20.0, 10.0), (35.0, 10.0)),
        )
    except ValueError as exc:
        assert "ImageGen" in str(exc)
    else:
        raise AssertionError("empty donor must fail")
