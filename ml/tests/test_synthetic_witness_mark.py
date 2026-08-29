import cv2
import numpy as np
import pytest

from crrc_vision.synthetic_state import validate_state
from crrc_vision.synthetic_witness_mark import (
    extract_witness_mark_geometry,
    extract_witness_mark_mask,
    remove_existing_witness_mark,
)


def test_extracts_red_and_yellow_imagegen_paint_only_inside_roi() -> None:
    image = np.full((100, 120, 3), 120, dtype=np.uint8)
    cv2.line(image, (35, 50), (55, 50), (20, 30, 220), 5)
    cv2.line(image, (55, 50), (75, 55), (20, 210, 230), 5)
    cv2.circle(image, (105, 10), 5, (20, 30, 220), -1)

    mask = extract_witness_mark_mask(image, (25, 35, 85, 70))

    assert mask[50, 40] > 0
    assert mask[53, 70] > 0
    assert mask[10, 105] == 0


def test_gray_image_has_no_witness_mark() -> None:
    image = np.full((80, 80, 3), 140, dtype=np.uint8)

    mask = extract_witness_mark_mask(image, (10, 10, 70, 70))

    assert np.count_nonzero(mask) == 0


def test_baseline_removes_preexisting_rust_colored_pixels() -> None:
    baseline = np.full((90, 90, 3), 120, dtype=np.uint8)
    cv2.circle(baseline, (30, 30), 8, (25, 45, 145), -1)
    generated = baseline.copy()
    cv2.line(generated, (35, 55), (65, 55), (20, 210, 230), 5)

    mask = extract_witness_mark_mask(generated, (15, 15, 75, 75), baseline_image=baseline)

    assert mask[30, 30] == 0
    assert mask[55, 50] > 0


def test_geometry_recovers_three_state_angle_bands() -> None:
    for state, angle in (("NORMAL", 2.0), ("SLIGHT_LOOSE", 9.0), ("OBVIOUS_LOOSE", 25.0)):
        image = np.full((140, 140, 3), 130, dtype=np.uint8)
        anchor = np.array([70.0, 70.0])
        cv2.line(image, (25, 70), (70, 70), (20, 210, 230), 6, cv2.LINE_AA)
        radians = np.deg2rad(angle)
        moving_end = anchor + np.array([45.0 * np.cos(radians), 45.0 * np.sin(radians)])
        cv2.line(
            image,
            (70, 70),
            tuple(np.rint(moving_end).astype(int)),
            (20, 30, 220),
            6,
            cv2.LINE_AA,
        )
        mask = extract_witness_mark_mask(image, (15, 45, 125, 105))

        geometry = extract_witness_mark_geometry(mask)
        audit = validate_state(state, geometry.fixed_segment_xyxy, geometry.moving_segment_xyxy)

        assert audit.accepted, audit.reason
        assert geometry.mask_area > 100


def test_removes_existing_paint_only_near_target() -> None:
    image = np.full((120, 140, 3), 135, dtype=np.uint8)
    cv2.line(image, (45, 60), (85, 60), (15, 25, 230), 5, cv2.LINE_AA)
    cv2.circle(image, (125, 15), 6, (15, 25, 230), -1)

    cleaned, removed_mask = remove_existing_witness_mark(image, (25, 35, 105, 85))
    target_residual = extract_witness_mark_mask(cleaned, (25, 35, 105, 85))

    assert np.count_nonzero(removed_mask) > 100
    assert np.count_nonzero(target_residual) < 40
    assert cleaned[15, 125, 2] > 200


def test_rejects_large_red_region_as_unsafe_for_inpainting() -> None:
    image = np.full((120, 140, 3), 135, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (100, 90), (15, 25, 230), -1)

    with pytest.raises(RuntimeError, match="unsafe witness-paint removal"):
        remove_existing_witness_mark(image, (35, 35, 105, 95))
