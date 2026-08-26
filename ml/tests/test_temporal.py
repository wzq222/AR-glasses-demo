import numpy as np
import pytest

from crrc_vision.temporal import (
    HomographyQuality,
    propagate_between_scenes,
    propagate_box,
    validate_homography,
)


def test_translation_propagates_box() -> None:
    matrix = np.array([[1, 0, 5], [0, 1, 7], [0, 0, 1]], dtype=float)

    assert propagate_box((10, 20, 30, 40), matrix, 200, 100) == (
        15,
        27,
        35,
        47,
    )


def test_low_inlier_homography_is_rejected() -> None:
    quality = HomographyQuality(matches=40, inliers=7, median_error=1.0, scale=1.0)

    assert validate_homography(quality) == ("LOW_INLIER_RATIO",)


def test_all_homography_quality_failures_are_reported_stably() -> None:
    quality = HomographyQuality(matches=10, inliers=2, median_error=4.0, scale=1.5)

    assert validate_homography(quality) == (
        "TOO_FEW_MATCHES",
        "LOW_INLIER_RATIO",
        "HIGH_REPROJECTION_ERROR",
        "INVALID_SCALE",
    )


def test_cross_scene_propagation_is_rejected() -> None:
    with pytest.raises(ValueError, match="same scene"):
        propagate_between_scenes(
            "scene-a",
            "scene-b",
            (1, 2, 3, 4),
            np.eye(3),
            100,
            100,
        )
