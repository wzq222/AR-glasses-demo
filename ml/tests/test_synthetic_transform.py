import numpy as np

from crrc_vision.synthetic_transform import (
    TransformLimits,
    apply_homography_points,
    sample_transform,
    transform_bbox,
)


def test_sampled_transform_stays_inside_limits_and_is_repeatable() -> None:
    limits = TransformLimits(
        rotation_deg=8,
        scale_min=0.85,
        scale_max=1.15,
        perspective_fraction=0.04,
    )
    first = sample_transform(640, 480, seed=20260829, limits=limits)
    second = sample_transform(640, 480, seed=20260829, limits=limits)
    np.testing.assert_allclose(first.matrix, second.matrix)
    assert abs(first.rotation_deg) <= 8
    assert 0.85 <= first.scale <= 1.15


def test_point_round_trip_error_is_below_two_pixels() -> None:
    transform = sample_transform(640, 480, seed=7, limits=TransformLimits())
    points = np.array([[100.0, 80.0], [240.0, 200.0]], dtype=np.float32)
    warped = apply_homography_points(points, transform.matrix)
    restored = apply_homography_points(warped, np.linalg.inv(transform.matrix))
    assert np.max(np.linalg.norm(restored - points, axis=1)) < 2.0


def test_bbox_uses_all_four_transformed_corners() -> None:
    matrix = np.array([[1.0, 0.2, 5.0], [0.1, 1.0, 7.0], [0.0, 0.0, 1.0]])
    bbox = transform_bbox((10.0, 20.0, 30.0, 40.0), matrix)
    assert bbox == (19.0, 28.0, 43.0, 50.0)
