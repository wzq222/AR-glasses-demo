import numpy as np
import pytest

from crrc_vision.synthetic_composite import Placement, composite_sample


def test_composite_transforms_bbox_and_endpoints() -> None:
    background = np.zeros((240, 320, 3), dtype=np.uint8)
    patch = np.full((80, 100, 3), 140, dtype=np.uint8)
    mask = np.full((80, 100), 255, dtype=np.uint8)
    result = composite_sample(
        background,
        patch,
        mask,
        (10, 10, 90, 70),
        (((20, 40), (45, 40)), ((45, 40), (75, 45))),
        Placement(x=100, y=80, scale=1.0, rotation_deg=0.0),
    )
    assert result.bbox_xyxy == pytest.approx((110, 90, 190, 150), abs=1.0)
    assert result.image.shape == background.shape
    assert result.segments[0][0] == pytest.approx((120, 120), abs=1.0)


def test_composite_rejects_border_collision() -> None:
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    patch = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.full((80, 80), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="边界"):
        composite_sample(
            background,
            patch,
            mask,
            (0, 0, 80, 80),
            (((10, 10), (20, 10)), ((20, 10), (30, 15))),
            Placement(x=60, y=60, scale=1.0, rotation_deg=0.0),
        )


def test_composite_rejects_tiny_target() -> None:
    background = np.zeros((240, 320, 3), dtype=np.uint8)
    patch = np.zeros((80, 80, 3), dtype=np.uint8)
    mask = np.full((80, 80), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="短边"):
        composite_sample(
            background,
            patch,
            mask,
            (10, 10, 30, 30),
            (((10, 10), (20, 10)), ((20, 10), (30, 15))),
            Placement(x=100, y=80, scale=0.5, rotation_deg=0.0),
        )
