"""Guarded geometry primitives for same-scene temporal propagation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class HomographyQuality:
    matches: int
    inliers: int
    median_error: float
    scale: float

    def __post_init__(self) -> None:
        if self.matches < 0 or self.inliers < 0:
            raise ValueError("match counts must be non-negative")
        if self.inliers > self.matches:
            raise ValueError("inliers cannot exceed matches")
        if self.median_error < 0:
            raise ValueError("median reprojection error must be non-negative")


def validate_homography(quality: HomographyQuality) -> tuple[str, ...]:
    """Return stable rejection codes; an empty tuple means propagation is allowed."""

    errors: list[str] = []
    if quality.matches < 20:
        errors.append("TOO_FEW_MATCHES")
    if quality.matches and quality.inliers / quality.matches < 0.35:
        errors.append("LOW_INLIER_RATIO")
    if quality.median_error > 3.0:
        errors.append("HIGH_REPROJECTION_ERROR")
    if not 0.75 <= quality.scale <= 1.33:
        errors.append("INVALID_SCALE")
    return tuple(errors)


def _validate_box(box: Box) -> None:
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid xyxy box: {box}")


def propagate_box(
    xyxy: Box,
    homography: NDArray[np.floating],
    image_width: int,
    image_height: int,
) -> Box:
    """Project all four box corners, bound them, and clip to the target image."""

    _validate_box(xyxy)
    matrix = np.asarray(homography, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("target image dimensions must be positive")

    x1, y1, x2, y2 = xyxy
    corners = np.array(
        [[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]],
        dtype=float,
    )
    projected = (matrix @ corners.T).T
    denominators = projected[:, 2]
    if np.any(np.isclose(denominators, 0.0)):
        raise ValueError("homography projects a corner to infinity")
    projected_xy = projected[:, :2] / denominators[:, None]
    if not np.isfinite(projected_xy).all():
        raise ValueError("homography produced non-finite coordinates")

    mapped = (
        max(0.0, min(float(image_width), float(projected_xy[:, 0].min()))),
        max(0.0, min(float(image_height), float(projected_xy[:, 1].min()))),
        max(0.0, min(float(image_width), float(projected_xy[:, 0].max()))),
        max(0.0, min(float(image_height), float(projected_xy[:, 1].max()))),
    )
    _validate_box(mapped)
    return mapped


def propagate_between_scenes(
    source_scene: str,
    target_scene: str,
    xyxy: Box,
    homography: NDArray[np.floating],
    image_width: int,
    image_height: int,
) -> Box:
    """Apply a homography only inside one explicitly identified scene."""

    if not source_scene or source_scene != target_scene:
        raise ValueError("temporal propagation requires the same scene")
    return propagate_box(xyxy, homography, image_width, image_height)
