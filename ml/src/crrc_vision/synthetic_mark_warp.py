from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class WarpedImageGenMark:
    image: np.ndarray
    fixed_segment_xyxy: Segment
    moving_segment_xyxy: Segment
    anchor_xy: Point
    transform: np.ndarray


def warp_imagegen_moving_mark(
    image: np.ndarray,
    mark_mask: np.ndarray,
    fixed_segment_xyxy: Segment,
    moving_segment_xyxy: Segment,
    angle_deg: float,
) -> WarpedImageGenMark:
    """Rotate existing moving-side ImageGen paint pixels without drawing paint."""
    if image.ndim != 3 or mark_mask.shape != image.shape[:2]:
        raise ValueError("image与mark_mask尺寸不一致")
    if np.count_nonzero(mark_mask) < 24:
        raise ValueError("ImageGen防松线像素不足")
    anchor = np.asarray(fixed_segment_xyxy[1], dtype=np.float64)
    moving_end = np.asarray(moving_segment_xyxy[1], dtype=np.float64)
    direction = moving_end - anchor
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        raise ValueError("ImageGen转动侧线段退化")
    direction /= length

    y_values, x_values = np.indices(mark_mask.shape)
    projection = (x_values - anchor[0]) * direction[0] + (y_values - anchor[1]) * direction[1]
    moving_mask = np.where((mark_mask > 0) & (projection >= 0.0), 255, 0).astype(np.uint8)
    if np.count_nonzero(moving_mask) < 12:
        raise ValueError("ImageGen转动侧防松线像素不足")

    removal_mask = cv2.dilate(moving_mask, np.ones((3, 3), np.uint8))
    cleaned = cv2.inpaint(image, removal_mask, 3.0, cv2.INPAINT_TELEA)
    matrix = cv2.getRotationMatrix2D((float(anchor[0]), float(anchor[1])), float(angle_deg), 1.0)
    height, width = image.shape[:2]
    warped_pixels = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    warped_mask = cv2.warpAffine(
        moving_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    alpha = cv2.GaussianBlur(warped_mask, (3, 3), 0).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    rendered = np.clip(
        warped_pixels.astype(np.float32) * alpha
        + cleaned.astype(np.float32) * (1.0 - alpha),
        0,
        255,
    ).astype(np.uint8)

    points = np.array([moving_segment_xyxy[0], moving_segment_xyxy[1]], dtype=np.float32)[None, :, :]
    transformed = cv2.transform(points, matrix)[0]
    moving: Segment = (
        (float(transformed[0, 0]), float(transformed[0, 1])),
        (float(transformed[1, 0]), float(transformed[1, 1])),
    )
    return WarpedImageGenMark(
        image=rendered,
        fixed_segment_xyxy=fixed_segment_xyxy,
        moving_segment_xyxy=moving,
        anchor_xy=(float(anchor[0]), float(anchor[1])),
        transform=matrix,
    )
