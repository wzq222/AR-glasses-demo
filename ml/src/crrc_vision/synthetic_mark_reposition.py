from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class RepositionedImageGenMark:
    image: np.ndarray
    mark_mask: np.ndarray
    fixed_segment_xyxy: Segment
    moving_segment_xyxy: Segment
    anchor_xy: Point


def _segment_transform(
    source: Segment,
    target: Segment,
    source_width: float,
    target_width: float,
) -> np.ndarray:
    source_start = np.asarray(source[0], dtype=np.float32)
    source_end = np.asarray(source[1], dtype=np.float32)
    target_start = np.asarray(target[0], dtype=np.float32)
    target_end = np.asarray(target[1], dtype=np.float32)
    source_vector = source_end - source_start
    target_vector = target_end - target_start
    source_length = float(np.linalg.norm(source_vector))
    target_length = float(np.linalg.norm(target_vector))
    if source_length < 1e-6 or target_length < 1e-6:
        raise ValueError("防松线线段退化")
    source_perpendicular = np.array([-source_vector[1], source_vector[0]], dtype=np.float32)
    source_perpendicular /= source_length
    target_perpendicular = np.array([-target_vector[1], target_vector[0]], dtype=np.float32)
    target_perpendicular /= target_length
    source_points = np.float32([
        source_start,
        source_end,
        source_start + source_perpendicular * source_width,
    ])
    target_points = np.float32([
        target_start,
        target_end,
        target_start + target_perpendicular * target_width,
    ])
    return cv2.getAffineTransform(source_points, target_points)


def _warp_paint(
    donor: np.ndarray,
    donor_mask: np.ndarray,
    source: Segment,
    target: Segment,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    source_length = float(np.linalg.norm(np.asarray(source[1]) - np.asarray(source[0])))
    source_width = max(2.0, float(np.count_nonzero(donor_mask)) / max(source_length, 1.0))
    target_width = max(4.0, min(10.0, source_width * 0.45))
    matrix = _segment_transform(source, target, source_width, target_width)
    width, height = size
    pixels = cv2.warpAffine(
        donor,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask = cv2.warpAffine(
        donor_mask,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return pixels, mask


def reposition_imagegen_mark(
    base_image: np.ndarray,
    donor_image: np.ndarray,
    donor_mask: np.ndarray,
    *,
    donor_segment_xyxy: Segment,
    fixed_target_xyxy: Segment,
    moving_target_xyxy: Segment,
) -> RepositionedImageGenMark:
    """Place only existing ImageGen paint pixels on reviewed mechanical interfaces."""
    if base_image.ndim != 3 or donor_image.shape != base_image.shape:
        raise ValueError("base_image与donor_image尺寸不一致")
    if donor_mask.shape != base_image.shape[:2] or np.count_nonzero(donor_mask) < 24:
        raise ValueError("ImageGen防松线供体像素不足")
    height, width = base_image.shape[:2]
    rendered = base_image.copy()
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    source_start = np.asarray(donor_segment_xyxy[0], dtype=np.float64)
    source_end = np.asarray(donor_segment_xyxy[1], dtype=np.float64)
    source_vector = source_end - source_start
    source_length_squared = float(source_vector @ source_vector)
    if source_length_squared < 1e-6:
        raise ValueError("ImageGen防松线供体线段退化")
    y_values, x_values = np.indices(donor_mask.shape)
    projection = (
        (x_values - source_start[0]) * source_vector[0]
        + (y_values - source_start[1]) * source_vector[1]
    ) / source_length_squared
    midpoint = tuple(((source_start + source_end) / 2.0).tolist())
    fixed_mask = np.where((donor_mask > 0) & (projection <= 0.5), donor_mask, 0).astype(np.uint8)
    moving_mask = np.where((donor_mask > 0) & (projection > 0.5), donor_mask, 0).astype(np.uint8)
    if min(np.count_nonzero(fixed_mask), np.count_nonzero(moving_mask)) < 12:
        raise ValueError("ImageGen防松线供体无法切分为两侧")
    source_targets = (
        (((float(source_start[0]), float(source_start[1])), midpoint), fixed_mask, fixed_target_xyxy),
        ((midpoint, (float(source_end[0]), float(source_end[1]))), moving_mask, moving_target_xyxy),
    )
    for source, source_mask, target in source_targets:
        pixels, mask = _warp_paint(
            donor_image,
            source_mask,
            source,
            target,
            (width, height),
        )
        alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
        rendered = np.clip(
            pixels.astype(np.float32) * alpha + rendered.astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        combined_mask = cv2.max(combined_mask, mask)
    anchor = (
        (fixed_target_xyxy[1][0] + moving_target_xyxy[0][0]) / 2.0,
        (fixed_target_xyxy[1][1] + moving_target_xyxy[0][1]) / 2.0,
    )
    return RepositionedImageGenMark(
        image=rendered,
        mark_mask=np.where(combined_mask > 0, 255, 0).astype(np.uint8),
        fixed_segment_xyxy=fixed_target_xyxy,
        moving_segment_xyxy=moving_target_xyxy,
        anchor_xy=anchor,
    )
