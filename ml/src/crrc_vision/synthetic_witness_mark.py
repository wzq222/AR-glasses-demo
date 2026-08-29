from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class WitnessMarkGeometry:
    fixed_segment_xyxy: Segment
    moving_segment_xyxy: Segment
    anchor_xy: Point
    mask_area: int
    component_count: int


def extract_witness_mark_mask(
    image: np.ndarray,
    fastener_bbox_xyxy: BBox,
    *,
    padding_fraction: float = 0.25,
    baseline_image: np.ndarray | None = None,
) -> np.ndarray:
    """Extract red/yellow ImageGen witness-paint pixels near one fastener.

    This is a selector for pixels already present in the generated image.  It
    never renders or extends a witness mark.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image必须是BGR三通道图像")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(float, fastener_bbox_xyxy)
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("fastener_bbox_xyxy越界")

    def color_mask(value: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(value, cv2.COLOR_BGR2HSV)
        red_low = cv2.inRange(hsv, np.array([0, 95, 55]), np.array([12, 255, 255]))
        red_high = cv2.inRange(hsv, np.array([168, 95, 55]), np.array([179, 255, 255]))
        yellow = cv2.inRange(hsv, np.array([14, 95, 70]), np.array([42, 255, 255]))
        blue = value[:, :, 0].astype(np.int16)
        green = value[:, :, 1].astype(np.int16)
        red = value[:, :, 2].astype(np.int16)
        red_dominance = ((red - green > 45) & (red - blue > 45) & (red > 125)).astype(np.uint8) * 255
        yellow_dominance = (
            (red - blue > 60)
            & (green - blue > 45)
            & (red > 135)
            & (green > 95)
        ).astype(np.uint8) * 255
        red_selected = cv2.bitwise_and(cv2.bitwise_or(red_low, red_high), red_dominance)
        yellow_selected = cv2.bitwise_and(yellow, yellow_dominance)
        return cv2.bitwise_or(red_selected, yellow_selected)

    selected_colors = color_mask(image)
    if baseline_image is not None:
        if baseline_image.shape != image.shape:
            raise ValueError("baseline_image与image尺寸不一致")
        baseline_colors = color_mask(baseline_image)
        baseline_colors = cv2.dilate(baseline_colors, np.ones((5, 5), np.uint8))
        selected_colors = cv2.bitwise_and(selected_colors, cv2.bitwise_not(baseline_colors))

    padding = padding_fraction * max(x2 - x1, y2 - y1)
    left = max(0, int(np.floor(x1 - padding)))
    top = max(0, int(np.floor(y1 - padding)))
    right = min(width, int(np.ceil(x2 + padding)))
    bottom = min(height, int(np.ceil(y2 + padding)))
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[top:bottom, left:right] = 255
    selected = cv2.bitwise_and(selected_colors, roi)
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats((selected > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(selected)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= 12:
            cleaned[labels == label] = 255
    return cleaned


def remove_existing_witness_mark(
    image: np.ndarray,
    fastener_bbox_xyxy: BBox,
    *,
    padding_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove pre-existing red/yellow witness paint before state replacement."""
    mask = extract_witness_mark_mask(
        image,
        fastener_bbox_xyxy,
        padding_fraction=padding_fraction,
    )
    if not np.any(mask):
        return image.copy(), mask
    x1, y1, x2, y2 = map(float, fastener_bbox_xyxy)
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
    )
    selected = np.zeros_like(mask)
    selected_area = 0
    halo = 0.15 * max(width, height)
    for label in range(1, count):
        center_x, center_y = centroids[label]
        if x1 - halo <= center_x <= x2 + halo and y1 - halo <= center_y <= y2 + halo:
            selected[labels == label] = 255
            selected_area += int(stats[label, cv2.CC_STAT_AREA])
    if selected_area == 0:
        return image.copy(), selected
    if selected_area > 0.10 * width * height:
        raise RuntimeError(
            f"unsafe witness-paint removal: selected={selected_area}, target_area={width * height:.1f}"
        )
    radius = max(3, min(4, int(round(min(width, height) * 0.03))))
    kernel_size = radius * 2 + 1
    expanded = cv2.dilate(selected, np.ones((kernel_size, kernel_size), np.uint8))
    cleaned = cv2.inpaint(image, expanded, float(radius), cv2.INPAINT_TELEA)
    return cleaned, expanded


def _fit_segment(points_xy: np.ndarray) -> Segment:
    if len(points_xy) < 2:
        raise ValueError("防松线像素不足")
    center = points_xy.mean(axis=0)
    _, _, vectors = np.linalg.svd(points_xy - center, full_matrices=False)
    direction = vectors[0]
    projections = (points_xy - center) @ direction
    low, high = np.percentile(projections, [2.0, 98.0])
    first = center + direction * low
    second = center + direction * high
    return (float(first[0]), float(first[1])), (float(second[0]), float(second[1]))


def extract_witness_mark_geometry(mask: np.ndarray) -> WitnessMarkGeometry:
    """Fit two line segments to pixels already selected from an ImageGen mark."""
    if mask.ndim != 2:
        raise ValueError("mask必须是单通道图像")
    y_values, x_values = np.nonzero(mask > 0)
    if len(x_values) < 24:
        raise ValueError("ImageGen防松线像素不足")
    points = np.column_stack([x_values, y_values]).astype(np.float64)
    center = points.mean(axis=0)
    _, _, vectors = np.linalg.svd(points - center, full_matrices=False)
    global_direction = vectors[0]
    projections = (points - center) @ global_direction
    split = float(np.median(projections))
    first_points = points[projections <= split]
    second_points = points[projections > split]
    first = _fit_segment(first_points)
    second = _fit_segment(second_points)

    candidates = [
        (np.linalg.norm(np.array(first[i]) - np.array(second[j])), i, j)
        for i in (0, 1)
        for j in (0, 1)
    ]
    _, first_near, second_near = min(candidates, key=lambda item: item[0])
    fixed: Segment = (first[1 - first_near], first[first_near])
    moving: Segment = (second[second_near], second[1 - second_near])
    anchor = (
        (fixed[1][0] + moving[0][0]) / 2.0,
        (fixed[1][1] + moving[0][1]) / 2.0,
    )
    component_count = max(0, cv2.connectedComponents((mask > 0).astype(np.uint8), 8)[0] - 1)
    return WitnessMarkGeometry(
        fixed_segment_xyxy=fixed,
        moving_segment_xyxy=moving,
        anchor_xy=anchor,
        mask_area=int(len(points)),
        component_count=component_count,
    )
