"""High-recall red/yellow anti-loosening mark proposals."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorMarkProposal:
    color: str
    mark_xyxy: tuple[int, int, int, int]
    roi_xyxy: tuple[int, int, int, int]
    line_xyxy: tuple[float, float, float, float]
    area: int
    elongation: float
    score: float


def _color_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hue, saturation, value = cv2.split(hsv)
    lightness, channel_a, channel_b = cv2.split(lab)
    a16 = channel_a.astype(np.int16)
    b16 = channel_b.astype(np.int16)

    red_hsv = (
        ((hue <= 15) | (hue >= 165))
        & (saturation >= 38)
        & (value >= 22)
    )
    red_lab = (
        (lightness >= 12)
        & (a16 >= 143)
        & ((a16 - b16) >= 4)
        & ((a16 - 128) >= 10)
    )
    yellow_hsv = (
        (hue >= 14)
        & (hue <= 43)
        & (saturation >= 32)
        & (value >= 28)
    )
    yellow_lab = (
        (lightness >= 18)
        & (b16 >= 145)
        & ((b16 - a16) >= 16)
        & ((b16 - 128) >= 12)
    )
    return {
        "red": np.where(red_hsv | red_lab, 255, 0).astype(np.uint8),
        "yellow": np.where(yellow_hsv | yellow_lab, 255, 0).astype(np.uint8),
    }


def _principal_line(points: np.ndarray) -> tuple[float, float, float, float]:
    coordinates = points.astype(np.float64)
    center = coordinates.mean(axis=0)
    if len(coordinates) == 1:
        x, y = center
        return float(x), float(y), float(x), float(y)
    centered = coordinates - center
    covariance = centered.T @ centered
    _, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, -1]
    projections = centered @ direction
    start = center + direction * projections.min()
    end = center + direction * projections.max()
    return tuple(float(value) for value in (*start, *end))  # type: ignore[return-value]


def _roi(
    mark_xyxy: tuple[int, int, int, int], *, width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = mark_xyxy
    long_axis = max(x2 - x1, y2 - y1)
    side = min(320.0, max(96.0, 6.0 * long_axis))
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    return (
        max(0, int(np.floor(center_x - side / 2.0))),
        max(0, int(np.floor(center_y - side / 2.0))),
        min(width, int(np.ceil(center_x + side / 2.0))),
        min(height, int(np.ceil(center_y + side / 2.0))),
    )


def _components(
    image: np.ndarray,
    color_masks: dict[str, np.ndarray],
    *,
    minimum_area: int,
) -> list[ColorMarkProposal]:
    raw_union = np.logical_or(
        color_masks["red"] > 0, color_masks["yellow"] > 0
    ).astype(np.uint8)
    # Paint strokes often cross the red/yellow hue boundary or are interrupted by
    # glare. A 3 px closing joins those pixels at mark level before ROI expansion,
    # while preserving independently visible nearby strokes.
    connected_mask = cv2.morphologyEx(
        raw_union, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
    )
    height, width = connected_mask.shape
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        connected_mask, connectivity=8
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    output: list[ColorMarkProposal] = []
    for label in range(1, count):
        x, y, box_width, box_height = (
            int(value) for value in stats[label, :4]
        )
        local = (
            labels[y : y + box_height, x : x + box_width] == label
        ) & (raw_union[y : y + box_height, x : x + box_width] > 0)
        local_ys, local_xs = np.nonzero(local)
        ys, xs = local_ys + y, local_xs + x
        area = len(xs)
        if area < minimum_area or not area:
            continue
        x, y = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        box_width, box_height = x2 - x, y2 - y
        points = np.column_stack((xs, ys))
        rectangle = cv2.minAreaRect(points.astype(np.float32))
        first, second = rectangle[1]
        long_axis = max(float(first), float(second), 1.0)
        short_axis = max(min(float(first), float(second)), 1.0)
        mark_xyxy = (x, y, x2, y2)
        saturation = float(hsv[ys, xs, 1].mean())
        hue = float(np.median(hsv[ys, xs, 0]))
        color = "yellow" if 14.0 <= hue <= 43.0 else "red"
        score = round(min(1.0, 0.35 + 0.45 * saturation / 255.0 + 0.2 * min(1.0, area / 80.0)), 6)
        output.append(
            ColorMarkProposal(
                color=color,
                mark_xyxy=mark_xyxy,
                roi_xyxy=_roi(mark_xyxy, width=width, height=height),
                line_xyxy=_principal_line(points),
                area=area,
                elongation=round(long_axis / short_axis, 6),
                score=score,
            )
        )
    return output


def find_color_mark_proposals(
    image: np.ndarray, *, minimum_area: int = 8
) -> list[ColorMarkProposal]:
    """Return mark-level components; expanded ROI overlap never deletes a mark."""

    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("BGR_IMAGE_REQUIRED")
    if image.dtype != np.uint8:
        raise ValueError("UINT8_IMAGE_REQUIRED")
    if minimum_area <= 0:
        raise ValueError("MINIMUM_AREA_MUST_BE_POSITIVE")

    proposals = _components(
        image, _color_masks(image), minimum_area=minimum_area
    )
    proposals.sort(
        key=lambda row: (
            row.mark_xyxy,
            row.color,
            -row.score,
        )
    )
    return proposals
