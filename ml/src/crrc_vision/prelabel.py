"""Color-mark based fastener candidate generation for auditable prelabels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VisionPoint:
    x: int
    y: int


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height

    def as_coco(self) -> list[int]:
        return [self.x, self.y, self.width, self.height]

    def iou(self, other: "BoundingBox") -> float:
        left = max(self.x, other.x)
        top = max(self.y, other.y)
        right = min(self.x + self.width, other.x + other.width)
        bottom = min(self.y + self.height, other.y + other.height)
        intersection = max(0, right - left) * max(0, bottom - top)
        union = self.width * self.height + other.width * other.height - intersection
        return intersection / union if union else 0.0


@dataclass(frozen=True)
class LineSegment:
    start: VisionPoint
    end: VisionPoint
    confidence: float


@dataclass(frozen=True)
class MarkedFastenerCandidate:
    bbox: BoundingBox
    line: LineSegment
    mark_color: str
    confidence: float
    mark_area: float


def read_bgr_image(path: Path) -> np.ndarray:
    """Read an image without relying on OpenCV's Unicode path handling."""
    encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def _square_box(center_x: float, center_y: float, side: int, width: int, height: int) -> BoundingBox:
    side = min(side, width, height)
    left = min(max(0, round(center_x - side / 2)), width - side)
    top = min(max(0, round(center_y - side / 2)), height - side)
    return BoundingBox(int(left), int(top), int(side), int(side))


def _line_from_contour(contour: np.ndarray) -> LineSegment:
    points = contour.reshape(-1, 2).astype(np.float32)
    direction_x, direction_y, _, _ = (
        float(value) for value in cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
    )
    direction = np.array([direction_x, direction_y], dtype=np.float32)
    centroid = points.mean(axis=0)
    projections = (points - centroid) @ direction
    start = centroid + projections.min() * direction
    end = centroid + projections.max() * direction

    _, _, width, height = cv2.boundingRect(contour)
    elongation = max(width, height) / max(1.0, min(width, height))
    confidence = min(0.99, 0.50 + 0.08 * elongation)
    return LineSegment(
        VisionPoint(int(round(start[0])), int(round(start[1]))),
        VisionPoint(int(round(end[0])), int(round(end[1]))),
        confidence,
    )


def _mask_candidates(
    mask: np.ndarray,
    color: str,
    image_width: int,
    image_height: int,
    min_mark_area: float,
) -> list[MarkedFastenerCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[MarkedFastenerCandidate] = []
    side = max(160, round(min(image_width, image_height) * 0.12))
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, width, height = cv2.boundingRect(contour)
        _, rectangle_size, _ = cv2.minAreaRect(contour)
        short_axis = min(rectangle_size)
        long_axis = max(rectangle_size)
        elongation = long_axis / max(1.0, short_axis)
        if (
            area < min_mark_area
            or area > 4000
            or long_axis > 180
            or elongation < 1.4
            or (width < 4 and height < 4)
        ):
            continue
        line = _line_from_contour(contour)
        bbox = _square_box(x + width / 2, y + height / 2, side, image_width, image_height)
        area_support = min(1.0, area / max(min_mark_area * 4.0, 1.0))
        confidence = round(0.65 * line.confidence + 0.35 * area_support, 6)
        candidates.append(MarkedFastenerCandidate(bbox, line, color, confidence, area))
    return candidates


def _deduplicate(candidates: list[MarkedFastenerCandidate], threshold: float) -> list[MarkedFastenerCandidate]:
    accepted: list[MarkedFastenerCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.bbox.y, item.bbox.x)):
        if any(candidate.bbox.iou(existing.bbox) > threshold for existing in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.bbox.y, item.bbox.x))


def find_marked_fasteners(
    image_bgr: np.ndarray, *, min_mark_area: float = 20.0, merge_iou: float = 0.35
) -> list[MarkedFastenerCandidate]:
    """Find red/yellow anti-loosening marks and return full-image candidates."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a three-channel BGR image")

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 120, 150), (8, 255, 255))
    red |= cv2.inRange(hsv, (172, 120, 150), (179, 255, 255))
    yellow = cv2.inRange(hsv, (20, 110, 130), (38, 255, 255))

    opening = np.ones((3, 3), dtype=np.uint8)
    closing = np.ones((5, 5), dtype=np.uint8)
    height, width = image_bgr.shape[:2]
    candidates: list[MarkedFastenerCandidate] = []
    for color, mask in (("red", red), ("yellow", yellow)):
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, closing)
        candidates.extend(_mask_candidates(cleaned, color, width, height, min_mark_area))

    return _deduplicate(candidates, merge_iou)
