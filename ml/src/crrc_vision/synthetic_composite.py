from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .synthetic_transform import apply_homography_points, transform_bbox


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Segments = tuple[tuple[Point, Point], tuple[Point, Point]]


@dataclass(frozen=True)
class Placement:
    x: float
    y: float
    scale: float
    rotation_deg: float


@dataclass(frozen=True)
class CompositeResult:
    image: np.ndarray
    bbox_xyxy: BBox
    segments: Segments
    anchor_xy: Point
    homography: np.ndarray
    seam_score: float


def _placement_matrix(width: int, height: int, placement: Placement) -> np.ndarray:
    if placement.scale <= 0:
        raise ValueError("scale必须为正数")
    center = (width / 2.0, height / 2.0)
    affine = cv2.getRotationMatrix2D(center, placement.rotation_deg, placement.scale)
    desired_center = (
        placement.x + width * placement.scale / 2.0,
        placement.y + height * placement.scale / 2.0,
    )
    affine[0, 2] += desired_center[0] - center[0]
    affine[1, 2] += desired_center[1] - center[1]
    return np.vstack([affine, np.array([0.0, 0.0, 1.0])])


def _inside(bbox: BBox, width: int, height: int) -> bool:
    x1, y1, x2, y2 = bbox
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _transform_segments(segments: Segments, matrix: np.ndarray) -> Segments:
    flat = np.array([segments[0][0], segments[0][1], segments[1][0], segments[1][1]])
    points = apply_homography_points(flat, matrix)
    return (
        ((float(points[0, 0]), float(points[0, 1])), (float(points[1, 0]), float(points[1, 1]))),
        ((float(points[2, 0]), float(points[2, 1])), (float(points[3, 0]), float(points[3, 1]))),
    )


def composite_sample(
    background: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: BBox,
    segments: Segments,
    placement: Placement,
    *,
    existing_boxes: tuple[BBox, ...] = (),
    max_overlap_iou: float = 0.05,
    minimum_short_side: float = 12.0,
    blend_mode: str = "alpha",
    preserve_mask: np.ndarray | None = None,
) -> CompositeResult:
    if background.ndim != 3 or patch.ndim != 3 or mask.ndim != 2:
        raise ValueError("background/patch/mask维度无效")
    if patch.shape[:2] != mask.shape[:2]:
        raise ValueError("patch与mask尺寸不一致")
    if preserve_mask is not None and preserve_mask.shape != mask.shape:
        raise ValueError("preserve_mask与patch尺寸不一致")

    background_height, background_width = background.shape[:2]
    patch_height, patch_width = patch.shape[:2]
    matrix = _placement_matrix(patch_width, patch_height, placement)
    transformed_bbox = transform_bbox(bbox_xyxy, matrix)
    if not _inside(transformed_bbox, background_width, background_height):
        raise ValueError("合成目标超出全图边界")
    short_side = min(
        transformed_bbox[2] - transformed_bbox[0],
        transformed_bbox[3] - transformed_bbox[1],
    )
    if short_side < minimum_short_side:
        raise ValueError(f"合成目标短边小于{minimum_short_side:g}像素")
    if any(_iou(transformed_bbox, existing) > max_overlap_iou for existing in existing_boxes):
        raise ValueError("合成目标与已有检查点重叠")

    warped_patch = cv2.warpPerspective(
        patch,
        matrix,
        (background_width, background_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_mask = cv2.warpPerspective(
        mask,
        matrix,
        (background_width, background_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if blend_mode == "mark_only":
        if preserve_mask is None:
            raise ValueError("mark_only模式需要preserve_mask")
        warped_preserve = cv2.warpPerspective(
            preserve_mask,
            matrix,
            (background_width, background_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        detail_alpha = cv2.GaussianBlur(warped_preserve, (3, 3), 0).astype(np.float32) / 255.0
        detail_alpha = np.clip(detail_alpha, 0.0, 1.0)[..., None]
        blended = np.clip(
            warped_patch.astype(np.float32) * detail_alpha
            + background.astype(np.float32) * (1.0 - detail_alpha),
            0,
            255,
        ).astype(np.uint8)
        seam_mask = warped_preserve
    elif blend_mode == "seamless":
        mask_points = cv2.findNonZero((warped_mask > 8).astype(np.uint8))
        if mask_points is None:
            raise ValueError("合成掩膜为空")
        mask_x, mask_y, mask_width, mask_height = cv2.boundingRect(mask_points)
        center = (mask_x + mask_width // 2, mask_y + mask_height // 2)
        blended = cv2.seamlessClone(
            warped_patch,
            background,
            warped_mask,
            center,
            cv2.MIXED_CLONE,
        )
        seam_mask = warped_mask
    elif blend_mode == "alpha":
        alpha = cv2.GaussianBlur(warped_mask, (5, 5), 0).astype(np.float32) / 255.0
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]
        blended = np.clip(
            warped_patch.astype(np.float32) * alpha
            + background.astype(np.float32) * (1.0 - alpha),
            0,
            255,
        ).astype(np.uint8)
        seam_mask = warped_mask
    else:
        raise ValueError(f"未知融合模式: {blend_mode}")

    if preserve_mask is not None and blend_mode != "mark_only":
        warped_preserve = cv2.warpPerspective(
            preserve_mask,
            matrix,
            (background_width, background_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        detail_alpha = cv2.GaussianBlur(warped_preserve, (3, 3), 0).astype(np.float32) / 255.0
        detail_alpha = np.clip(detail_alpha, 0.0, 1.0)[..., None]
        blended = np.clip(
            warped_patch.astype(np.float32) * detail_alpha
            + blended.astype(np.float32) * (1.0 - detail_alpha),
            0,
            255,
        ).astype(np.uint8)

    edge = cv2.morphologyEx((seam_mask > 8).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    difference = np.mean(np.abs(warped_patch.astype(np.float32) - background.astype(np.float32)), axis=2)
    seam_score = float(difference[edge > 0].mean()) if np.any(edge > 0) else 0.0
    transformed_segments = _transform_segments(segments, matrix)
    anchor = transformed_segments[0][1]
    return CompositeResult(
        image=blended,
        bbox_xyxy=transformed_bbox,
        segments=transformed_segments,
        anchor_xy=anchor,
        homography=matrix,
        seam_score=seam_score,
    )
