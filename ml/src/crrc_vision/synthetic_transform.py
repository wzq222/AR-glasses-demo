from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TransformLimits:
    rotation_deg: float = 8.0
    scale_min: float = 0.85
    scale_max: float = 1.15
    perspective_fraction: float = 0.04
    brightness_min: float = 0.8
    brightness_max: float = 1.2
    contrast_min: float = 0.85
    contrast_max: float = 1.15
    noise_sigma_max: float = 4.0


@dataclass(frozen=True)
class SampledTransform:
    matrix: np.ndarray
    rotation_deg: float
    scale: float


def sample_transform(
    width: int,
    height: int,
    seed: int,
    limits: TransformLimits,
) -> SampledTransform:
    if width <= 0 or height <= 0:
        raise ValueError("图像尺寸必须为正数")
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-limits.rotation_deg, limits.rotation_deg))
    scale = float(rng.uniform(limits.scale_min, limits.scale_max))
    affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    affine3 = np.vstack([affine, np.array([0.0, 0.0, 1.0])])

    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )
    max_dx = width * limits.perspective_fraction
    max_dy = height * limits.perspective_fraction
    jitter = np.column_stack(
        [rng.uniform(-max_dx, max_dx, 4), rng.uniform(-max_dy, max_dy, 4)]
    ).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(corners, corners + jitter)
    return SampledTransform(matrix=perspective @ affine3, rotation_deg=angle, scale=scale)


def apply_homography_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("points必须是N×2")
    homogeneous = np.column_stack([values, np.ones(len(values))])
    projected = homogeneous @ np.asarray(matrix, dtype=np.float64).T
    if np.any(np.isclose(projected[:, 2], 0.0)):
        raise ValueError("单应变换产生无穷远点")
    return (projected[:, :2] / projected[:, 2:3]).astype(np.float32)


def transform_bbox(
    bbox_xyxy: tuple[float, float, float, float], matrix: np.ndarray
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    warped = apply_homography_points(corners, matrix)
    return (
        float(np.min(warped[:, 0])),
        float(np.min(warped[:, 1])),
        float(np.max(warped[:, 0])),
        float(np.max(warped[:, 1])),
    )


def warp_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = output_size
    warped_image = cv2.warpPerspective(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )
    warped_mask = cv2.warpPerspective(
        mask, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    return warped_image, warped_mask


def apply_photometric(
    image: np.ndarray,
    seed: int,
    limits: TransformLimits = TransformLimits(),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    brightness = float(rng.uniform(limits.brightness_min, limits.brightness_max))
    contrast = float(rng.uniform(limits.contrast_min, limits.contrast_max))
    adjusted = image.astype(np.float32) * (brightness * contrast)
    adjusted += 127.5 * (1.0 - contrast)
    sigma = float(rng.uniform(0.0, limits.noise_sigma_max))
    if sigma > 0:
        adjusted += rng.normal(0.0, sigma, adjusted.shape)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    if bool(rng.integers(0, 2)):
        adjusted = cv2.GaussianBlur(adjusted, (3, 3), 0)
    return adjusted
