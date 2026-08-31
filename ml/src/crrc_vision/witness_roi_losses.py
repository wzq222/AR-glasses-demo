"""Losses that remain sensitive to sparse witness masks and keypoint heatmaps."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def _validate_pair(logits: Tensor, target: Tensor) -> None:
    if logits.shape != target.shape:
        raise ValueError(
            f"logits and target shape mismatch: {tuple(logits.shape)} != {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("logits and target must contain finite values")
    if bool((target < 0.0).any()) or bool((target > 1.0).any()):
        raise ValueError("target values must be within 0..1")


def witness_mask_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Balanced BCE plus soft Dice for a sparse witness-mark foreground."""
    _validate_pair(logits, target)
    positive = target.sum()
    if float(positive.detach().item()) <= 0.0:
        raise ValueError("witness mask target must contain positive mass")
    negative = target.numel() - positive
    positive_weight = torch.clamp(negative / positive, min=1.0, max=100.0)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=positive_weight
    )
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def keypoint_distribution_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Cross entropy between spatial heatmap distributions.

    Unlike pixelwise MSE, every keypoint channel contributes equal loss even
    though its positive Gaussian occupies far below one percent of the ROI.
    """
    _validate_pair(logits, target)
    if logits.ndim != 4:
        raise ValueError("keypoint logits and target must be NCHW")
    flat_target = target.flatten(start_dim=2)
    mass = flat_target.sum(dim=2, keepdim=True)
    if bool((mass <= 0.0).any()):
        raise ValueError("each keypoint target must contain positive mass")
    distribution = flat_target / mass
    log_probability = F.log_softmax(logits.flatten(start_dim=2), dim=2)
    return -(distribution * log_probability).sum(dim=2).mean()


def spatial_soft_argmax(logits: Tensor) -> Tensor:
    """Return differentiable normalized ``(x, y)`` coordinates per channel."""
    if logits.ndim != 4:
        raise ValueError("keypoint logits must be NCHW")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("keypoint logits must contain finite values")
    _, _, height, width = logits.shape
    probability = F.softmax(logits.flatten(start_dim=2), dim=2).reshape_as(logits)
    x_axis = torch.linspace(0.0, 1.0, width, dtype=logits.dtype, device=logits.device)
    y_axis = torch.linspace(0.0, 1.0, height, dtype=logits.dtype, device=logits.device)
    x = (probability.sum(dim=2) * x_axis).sum(dim=2)
    y = (probability.sum(dim=3) * y_axis).sum(dim=2)
    return torch.stack((x, y), dim=2)


def _target_expected_points(target: Tensor) -> Tensor:
    mass = target.sum(dim=(-2, -1), keepdim=True)
    if bool((mass <= 0.0).any()):
        raise ValueError("each keypoint target must contain positive mass")
    return spatial_soft_argmax(torch.log(target / mass + 1.0e-12))


def _relative_angle_radians(points: Tensor) -> Tensor:
    fixed = points[:, 1] - points[:, 0]
    moving = points[:, 3] - points[:, 2]
    cross = torch.abs(fixed[:, 0] * moving[:, 1] - fixed[:, 1] * moving[:, 0])
    dot = torch.abs((fixed * moving).sum(dim=1))
    return torch.atan2(cross, dot + 1.0e-6)


def keypoint_geometry_loss(logits: Tensor, target: Tensor) -> Tensor:
    """Couple endpoint localization with the relative witness-line angle."""
    _validate_pair(logits, target)
    if logits.ndim != 4 or logits.shape[1] != 4:
        raise ValueError("keypoint geometry expects N x 4 x H x W")
    predicted_points = spatial_soft_argmax(logits)
    target_points = _target_expected_points(target)
    coordinate_loss = F.smooth_l1_loss(predicted_points, target_points)
    predicted_angle = _relative_angle_radians(predicted_points)
    target_angle = _relative_angle_radians(target_points)
    angle_loss = F.smooth_l1_loss(
        predicted_angle / (torch.pi * 0.5),
        target_angle / (torch.pi * 0.5),
    )
    return coordinate_loss + angle_loss
