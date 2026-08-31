"""Lightweight multi-head ROI model for explainable witness-mark geometry."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


SEGMENTATION_CHANNELS = (
    "fixed_part",
    "moving_part",
    "witness_mark",
    "joint_boundary",
)
KEYPOINT_CHANNELS = (
    "fixed_outer_endpoint",
    "fixed_joint_intersection",
    "moving_joint_intersection",
    "moving_outer_endpoint",
)
QUALITY_CHANNELS = (
    "mark_integrity",
    "occlusion",
    "blur",
    "topology_confidence",
)


class _ConvNormActivation(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
        )


class MobileNetV3SmallWitnessRoi(nn.Module):
    """MobileNetV3-Small + LR-ASPP-like shared decoder.

    The model predicts evidence, not a loose/aligned class.  Geometry and the
    fail-closed state contract remain outside the network.
    """

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.low_projection = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.high_projection = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(576, 128, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            _ConvNormActivation(160, 96),
            _ConvNormActivation(96, 64),
        )
        self.detail_projection = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.keypoint_decoder = _ConvNormActivation(80, 48)
        self.segmentation_head = nn.Conv2d(64, len(SEGMENTATION_CHANNELS), kernel_size=1)
        self.keypoint_head = nn.Conv2d(48, len(KEYPOINT_CHANNELS), kernel_size=1)
        self.quality_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(576, 96),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(96, len(QUALITY_CHANNELS)),
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError("witness ROI input must be NCHW RGB")
            if not images.is_floating_point():
                raise ValueError("witness ROI input must be floating point")
            if not bool(torch.isfinite(images).all()):
                raise ValueError("witness ROI input must contain finite values")

        input_size = images.shape[-2:]
        features = images
        detail = None
        low = None
        for index, layer in enumerate(self.backbone):
            features = layer(features)
            if index == 1:
                detail = features
            if index == 3:
                low = features
        if detail is None or low is None:  # pragma: no cover - fixed torchvision topology guard
            raise RuntimeError("MobileNetV3 low-level feature is unavailable")

        high = self.high_projection(features) * self.context_gate(features)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat((self.low_projection(low), high), dim=1))
        segmentation = F.interpolate(
            self.segmentation_head(fused),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        keypoint_features = F.interpolate(
            fused, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        keypoint_features = self.keypoint_decoder(
            torch.cat((keypoint_features, self.detail_projection(detail)), dim=1)
        )
        keypoints = F.interpolate(
            self.keypoint_head(keypoint_features),
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        quality = self.quality_head(features)
        return segmentation, keypoints, quality


def validate_witness_roi_outputs(
    segmentation: Tensor,
    keypoints: Tensor,
    quality: Tensor,
    *,
    batch_size: int,
    height: int = 320,
    width: int = 320,
) -> None:
    expected = {
        "segmentation": (batch_size, len(SEGMENTATION_CHANNELS), height, width),
        "keypoints": (batch_size, len(KEYPOINT_CHANNELS), height, width),
        "quality": (batch_size, len(QUALITY_CHANNELS)),
    }
    values = {
        "segmentation": segmentation,
        "keypoints": keypoints,
        "quality": quality,
    }
    for name, tensor in values.items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(
                f"{name} output shape mismatch: expected {expected[name]}, got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} output must contain finite values")
