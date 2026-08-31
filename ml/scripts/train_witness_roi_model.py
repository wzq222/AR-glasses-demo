"""Train the witness ROI evidence model on approved synthetic geometry only."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "ml" / "src"))

from crrc_vision.assets import asset_root  # noqa: E402
from crrc_vision.synthetic_contract import (  # noqa: E402
    FROZEN_FORMAL_TRUTH_SHA256,
    assert_external_output,
    assert_formal_truth_unchanged,
)
from crrc_vision.synthetic_state import relative_angle_deg  # noqa: E402
from crrc_vision.witness_roi_losses import (  # noqa: E402
    keypoint_distribution_loss,
    keypoint_geometry_loss,
    witness_mask_loss,
)
from crrc_vision.witness_roi_model import MobileNetV3SmallWitnessRoi  # noqa: E402


INPUT_SIZE = 320
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _atomic_json(path: Path, document: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class WitnessRoiDataset:
    def __init__(self, rows: list[dict[str, object]], source_root: Path, *, augment: bool) -> None:
        import torch

        self.rows = rows
        self.source_root = source_root
        self.augment = augment
        coordinates = torch.arange(INPUT_SIZE, dtype=torch.float32)
        self.grid_y, self.grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _crop_box(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float]:
        center_x = (bbox[0] + bbox[2]) * 0.5
        center_y = (bbox[1] + bbox[3]) * 0.5
        side = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 1.5
        side = max(side, 16.0)
        left = max(0.0, center_x - side * 0.5)
        top = max(0.0, center_y - side * 0.5)
        right = min(float(width), center_x + side * 0.5)
        bottom = min(float(height), center_y + side * 0.5)
        return left, top, right, bottom

    def _heatmap(self, x: float, y: float, sigma: float = 4.0):
        import torch

        return torch.exp(-((self.grid_x - x) ** 2 + (self.grid_y - y) ** 2) / (2.0 * sigma * sigma))

    def __getitem__(self, index: int):
        import torch
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as vision_f
        from torchvision.transforms.v2 import RandomPerspective

        row = self.rows[index]
        image_path = self.source_root / str(row["image_path"])
        mask_path = self.source_root / str(row["witness_mark_mask_path"])
        with Image.open(image_path) as image_source:
            image = image_source.convert("RGB")
        with Image.open(mask_path) as mask_source:
            mask = mask_source.convert("L")
        crop_box = self._crop_box(
            [float(value) for value in row["fastener_bbox_xyxy"]],  # type: ignore[index]
            image.width,
            image.height,
        )
        image = image.crop(crop_box).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        mask = mask.crop(crop_box).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.NEAREST)
        image_tensor = vision_f.pil_to_tensor(image).float() / 255.0
        witness_mask = (vision_f.pil_to_tensor(mask).float() / 255.0).clamp(0.0, 1.0)

        left, top, right, bottom = crop_box
        scale_x = INPUT_SIZE / (right - left)
        scale_y = INPUT_SIZE / (bottom - top)
        fixed = row["fixed_segment_xyxy"]  # type: ignore[assignment]
        moving = row["moving_segment_xyxy"]  # type: ignore[assignment]
        source_points = (fixed[0], fixed[1], moving[0], moving[1])
        points = [
            (
                min(INPUT_SIZE - 1.0, max(0.0, (float(point[0]) - left) * scale_x)),
                min(INPUT_SIZE - 1.0, max(0.0, (float(point[1]) - top) * scale_y)),
            )
            for point in source_points
        ]
        keypoints = torch.stack([self._heatmap(x, y) for x, y in points])

        if self.augment:
            if random.random() < 0.5:
                image_tensor = vision_f.hflip(image_tensor)
                witness_mask = vision_f.hflip(witness_mask)
                keypoints = vision_f.hflip(keypoints)
            angle = random.uniform(-7.0, 7.0)
            translate = [random.randint(-13, 13), random.randint(-13, 13)]
            scale = random.uniform(0.90, 1.10)
            image_tensor = vision_f.affine(
                image_tensor, angle, translate, scale, [0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            witness_mask = vision_f.affine(
                witness_mask, angle, translate, scale, [0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
            )
            keypoints = vision_f.affine(
                keypoints, angle, translate, scale, [0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
            )
            if random.random() < 0.35:
                start, end = RandomPerspective.get_params(INPUT_SIZE, INPUT_SIZE, 0.10)
                image_tensor = vision_f.perspective(
                    image_tensor, start, end, InterpolationMode.BILINEAR
                )
                witness_mask = vision_f.perspective(
                    witness_mask, start, end, InterpolationMode.NEAREST
                )
                keypoints = vision_f.perspective(
                    keypoints, start, end, InterpolationMode.BILINEAR
                )
            image_tensor = vision_f.adjust_brightness(image_tensor, random.uniform(0.78, 1.22))
            image_tensor = vision_f.adjust_contrast(image_tensor, random.uniform(0.78, 1.22))
            image_tensor = vision_f.adjust_saturation(image_tensor, random.uniform(0.85, 1.15))
            if random.random() < 0.20:
                image_tensor = vision_f.gaussian_blur(image_tensor, [3, 3], [0.1, 1.2])

        image_tensor = vision_f.normalize(image_tensor, IMAGENET_MEAN, IMAGENET_STD)
        quality = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        return image_tensor, witness_mask[0], keypoints, quality, str(row["sample_id"])


def _decode_point(heatmap) -> tuple[float, float]:
    index = int(heatmap.reshape(-1).argmax().item())
    return float(index % INPUT_SIZE), float(index // INPUT_SIZE)


def _evaluate(model, loader, device) -> dict[str, float]:
    import torch

    model.eval()
    intersection = 0.0
    union = 0.0
    point_errors: list[float] = []
    angle_errors: list[float] = []
    with torch.inference_mode():
        for images, masks, heatmaps, _, _ in loader:
            segmentation, predicted_heatmaps, _ = model(images.to(device))
            predicted_masks = torch.sigmoid(segmentation[:, 2]).cpu() >= 0.5
            target_masks = masks >= 0.5
            intersection += float((predicted_masks & target_masks).sum().item())
            union += float((predicted_masks | target_masks).sum().item())
            predicted_heatmaps = predicted_heatmaps.cpu()
            for batch_index in range(images.shape[0]):
                predicted_points = [
                    _decode_point(predicted_heatmaps[batch_index, channel])
                    for channel in range(4)
                ]
                target_points = [
                    _decode_point(heatmaps[batch_index, channel])
                    for channel in range(4)
                ]
                point_errors.extend(
                    math.hypot(predicted[0] - target[0], predicted[1] - target[1])
                    for predicted, target in zip(predicted_points, target_points, strict=True)
                )
                try:
                    predicted_angle = relative_angle_deg(
                        (predicted_points[0], predicted_points[1]),
                        (predicted_points[2], predicted_points[3]),
                    )
                    target_angle = relative_angle_deg(
                        (target_points[0], target_points[1]),
                        (target_points[2], target_points[3]),
                    )
                    angle_errors.append(abs(predicted_angle - target_angle))
                except ValueError:
                    angle_errors.append(90.0)
    point_errors.sort()
    angle_errors.sort()
    return {
        "synthetic_witness_mask_iou": intersection / union if union else 0.0,
        "synthetic_keypoint_error_mean_px": sum(point_errors) / len(point_errors),
        "synthetic_keypoint_error_p95_px": point_errors[max(0, math.ceil(len(point_errors) * 0.95) - 1)],
        "synthetic_angle_error_mean_degrees": sum(angle_errors) / len(angle_errors),
        "synthetic_angle_error_p95_degrees": angle_errors[max(0, math.ceil(len(angle_errors) * 0.95) - 1)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="runs/witness-roi-v1/dataset/manifest.json")
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/witness-roi-v1/train-mobilenetv3-small")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.nn import functional as torch_f
    from torch.utils.data import DataLoader

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    root = asset_root().resolve()
    dataset_path = (root / args.dataset).resolve()
    formal_truth = (root / args.formal_truth).resolve()
    output = assert_external_output((root / args.output).resolve(), REPOSITORY_ROOT)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output}")
    output.mkdir(parents=True, exist_ok=True)
    formal_hash = assert_formal_truth_unchanged(formal_truth)
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    if manifest.get("input_hashes", {}).get("formal_truth_sha256") != formal_hash:
        raise RuntimeError("WITNESS_ROI_DATASET_TRUTH_HASH_MISMATCH")
    if manifest.get("governance") != {
        "synthetic_geometry_only": True,
        "real_state_truth": False,
        "sealed_test_opened": False,
    }:
        raise RuntimeError("WITNESS_ROI_DATASET_GOVERNANCE_INVALID")
    rows = manifest.get("examples")
    if not isinstance(rows, list) or not rows or {row.get("split") for row in rows} != {"train"}:
        raise RuntimeError("WITNESS_ROI_DATASET_SPLIT_INVALID")
    source_root = Path(manifest["source_root"]).resolve()

    _seed_everything(args.seed)
    train_dataset = WitnessRoiDataset(rows, source_root, augment=True)
    evaluation_dataset = WitnessRoiDataset(rows, source_root, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
    )
    evaluation_loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3SmallWitnessRoi(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for images, masks, heatmaps, quality, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            heatmaps = heatmaps.to(device)
            quality = quality.to(device)
            optimizer.zero_grad(set_to_none=True)
            segmentation, predicted_heatmaps, predicted_quality = model(images)
            mark_loss = witness_mask_loss(segmentation[:, 2], masks)
            keypoint_loss = keypoint_distribution_loss(predicted_heatmaps, heatmaps)
            geometry_loss = keypoint_geometry_loss(predicted_heatmaps, heatmaps)
            quality_loss = torch_f.binary_cross_entropy_with_logits(
                predicted_quality, quality
            )
            loss = (
                mark_loss
                + 0.5 * keypoint_loss
                + 2.0 * geometry_loss
                + 0.15 * quality_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record))

    metrics = _evaluate(model, evaluation_loader, device)
    checkpoint_path = output / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "MobileNetV3SmallWitnessRoi",
            "input_size": INPUT_SIZE,
            "seed": args.seed,
            "epochs": args.epochs,
            "synthetic_geometry_only": True,
        },
        checkpoint_path,
    )
    _atomic_json(output / "history.json", history)
    report = {
        "schema_version": "witness-roi-training-v1",
        "architecture": "MobileNetV3SmallWitnessRoi",
        "device": str(device),
        "epochs": args.epochs,
        "sample_count": len(rows),
        "validation_scope": "same 24 synthetic train samples; overfit smoke only",
        "real_state_accuracy_validated": False,
        "supervised_heads": ["witness_mark", "four_keypoint_heatmaps", "synthetic_quality"],
        "unsupervised_heads": ["fixed_part", "moving_part", "joint_boundary"],
        "metrics": metrics,
        "input_hashes": {
            "dataset_sha256": sha256(dataset_path.read_bytes()).hexdigest().upper(),
            "formal_truth_sha256": formal_hash,
        },
        "checkpoint_sha256": sha256(checkpoint_path.read_bytes()).hexdigest().upper(),
    }
    _atomic_json(output / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
