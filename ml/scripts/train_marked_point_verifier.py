"""Train and evaluate a MobileNetV3 verifier on every E1 validation proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from crrc_vision.assets import asset_root
from crrc_vision.marked_point_verifier import select_pipeline_threshold


FORMAL_TRUTH_SHA256 = (
    "B659FC8160BD7C49491BA4C560E1AF047CA837E54EE93E79826FEBAABCB0F001"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class ProposalDataset:
    def __init__(self, rows: list[dict[str, object]], transform, class_to_index) -> None:
        self.rows = rows
        self.transform = transform
        self.class_to_index = class_to_index

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        with Image.open(str(row["crop_path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        target = torch.tensor(self.class_to_index[str(row["label"])], dtype=torch.long)
        return tensor, target, index


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _evaluate(
    model,
    loader,
    rows,
    device,
    *,
    marked_index: int,
    image_count: int,
    minimum_recall: float,
):
    import torch

    model.eval()
    probabilities: dict[int, float] = {}
    losses: list[float] = []
    criterion = torch.nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, targets, indices in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            losses.append(float(criterion(logits, targets).item()))
            scores = torch.softmax(logits, dim=1)[:, marked_index].cpu().tolist()
            probabilities.update(
                (int(index), float(score))
                for index, score in zip(indices.tolist(), scores, strict=True)
            )
    scored = [
        {**row, "score": probabilities[index]}
        for index, row in enumerate(rows)
    ]
    pipeline = select_pipeline_threshold(
        scored,
        image_count=image_count,
        minimum_truth_recall=minimum_recall,
    )
    labels = np.asarray([row["label"] == "marked_point" for row in scored])
    predictions = np.asarray([row["score"] >= pipeline.threshold for row in scored])
    true_positive = int(np.logical_and(labels, predictions).sum())
    false_positive = int(np.logical_and(~labels, predictions).sum())
    false_negative = int(np.logical_and(labels, ~predictions).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    candidate_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    return scored, {
        "loss": sum(losses) / len(losses),
        "candidate_precision": precision,
        "candidate_recall": candidate_recall,
        "pipeline": asdict(pipeline),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="runs/marked-point-verifier-e3/dataset-v2/manifest.json"
    )
    parser.add_argument("--formal-truth", default="annotations/fastener-v2/instances.json")
    parser.add_argument("--output", default="runs/marked-point-verifier-e3/mobilenetv3-small")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--minimum-truth-recall", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    from torchvision.transforms import v2

    root = asset_root().resolve()
    dataset_path = (root / args.dataset).resolve()
    formal_truth = (root / args.formal_truth).resolve()
    output = (root / args.output).resolve()
    if _sha256(formal_truth) != FORMAL_TRUTH_SHA256:
        raise RuntimeError("FORMAL_TRUTH_HASH_MISMATCH")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OUTPUT_NOT_EMPTY:{output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    if manifest["input_hashes"]["formal_truth_sha256"] != FORMAL_TRUTH_SHA256:
        raise RuntimeError("VERIFIER_DATASET_TRUTH_HASH_MISMATCH")
    if manifest.get("sealed_test_opened") is not False:
        raise RuntimeError("SEALED_TEST_STATE_INVALID")
    rows = manifest["examples"]
    classes = list(manifest["labels"])
    if len(classes) < 2 or len(set(classes)) != len(classes) or "marked_point" not in classes:
        raise RuntimeError("VERIFIER_CLASSES_INVALID")
    class_to_index = {label: index for index, label in enumerate(classes)}
    marked_index = class_to_index["marked_point"]
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    train_scenes = {row["scene_group"] for row in train_rows}
    val_scenes = {row["scene_group"] for row in val_rows}
    if train_scenes & val_scenes:
        raise RuntimeError("VERIFIER_SCENE_LEAKAGE")

    _seed_everything(args.seed)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop((224, 224), scale=(0.78, 1.0), ratio=(0.85, 1.15)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(7),
            v2.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.15, hue=0.025),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=weights.transforms().mean, std=weights.transforms().std),
        ]
    )
    val_transform = weights.transforms()
    train_dataset = ProposalDataset(train_rows, train_transform, class_to_index)
    val_dataset = ProposalDataset(val_rows, val_transform, class_to_index)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mobilenet_v3_small(weights=weights)
    model.classifier[-1] = torch.nn.Linear(
        model.classifier[-1].in_features, len(classes)
    )
    model.to(device)
    class_counts = torch.tensor(
        [sum(row["label"] == label for row in train_rows) for label in classes],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(class_counts == 0):
        raise RuntimeError("VERIFIER_TRAIN_CLASS_EMPTY")
    class_weights = torch.sqrt(class_counts.sum() / class_counts)
    class_weights /= class_weights.mean()
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.02
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )

    history: list[dict[str, object]] = []
    best_key = (math.inf, math.inf)
    best_epoch = 0
    best_predictions: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()
        scored, evaluation = _evaluate(
            model,
            val_loader,
            val_rows,
            device,
            marked_index=marked_index,
            image_count=len(val_scenes),
            minimum_recall=args.minimum_truth_recall,
        )
        record = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / len(train_losses),
            "learning_rate": optimizer.param_groups[0]["lr"],
            **evaluation,
        }
        history.append(record)
        key = (
            float(evaluation["pipeline"]["candidates_per_image"]),
            float(evaluation["loss"]),
        )
        if key < best_key:
            best_key = key
            best_epoch = epoch
            best_predictions = scored
            torch.save(
                {
                    "schema_version": "marked-point-verifier-checkpoint-v1",
                    "architecture": "mobilenet_v3_small",
                    "classes": classes,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dataset_sha256": _sha256(dataset_path),
                },
                output / "best.pt",
            )
        _atomic_json(
            output / "training-progress.json",
            {
                "schema_version": "marked-point-verifier-progress-v1",
                "status": "running",
                "classes": classes,
                "seed": args.seed,
                "completed_epochs": epoch,
                "planned_epochs": args.epochs,
                "best_epoch": best_epoch,
                "history": history,
            },
        )
        print(json.dumps(record, ensure_ascii=False), flush=True)

    result = {
        "schema_version": "marked-point-verifier-training-v1",
        "architecture": "mobilenet_v3_small",
        "classes": classes,
        "class_weights": class_weights.detach().cpu().tolist(),
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "dataset_sha256": _sha256(dataset_path),
        "formal_truth_sha256": _sha256(formal_truth),
        "sealed_test_opened": False,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "train_scenes": len(train_scenes),
        "val_scenes": len(val_scenes),
        "minimum_truth_recall": args.minimum_truth_recall,
        "best_epoch": best_epoch,
        "best": history[best_epoch - 1],
        "history": history,
    }
    _atomic_json(output / "results.json", result)
    _atomic_json(
        output / "training-progress.json",
        {
            "schema_version": "marked-point-verifier-progress-v1",
            "status": "complete",
            "classes": classes,
            "seed": args.seed,
            "completed_epochs": args.epochs,
            "planned_epochs": args.epochs,
            "best_epoch": best_epoch,
            "history": history,
        },
    )
    _atomic_json(output / "best-val-predictions.json", best_predictions)
    print(json.dumps({"best_epoch": best_epoch, "best": result["best"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
