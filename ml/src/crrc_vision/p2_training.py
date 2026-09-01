"""Guardrails and reproducible arguments for high-accuracy P2 training."""

from __future__ import annotations

import csv
import math
import pickle
from pathlib import Path


P2_SEEDS = (20260828, 20260829, 20260830)
P2_MODEL_YAMLS = {
    "n": "yolov8n-p2.yaml",
    "s": "yolov8s-p2.yaml",
    "m": "yolov8m-p2.yaml",
}


def p2_model_yaml(variant: str) -> str:
    try:
        return P2_MODEL_YAMLS[variant]
    except KeyError as exc:
        raise ValueError(f"INVALID_P2_VARIANT:{variant}") from exc


def validate_pretraining_mode(*, variant: str, transfer_pretrained: bool) -> None:
    p2_model_yaml(variant)
    if variant == "m" and not transfer_pretrained:
        raise ValueError("M_CHALLENGER_REQUIRES_TRANSFER")


def validate_training_checkpoint(
    *,
    checkpoint_version: str,
    transfer_pretrained: bool,
    actual_sha256: str,
    expected_sha256: str | None,
) -> None:
    if checkpoint_version == "8.2.40":
        return
    if not transfer_pretrained:
        raise ValueError(f"CHECKPOINT_VERSION_MISMATCH:{checkpoint_version}")
    if expected_sha256 is None or actual_sha256.upper() != expected_sha256.upper():
        raise ValueError("TRANSFER_CHECKPOINT_HASH_MISMATCH")


def checkpoint_model(checkpoint: dict[str, object]) -> object:
    """Return an export model or the EMA model kept in a resumable checkpoint."""

    model = checkpoint.get("model")
    if model is None:
        model = checkpoint.get("ema")
    if model is None:
        raise ValueError("CHECKPOINT_MODEL_MISSING")
    return model


def is_recoverable_finalization_failure(
    *,
    error: BaseException,
    results_csv: Path,
    best: Path,
    last: Path,
    expected_epochs: int,
    patience: int | None = None,
) -> bool:
    """Recognize the pinned runtime's post-training torch.load incompatibility.

    Recovery is deliberately narrow: the exact weights-only failure must happen
    after either every requested epoch was recorded or a legitimate early-stop
    window elapsed, and both local checkpoints exist.
    """

    if (
        not isinstance(error, pickle.UnpicklingError)
        or "Weights only load failed" not in str(error)
    ):
        return False
    if expected_epochs <= 0 or not all(path.is_file() for path in (results_csv, best, last)):
        return False
    try:
        progress: list[tuple[int, float | None]] = []
        with results_csv.open("r", encoding="utf-8-sig", newline="") as stream:
            for raw_row in csv.DictReader(stream):
                row = {str(key).strip(): value for key, value in raw_row.items()}
                epoch = int(float(row["epoch"]))
                fitness = None
                if "metrics/mAP50(B)" in row and "metrics/mAP50-95(B)" in row:
                    fitness = 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(
                        row["metrics/mAP50-95(B)"]
                    )
                progress.append((epoch, fitness))
    except (KeyError, OSError, TypeError, ValueError):
        return False
    if not progress:
        return False
    completed = max(epoch for epoch, _ in progress)
    if completed >= expected_epochs:
        return True
    measured = [(epoch, fitness) for epoch, fitness in progress if fitness is not None]
    if patience is None or patience <= 0 or not measured:
        return False
    best_epoch, _ = max(measured, key=lambda item: (item[1], item[0]))
    return completed - best_epoch >= patience


def build_resume_kwargs(*, batch_size: int) -> dict[str, object]:
    if batch_size <= 0:
        raise ValueError("INVALID_RESUME_BATCH")
    return {"resume": True, "batch": batch_size, "workers": 0}


def validate_training_inputs(
    *,
    asset_root: Path,
    train_coco: Path,
    val_coco: Path,
    pretrained: Path,
    output_root: Path,
    ultralytics_version: str,
    allow_existing_output: bool = False,
) -> None:
    root = asset_root.resolve()
    for path in (train_coco, val_coco, pretrained):
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ValueError(f"TRAINING_PATH_OUTSIDE_ASSETS:{path}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
    output = output_root.resolve()
    if root not in output.parents:
        raise ValueError(f"TRAINING_PATH_OUTSIDE_ASSETS:{output_root}")
    if any("sealed" in part.lower() for path in (train_coco, val_coco) for part in path.parts):
        raise ValueError("SEALED_TEST_PATH_FORBIDDEN")
    if allow_existing_output:
        if not output.is_dir() or not any(output.iterdir()):
            raise FileNotFoundError(f"RESUME_OUTPUT_MISSING:{output}")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError(f"TRAINING_OUTPUT_NOT_EMPTY:{output}")
    if ultralytics_version != "8.2.40":
        raise ValueError(
            f"ULTRALYTICS_VERSION_MISMATCH:{ultralytics_version}:expected=8.2.40"
        )


def validate_synthetic_ablation_mode(
    train_document: dict,
    *,
    maximum_synthetic_fraction: float | None,
    batch_size: int,
) -> dict[str, int]:
    images = train_document.get("images", [])
    synthetic = [item for item in images if item.get("synthetic") is True]
    real = [item for item in images if item.get("synthetic") is not True]
    if synthetic and maximum_synthetic_fraction is None:
        raise ValueError("SYNTHETIC_TRAIN_REQUIRES_CAP")
    if not synthetic:
        return {"real_images": len(real), "synthetic_images": 0, "synthetic_per_batch": 0}
    if maximum_synthetic_fraction is None or maximum_synthetic_fraction > 0.30:
        raise ValueError("SYNTHETIC_BATCH_CAP_TOO_HIGH")
    synthetic_per_batch = math.floor(batch_size * maximum_synthetic_fraction)
    if synthetic_per_batch < 1:
        raise ValueError("BATCH_TOO_SMALL_FOR_SYNTHETIC_CAP")
    if any(int(item.get("id", -1)) < 1_000_000 for item in synthetic):
        raise ValueError("SYNTHETIC_IMAGE_ID_NOT_RESERVED")
    return {
        "real_images": len(real),
        "synthetic_images": len(synthetic),
        "synthetic_per_batch": synthetic_per_batch,
    }


def build_train_kwargs(
    *,
    seed: int,
    dataset_yaml: Path,
    run_root: Path,
    epochs: int,
    batch_size: int,
    fine_tune: bool = False,
) -> dict[str, object]:
    if seed not in P2_SEEDS:
        raise ValueError(f"INVALID_P2_SEED:{seed}")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("INVALID_TRAINING_DURATION")
    kwargs: dict[str, object] = {
        "data": str(dataset_yaml.resolve()),
        "epochs": epochs,
        "patience": 15,
        "batch": batch_size,
        "imgsz": 640,
        "device": 0,
        "workers": 0,
        "project": str(run_root.resolve()),
        "name": "train",
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.1,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "seed": seed,
        "deterministic": True,
        "single_cls": True,
        "amp": True,
        "cache": False,
        "hsv_h": 0.01,
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 5.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "close_mosaic": 0,
        "max_det": 300,
        "verbose": False,
        "plots": True,
    }
    if fine_tune:
        kwargs.update(
            {
                "patience": 8,
                "lr0": 0.00005,
                "lrf": 0.2,
                "warmup_epochs": 0.0,
                "warmup_bias_lr": 0.0,
            }
        )
    return kwargs
