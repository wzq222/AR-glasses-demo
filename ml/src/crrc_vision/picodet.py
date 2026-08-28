"""Guarded PicoDet dataset preparation and external training command contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PINNED_PADDLEDETECTION_REVISION = "v2.9.0"
PINNED_PADDLEDETECTION_COMMIT = "b25522a0f4bde8c80603f3ba5e3472059972e3b5"
EXPECTED_CATEGORIES = ((1, "fastener"), (2, "pipe_joint"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


@dataclass(frozen=True)
class PicodetReadiness:
    train_groups: int
    val_groups: int
    train_images: int
    val_images: int
    annotations: int
    reasons: tuple[str, ...]

    @property
    def can_train(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, object]:
        return {
            "can_train": self.can_train,
            "train_groups": self.train_groups,
            "val_groups": self.val_groups,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "annotations": self.annotations,
            "reasons": list(self.reasons),
        }


def _rows(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def validate_silver_dataset(
    document: dict[str, object],
    source_root: Path,
    *,
    minimum_train_groups: int = 64,
    minimum_val_groups: int = 16,
) -> PicodetReadiness:
    """Validate the scene-isolated, real-image silver dataset before training."""

    images = _rows(document.get("images"), "images")
    annotations = _rows(document.get("annotations"), "annotations")
    categories = _rows(document.get("categories"), "categories")
    reasons: set[str] = set()

    actual_categories = tuple(
        sorted((int(row.get("id", -1)), str(row.get("name", ""))) for row in categories)
    )
    if actual_categories != EXPECTED_CATEGORIES:
        reasons.add("INVALID_CATEGORY")

    split_groups: dict[str, set[str]] = {"train": set(), "val": set()}
    split_images: dict[str, int] = {"train": 0, "val": 0}
    image_ids: set[int] = set()
    root = source_root.resolve()
    for image in images:
        try:
            image_id = int(image["id"])
            split = str(image["split"])
            scene_group = str(image["scene_group"])
            relative_path = Path(str(image["relative_path"]))
        except (KeyError, TypeError, ValueError):
            reasons.add("INVALID_IMAGE_RECORD")
            continue
        if image_id in image_ids:
            reasons.add("DUPLICATE_IMAGE")
        image_ids.add(image_id)
        if split not in split_groups or not scene_group:
            reasons.add("INVALID_SPLIT")
            continue
        if image.get("image_review_status") != "complete":
            reasons.add("INCOMPLETE_IMAGE")
        if split == "val" and image.get("synthetic") is True:
            reasons.add("SYNTHETIC_VAL_IMAGE")
        split_groups[split].add(scene_group)
        split_images[split] += 1
        image_path = (root / relative_path).resolve()
        if root not in image_path.parents or not image_path.is_file():
            reasons.add("IMAGE_MISSING")
            continue
        expected_hash = str(image.get("sha256") or "").upper()
        if not expected_hash or _sha256(image_path) != expected_hash:
            reasons.add("IMAGE_HASH_MISMATCH")

    if split_groups["train"] & split_groups["val"]:
        reasons.add("SCENE_SPLIT_LEAKAGE")
    if len(split_groups["train"]) < minimum_train_groups:
        reasons.add(f"TRAIN_GROUPS_BELOW_{minimum_train_groups}")
    if len(split_groups["val"]) < minimum_val_groups:
        reasons.add(f"VAL_GROUPS_BELOW_{minimum_val_groups}")

    valid_category_ids = {category_id for category_id, _ in EXPECTED_CATEGORIES}
    if not annotations:
        reasons.add("NO_ANNOTATIONS")
    for annotation in annotations:
        try:
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            bbox = annotation["bbox"]
        except (KeyError, TypeError, ValueError):
            reasons.add("INVALID_ANNOTATION")
            continue
        if image_id not in image_ids:
            reasons.add("UNKNOWN_ANNOTATION_IMAGE")
        if category_id not in valid_category_ids:
            reasons.add("INVALID_CATEGORY")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            reasons.add("INVALID_BBOX")
        if annotation.get("review_status") != "accept":
            reasons.add("UNACCEPTED_ANNOTATION")

    return PicodetReadiness(
        train_groups=len(split_groups["train"]),
        val_groups=len(split_groups["val"]),
        train_images=split_images["train"],
        val_images=split_images["val"],
        annotations=len(annotations),
        reasons=tuple(sorted(reasons)),
    )


def prepare_picodet_dataset(
    *,
    document_path: Path,
    source_root: Path,
    runtime_source_root: Path | None = None,
    run_root: Path,
    formal_truth_path: Path,
    expected_truth_sha256: str,
) -> dict[str, object]:
    """Write deterministic train/val COCO plus a provenance manifest outside Git."""

    truth_before = _sha256(formal_truth_path)
    if truth_before != expected_truth_sha256.upper():
        raise ValueError("FORMAL_TRUTH_HASH_MISMATCH")
    document = json.loads(document_path.read_text(encoding="utf-8"))
    readiness = validate_silver_dataset(document, source_root)
    if not readiness.can_train:
        raise ValueError("PICODET_TRAINING_REFUSED:" + ",".join(readiness.reasons))

    images = _rows(document["images"], "images")
    annotations = _rows(document["annotations"], "annotations")
    dataset_root = run_root / "dataset"
    annotations_root = dataset_root / "annotations"
    annotations_root.mkdir(parents=True, exist_ok=True)
    source = source_root.resolve()
    runtime_source = runtime_source_root.absolute() if runtime_source_root else source
    for split in ("train", "val"):
        split_images = [dict(image) for image in images if image["split"] == split]
        split_ids = {int(image["id"]) for image in split_images}
        for image in split_images:
            image["file_name"] = (
                runtime_source / str(image["relative_path"])
            ).absolute().as_posix()
        split_document = {
            "info": document.get("info", {}),
            "images": sorted(split_images, key=lambda image: int(image["id"])),
            "annotations": sorted(
                (dict(annotation) for annotation in annotations if int(annotation["image_id"]) in split_ids),
                key=lambda annotation: int(annotation["id"]),
            ),
            "categories": document["categories"],
        }
        (annotations_root / f"{split}.json").write_bytes(_json_bytes(split_document))

    truth_after = _sha256(formal_truth_path)
    if truth_after != truth_before:
        raise RuntimeError("FORMAL_TRUTH_CHANGED")
    manifest: dict[str, object] = {
        "schema_version": "picodet-training-manifest-v1",
        "status": "ready",
        **readiness.to_dict(),
        "silver_path": str(document_path.resolve()),
        "silver_sha256": _sha256(document_path),
        "formal_truth_path": str(formal_truth_path.resolve()),
        "formal_truth_sha256": truth_after,
        "source_root": str(source),
        "paddledetection_revision": PINNED_PADDLEDETECTION_REVISION,
    }
    (run_root / "training-manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def build_train_command(
    *,
    python: Path,
    paddledetection_root: Path,
    variant: str,
    run_root: Path,
    epochs: int,
    batch_size: int,
) -> list[str]:
    """Build the pinned, single-GPU PicoDet command shared by real and test runners."""

    if variant not in {"s", "m"}:
        raise ValueError("variant must be s or m")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    train_script = paddledetection_root.resolve() / "tools" / "train.py"
    config = (
        paddledetection_root.resolve()
        / "configs"
        / "picodet"
        / f"picodet_{variant}_416_coco_lcnet.yml"
    )
    if not train_script.is_file() or not config.is_file():
        raise FileNotFoundError("pinned PaddleDetection train script or PicoDet config is missing")
    output = run_root.resolve() / "output"
    dataset = run_root.resolve() / "dataset"
    return [
        str(python),
        str(train_script),
        "-c",
        str(config),
        "--eval",
        "--amp",
        "-o",
        f"epoch={epochs}",
        f"TrainReader.batch_size={batch_size}",
        "LearningRate.base_lr=0.04",
        f"TrainDataset.dataset_dir={dataset.as_posix()}",
        "TrainDataset.image_dir=",
        "TrainDataset.anno_path=annotations/train.json",
        f"EvalDataset.dataset_dir={dataset.as_posix()}",
        "EvalDataset.image_dir=",
        "EvalDataset.anno_path=annotations/val.json",
        "num_classes=2",
        f"save_dir={output.as_posix()}",
        "seed=20260828",
    ]


def write_picodet_config(
    *,
    paddledetection_root: Path,
    runtime_paddledetection_root: Path | None = None,
    variant: str,
    run_root: Path,
    runtime_run_root: Path | None = None,
    epochs: int,
    batch_size: int,
) -> Path:
    """Write a small run-local config that inherits an untouched official model."""

    if variant not in {"s", "m"}:
        raise ValueError("variant must be s or m")
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    base = (
        paddledetection_root.resolve()
        / "configs"
        / "picodet"
        / f"picodet_{variant}_416_coco_lcnet.yml"
    )
    if not base.is_file():
        raise FileNotFoundError(base)
    run = run_root.resolve()
    runtime_run = runtime_run_root.absolute() if runtime_run_root else run
    runtime_checkout = (
        runtime_paddledetection_root.absolute()
        if runtime_paddledetection_root
        else paddledetection_root.resolve()
    )
    runtime_base = (
        runtime_checkout
        / "configs"
        / "picodet"
        / f"picodet_{variant}_416_coco_lcnet.yml"
    )
    dataset = (runtime_run / "dataset").as_posix()
    output = (runtime_run / "output").as_posix()
    config_root = run / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    config = config_root / base.name
    warmup_steps = min(100, max(10, epochs * 2))
    content = f"""_BASE_: ['{runtime_base.as_posix()}']

use_gpu: true
epoch: {epochs}
snapshot_epoch: 10
worker_num: 0
num_classes: 2
save_dir: '{output}'

TrainDataset:
  name: COCODataSet
  image_dir: ''
  anno_path: annotations/train.json
  dataset_dir: '{dataset}'
  data_fields: ['image', 'gt_bbox', 'gt_class', 'is_crowd']

EvalDataset:
  name: COCODataSet
  image_dir: ''
  anno_path: annotations/val.json
  dataset_dir: '{dataset}'
  allow_empty: true

TestDataset:
  name: ImageFolder
  anno_path: annotations/val.json
  dataset_dir: '{dataset}'

TrainReader:
  batch_size: {batch_size}

LearningRate:
  base_lr: 0.04
  schedulers:
  - !CosineDecay
    max_epochs: {epochs}
  - !LinearWarmup
    start_factor: 0.1
    steps: {warmup_steps}
"""
    config.write_text(content, encoding="utf-8")
    return config
