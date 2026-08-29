"""Deterministic train-only mixing for the synthetic witness-mark ablation."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path
from typing import Iterable


SYNTHETIC_IMAGE_ID_OFFSET = 1_000_000


def merge_training_documents(
    real_document: dict,
    synthetic_document: dict,
    *,
    synthetic_image_root: Path,
    maximum_synthetic_fraction: float,
) -> dict:
    if real_document.get("info", {}).get("partition") != "train":
        raise ValueError("REAL_PARTITION_MUST_BE_TRAIN")
    real_images = copy.deepcopy(real_document.get("images", []))
    synthetic_images = copy.deepcopy(synthetic_document.get("images", []))
    if not real_images or not synthetic_images:
        raise ValueError("ABLATION_REQUIRES_REAL_AND_SYNTHETIC_IMAGES")
    fraction = len(synthetic_images) / (len(real_images) + len(synthetic_images))
    if fraction > maximum_synthetic_fraction + 1e-12:
        raise ValueError(f"SYNTHETIC_FRACTION_EXCEEDED:{fraction:.6f}")
    for image in real_images:
        if image.get("synthetic") is True:
            raise ValueError("REAL_DOCUMENT_CONTAINS_SYNTHETIC_IMAGE")
        image["synthetic"] = False
    synthetic_id_map: dict[int, int] = {}
    for offset, image in enumerate(sorted(synthetic_images, key=lambda item: int(item["id"]))):
        old_id = int(image["id"])
        new_id = SYNTHETIC_IMAGE_ID_OFFSET + offset
        synthetic_id_map[old_id] = new_id
        image["id"] = new_id
        image["file_name"] = str((synthetic_image_root / str(image["file_name"])).resolve())
        image["scene_group"] = f"synthetic::{image.get('source_scene_id', old_id)}::{old_id}"
        image["synthetic"] = True
        image["eligible_split"] = "train"
    annotations = copy.deepcopy(real_document.get("annotations", []))
    next_annotation_id = max((int(item["id"]) for item in annotations), default=0) + 1
    for annotation in sorted(synthetic_document.get("annotations", []), key=lambda item: int(item["id"])):
        copied = copy.deepcopy(annotation)
        copied["id"] = next_annotation_id
        copied["image_id"] = synthetic_id_map[int(annotation["image_id"])]
        copied["category_id"] = 1
        annotations.append(copied)
        next_annotation_id += 1
    return {
        "info": {
            "partition": "train",
            "schema_version": "synthetic-ablation-coco-v1",
            "real_images": len(real_images),
            "synthetic_images": len(synthetic_images),
            "synthetic_fraction": fraction,
            "maximum_synthetic_fraction": maximum_synthetic_fraction,
        },
        "images": [*real_images, *synthetic_images],
        "annotations": annotations,
        "categories": copy.deepcopy(real_document.get("categories", [])),
    }


def _deterministic_take(values: list[int], count: int, rng: random.Random) -> list[int]:
    if not values or count <= 0:
        return []
    selected: list[int] = []
    while len(selected) < count:
        shuffled = values.copy()
        rng.shuffle(shuffled)
        selected.extend(shuffled[: count - len(selected)])
    return selected


def build_capped_batches(
    real_indices: Iterable[int],
    synthetic_indices: Iterable[int],
    *,
    batch_size: int,
    maximum_synthetic_fraction: float,
    seed: int,
    epoch: int,
) -> list[list[int]]:
    real = list(real_indices)
    synthetic = list(synthetic_indices)
    if batch_size <= 0 or not 0 < maximum_synthetic_fraction < 1:
        raise ValueError("INVALID_SYNTHETIC_BATCH_POLICY")
    synthetic_per_batch = math.floor(batch_size * maximum_synthetic_fraction)
    if synthetic_per_batch < 1 or not real or not synthetic:
        raise ValueError("BATCH_TOO_SMALL_FOR_SYNTHETIC_CAP")
    batch_count = math.ceil(len(real) / batch_size)
    real_per_batch = batch_size - synthetic_per_batch
    rng = random.Random(seed + 1_000_003 * epoch)
    selected_real = _deterministic_take(real, batch_count * real_per_batch, rng)
    selected_synthetic = _deterministic_take(
        synthetic, batch_count * synthetic_per_batch, rng
    )
    batches = []
    for batch_index in range(batch_count):
        batch = selected_real[
            batch_index * real_per_batch : (batch_index + 1) * real_per_batch
        ] + selected_synthetic[
            batch_index * synthetic_per_batch : (batch_index + 1) * synthetic_per_batch
        ]
        rng.shuffle(batch)
        batches.append(batch)
    return batches
