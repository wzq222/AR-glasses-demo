"""Pure coordinate and NMS helpers for full-image plus sliced detection."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np


def merge_target_coco_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return one-class COCO ground truth for physical-target evaluation."""

    merged = copy.deepcopy(document)
    annotations = merged.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("annotations must be a list")
    for annotation in annotations:
        annotation["category_id"] = 1
    merged["categories"] = [{"id": 1, "name": "fastener_target"}]
    return merged


def merge_target_coco_predictions(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return copied predictions mapped to the physical-target category."""

    merged = copy.deepcopy(predictions)
    for prediction in merged:
        prediction["category_id"] = 1
    return merged


def merge_target_classes(detections: np.ndarray) -> np.ndarray:
    """Map detector subclasses to one physical target without mutating input."""

    merged = detections.astype(np.float32, copy=True)
    if merged.ndim != 2 or merged.shape[1] != 6:
        raise ValueError("detections must have shape [N, 6]")
    merged[:, 0] = 0.0
    return merged


def runtime_path_text(path: Path) -> str:
    """Return an absolute runtime path without resolving Windows junctions."""

    return str(path.absolute())


def _iou(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    left = np.maximum(reference[0], candidates[:, 0])
    top = np.maximum(reference[1], candidates[:, 1])
    right = np.minimum(reference[2], candidates[:, 2])
    bottom = np.minimum(reference[3], candidates[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    reference_area = max(0.0, reference[2] - reference[0]) * max(
        0.0, reference[3] - reference[1]
    )
    candidate_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0, candidates[:, 3] - candidates[:, 1]
    )
    union = reference_area + candidate_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def fuse_image_detections(
    full: np.ndarray,
    sliced: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Fuse ``[class, score, x1, y1, x2, y2]`` rows with class-wise NMS."""

    if not 0.0 < iou_threshold < 1.0:
        raise ValueError("iou_threshold must be between zero and one")
    arrays = [array for array in (full, sliced) if array.size]
    if not arrays:
        return np.empty((0, 6), dtype=np.float32)
    detections = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    if detections.ndim != 2 or detections.shape[1] != 6:
        raise ValueError("detections must have shape [N, 6]")

    kept: list[np.ndarray] = []
    for class_id in sorted(set(detections[:, 0].astype(int).tolist())):
        class_rows = detections[detections[:, 0].astype(int) == class_id]
        order = np.argsort(-class_rows[:, 1], kind="stable")
        while order.size:
            current = order[0]
            kept.append(class_rows[current])
            if order.size == 1:
                break
            remaining = order[1:]
            overlaps = _iou(class_rows[current, 2:6], class_rows[remaining, 2:6])
            order = remaining[overlaps <= iou_threshold]
    return np.asarray(sorted(kept, key=lambda row: -float(row[1])), dtype=np.float32)


def nms_image_detections(
    detections: np.ndarray,
    *,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Deduplicate one image's rows, including boxes from overlapping tiles."""

    empty = np.empty((0, 6), dtype=np.float32)
    return fuse_image_detections(empty, detections, iou_threshold=iou_threshold)


def select_detection_mode(
    full: np.ndarray,
    sliced: np.ndarray,
    *,
    mode: str,
    iou_threshold: float,
) -> np.ndarray:
    """Select full, sliced, or fused detections for a controlled comparison."""

    if mode == "full":
        return full
    if mode == "sliced":
        return sliced
    if mode == "fused":
        return fuse_image_detections(full, sliced, iou_threshold=iou_threshold)
    raise ValueError("mode must be full, sliced, or fused")


def to_coco_predictions(
    *,
    image_id: int,
    detections: np.ndarray,
    image_width: int,
    image_height: int,
) -> list[dict[str, object]]:
    """Convert detector rows to clipped COCO result records."""

    predictions: list[dict[str, object]] = []
    for row in detections:
        class_id, score, x1, y1, x2, y2 = (float(value) for value in row)
        x1 = min(max(x1, 0.0), float(image_width))
        y1 = min(max(y1, 0.0), float(image_height))
        x2 = min(max(x2, 0.0), float(image_width))
        y2 = min(max(y2, 0.0), float(image_height))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            continue
        predictions.append(
            {
                "image_id": int(image_id),
                "category_id": int(class_id) + 1,
                "bbox": [x1, y1, width, height],
                "score": score,
            }
        )
    return predictions
