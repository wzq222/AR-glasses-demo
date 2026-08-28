"""Deterministic validation-only detection error matching and taxonomy."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ErrorEvidence:
    area_ratio: float
    border_distance_ratio: float
    brightness: float
    focus_score: float
    local_contrast: float
    nearby_density: int
    annotation_dispute: bool = False
    occluded: bool = False
    reflection: bool = False


@dataclass(frozen=True)
class DetectionError:
    kind: str
    image_id: object
    truth_id: object | None
    prediction_index: int | None
    bbox: tuple[float, float, float, float]
    score: float | None


_PRIORITY = (
    "annotation_dispute",
    "border_truncation",
    "tiny",
    "dark",
    "blur",
    "occlusion",
    "reflection",
    "dense_pipes",
    "lookalike",
)


def classify_error(evidence: ErrorEvidence) -> tuple[str, tuple[str, ...]]:
    """Assign exactly one primary bucket using the fixed project priority."""

    active = {
        "annotation_dispute": evidence.annotation_dispute,
        "border_truncation": evidence.border_distance_ratio <= 0.01,
        "tiny": evidence.area_ratio <= 0.0025,
        "dark": evidence.brightness < 65.0,
        "blur": evidence.focus_score < 35.0,
        "occlusion": evidence.occluded,
        "reflection": evidence.reflection,
        "dense_pipes": evidence.nearby_density >= 4,
        "lookalike": True,
    }
    primary = next(name for name in _PRIORITY if active[name])
    secondary = tuple(
        name for name in _PRIORITY if name != primary and name != "lookalike" and active[name]
    )
    return primary, secondary


def _bbox(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    value = row.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("INVALID_BBOX")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or box[2] <= 0 or box[3] <= 0:
        raise ValueError("INVALID_BBOX")
    return box


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = width * height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union else 0.0


def detection_errors(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    threshold: float,
    iou_threshold: float = 0.50,
) -> list[DetectionError]:
    """Return deterministic unmatched predictions and annotations."""

    images = truth.get("images")
    annotations = truth.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("INVALID_TRUTH")
    image_ids = {row["id"] for row in images}
    truth_by_image: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for annotation in annotations:
        if annotation.get("image_id") not in image_ids:
            raise ValueError("TRUTH_ANNOTATION_UNKNOWN_IMAGE")
        _bbox(annotation)
        truth_by_image[annotation["image_id"]].append(annotation)

    eligible: list[tuple[int, Mapping[str, object]]] = []
    for index, prediction in enumerate(predictions):
        if prediction.get("image_id") not in image_ids:
            raise ValueError("PREDICTION_UNKNOWN_IMAGE")
        score = float(prediction.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError("INVALID_PREDICTION_SCORE")
        _bbox(prediction)
        if score >= threshold:
            eligible.append((index, prediction))
    eligible.sort(
        key=lambda pair: (
            -float(pair[1]["score"]),
            int(pair[1]["image_id"]),
            _bbox(pair[1]),
            pair[0],
        )
    )

    matched: dict[object, set[int]] = defaultdict(set)
    errors: list[DetectionError] = []
    for prediction_index, prediction in eligible:
        image_id = prediction["image_id"]
        candidates = [
            (index, _iou(_bbox(prediction), _bbox(annotation)))
            for index, annotation in enumerate(truth_by_image[image_id])
            if index not in matched[image_id]
        ]
        best = max(candidates, key=lambda item: (item[1], -item[0]), default=None)
        if best is not None and best[1] + 1e-12 >= iou_threshold:
            matched[image_id].add(best[0])
            continue
        errors.append(
            DetectionError(
                kind="false_positive",
                image_id=image_id,
                truth_id=None,
                prediction_index=prediction_index,
                bbox=_bbox(prediction),
                score=float(prediction["score"]),
            )
        )
    for image_id in sorted(image_ids, key=int):
        for index, annotation in enumerate(truth_by_image[image_id]):
            if index in matched[image_id]:
                continue
            errors.append(
                DetectionError(
                    kind="false_negative",
                    image_id=image_id,
                    truth_id=annotation.get("id"),
                    prediction_index=None,
                    bbox=_bbox(annotation),
                    score=None,
                )
            )
    return errors


def validate_diagnostic_truth(
    path: Path,
    truth: Mapping[str, object],
    *,
    truth_sha256: str,
    forbidden_truth_hashes: Set[str],
) -> None:
    partition = str(truth.get("info", {}).get("partition", "")).lower()
    normalized_hash = truth_sha256.upper()
    forbidden = {value.upper() for value in forbidden_truth_hashes}
    if (
        partition == "sealed_test"
        or any("sealed" in part.lower() for part in path.parts)
        or normalized_hash in forbidden
    ):
        raise ValueError("SEALED_TRUTH_FORBIDDEN")
    if partition != "val":
        raise ValueError(f"VALIDATION_PARTITION_REQUIRED:{partition}")


def threshold_from_selection(
    selection: Mapping[str, object], *, prediction_sha256: str
) -> float:
    if (
        selection.get("schema_version") != "high-accuracy-selection-v1"
        or selection.get("mode") != "val"
        or selection.get("sealed_test_opened") is not False
    ):
        raise ValueError("INVALID_VALIDATION_SELECTION")
    if str(selection.get("prediction_sha256", "")).upper() != prediction_sha256.upper():
        raise ValueError("SELECTION_PREDICTION_HASH_MISMATCH")
    threshold = float(selection.get("threshold", float("nan")))
    if not math.isfinite(threshold):
        raise ValueError("INVALID_SELECTION_THRESHOLD")
    return threshold
