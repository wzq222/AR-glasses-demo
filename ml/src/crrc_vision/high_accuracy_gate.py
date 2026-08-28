"""Deterministic target-level accuracy metrics for the high-accuracy gate."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class AccuracyReport:
    precision: float
    recall: float
    complete_scene_rate: float
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    complete_scenes: int
    total_scenes: int
    passed: bool


@dataclass(frozen=True)
class SeedSummary:
    seeds: tuple[int, ...]
    recall_mean: float
    recall_std: float
    recall_range: float
    worst_seed: int
    worst_recall: float


def _rows(value: object, name: str) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


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
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_at_threshold(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    threshold: float,
    iou_threshold: float = 0.50,
    minimum_precision: float = 0.90,
    minimum_recall: float = 0.95,
    minimum_complete_scene_rate: float = 0.90,
    enforce_sealed_minimum: bool = False,
) -> AccuracyReport:
    """Evaluate COCO xywh predictions using deterministic one-to-one matching."""

    if not math.isfinite(threshold):
        raise ValueError("INVALID_SCORE_THRESHOLD")
    images = _rows(truth.get("images"), "truth images")
    annotations = _rows(truth.get("annotations"), "truth annotations")
    image_ids = [row.get("id", row.get("image_id")) for row in images]
    if len(set(image_ids)) != len(image_ids) or None in image_ids:
        raise ValueError("INVALID_TRUTH_IMAGE_IDS")
    image_id_set = set(image_ids)
    if enforce_sealed_minimum:
        if len(images) < 30:
            raise ValueError(f"SEALED_TEST_SCENE_COUNT_TOO_LOW:{len(images)}")
        if len(annotations) < 200:
            raise ValueError(f"SEALED_TEST_BOX_COUNT_TOO_LOW:{len(annotations)}")

    truth_by_image: dict[object, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in annotations:
        image_id = annotation.get("image_id")
        if image_id not in image_id_set:
            raise ValueError(f"TRUTH_ANNOTATION_UNKNOWN_IMAGE:{image_id}")
        truth_by_image[image_id].append(_bbox(annotation))

    eligible: list[tuple[int, Mapping[str, object]]] = []
    for index, prediction in enumerate(predictions):
        image_id = prediction.get("image_id")
        if image_id not in image_id_set:
            raise ValueError(f"PREDICTION_UNKNOWN_IMAGE:{image_id}")
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
    true_positives = 0
    false_positives = 0
    for _, prediction in eligible:
        image_id = prediction["image_id"]
        prediction_box = _bbox(prediction)
        available = [
            (index, _iou(prediction_box, truth_box))
            for index, truth_box in enumerate(truth_by_image[image_id])
            if index not in matched[image_id]
        ]
        best = max(available, key=lambda item: (item[1], -item[0]), default=None)
        if best is not None and best[1] + 1e-12 >= iou_threshold:
            matched[image_id].add(best[0])
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = len(annotations) - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = true_positives / len(annotations) if annotations else 1.0
    complete_scenes = sum(
        len(matched[image_id]) == len(truth_by_image[image_id])
        for image_id in image_ids
    )
    complete_scene_rate = complete_scenes / len(images) if images else 0.0
    passed = (
        precision >= minimum_precision
        and recall >= minimum_recall
        and complete_scene_rate >= minimum_complete_scene_rate
    )
    return AccuracyReport(
        precision=precision,
        recall=recall,
        complete_scene_rate=complete_scene_rate,
        threshold=float(threshold),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        complete_scenes=complete_scenes,
        total_scenes=len(images),
        passed=passed,
    )


def select_threshold(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    minimum_precision: float = 0.90,
    iou_threshold: float = 0.50,
) -> float:
    """Select maximum-recall validation threshold under a precision constraint."""

    thresholds = sorted(
        {float(row.get("score", float("nan"))) for row in predictions}, reverse=True
    )
    candidates: list[AccuracyReport] = []
    for threshold in thresholds:
        report = evaluate_at_threshold(
            predictions,
            truth,
            threshold=threshold,
            iou_threshold=iou_threshold,
            minimum_precision=minimum_precision,
        )
        if report.precision >= minimum_precision:
            candidates.append(report)
    if not candidates:
        raise ValueError("NO_THRESHOLD_MEETS_PRECISION")
    selected = max(
        candidates,
        key=lambda report: (report.recall, report.precision, report.threshold),
    )
    return selected.threshold


def summarize_seed_reports(
    seeds: Sequence[int], reports: Sequence[AccuracyReport]
) -> SeedSummary:
    if len(seeds) != 3 or len(reports) != 3 or len(set(seeds)) != 3:
        raise ValueError("EXACTLY_THREE_UNIQUE_SEEDS_REQUIRED")
    recalls = [report.recall for report in reports]
    worst_index = min(range(3), key=lambda index: (recalls[index], seeds[index]))
    return SeedSummary(
        seeds=tuple(seeds),
        recall_mean=statistics.fmean(recalls),
        recall_std=statistics.pstdev(recalls),
        recall_range=max(recalls) - min(recalls),
        worst_seed=seeds[worst_index],
        worst_recall=recalls[worst_index],
    )
