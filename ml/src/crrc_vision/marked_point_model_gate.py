from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from crrc_vision.high_accuracy_gate import evaluate_at_threshold


@dataclass(frozen=True)
class ProposalReport:
    threshold: float
    coverage_recall: float
    candidate_relevance: float
    candidates_per_image: float
    complete_scenes: int
    images: int
    covered_truth: int
    uncovered_truth: int
    eligible_candidates: int
    irrelevant_candidates: int
    strict_iou_recall: float
    strict_iou_precision: float


def _bbox(row: Mapping[str, object]) -> tuple[float, float, float, float]:
    value = row.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("INVALID_BBOX")
    box = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in box) or box[2] <= 0 or box[3] <= 0:
        raise ValueError("INVALID_BBOX")
    return box


def is_proposal_match(
    proposal: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> bool:
    return _iou(proposal, target) >= 0.30


def _iou(
    proposal: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
) -> float:
    px, py, pw, ph = proposal
    tx, ty, tw, th = target
    intersection = max(0.0, min(px + pw, tx + tw) - max(px, tx)) * max(
        0.0, min(py + ph, ty + th) - max(py, ty)
    )
    proposal_area = pw * ph
    target_area = tw * th
    union = proposal_area + target_area - intersection
    return intersection / union if union else 0.0


def _coverage_metrics(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    threshold: float,
) -> tuple[int, int, int, int, int, int]:
    images = truth.get("images")
    annotations = truth.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("INVALID_TRUTH")
    image_ids = [row.get("id") for row in images if isinstance(row, Mapping)]
    if len(image_ids) != len(images) or None in image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("INVALID_TRUTH_IMAGE_IDS")
    image_id_set = set(image_ids)
    truth_by_image: dict[object, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            raise ValueError("INVALID_TRUTH")
        image_id = annotation.get("image_id")
        if image_id not in image_id_set:
            raise ValueError(f"TRUTH_ANNOTATION_UNKNOWN_IMAGE:{image_id}")
        truth_by_image[image_id].append(_bbox(annotation))

    predictions_by_image: dict[object, list[tuple[float, float, float, float]]] = defaultdict(list)
    for prediction in predictions:
        image_id = prediction.get("image_id")
        if image_id not in image_id_set:
            raise ValueError(f"PREDICTION_UNKNOWN_IMAGE:{image_id}")
        score = float(prediction.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError("INVALID_PREDICTION_SCORE")
        box = _bbox(prediction)
        if score >= threshold:
            predictions_by_image[image_id].append(box)

    covered_truth = 0
    complete_scenes = 0
    relevant_candidates = 0
    for image_id in image_ids:
        target_boxes = truth_by_image[image_id]
        proposal_boxes = predictions_by_image[image_id]
        target_to_proposal: dict[int, int] = {}

        def assign(proposal_index: int, visited_targets: set[int]) -> bool:
            matches = sorted(
                (
                    (target_index, _iou(proposal_boxes[proposal_index], target))
                    for target_index, target in enumerate(target_boxes)
                    if is_proposal_match(proposal_boxes[proposal_index], target)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            for target_index, _ in matches:
                if target_index in visited_targets:
                    continue
                visited_targets.add(target_index)
                previous = target_to_proposal.get(target_index)
                if previous is None or assign(previous, visited_targets):
                    target_to_proposal[target_index] = proposal_index
                    return True
            return False

        for proposal_index in range(len(proposal_boxes)):
            assign(proposal_index, set())
        hits = len(target_to_proposal)
        covered_truth += hits
        complete_scenes += hits == len(target_boxes)
        relevant_candidates += sum(
            any(is_proposal_match(proposal, target) for target in target_boxes)
            for proposal in proposal_boxes
        )
    eligible_candidates = sum(len(rows) for rows in predictions_by_image.values())
    return (
        covered_truth,
        len(annotations) - covered_truth,
        complete_scenes,
        eligible_candidates,
        relevant_candidates,
        len(images),
    )


def select_proposal_threshold(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    minimum_recall: float = 0.99,
    iou_threshold: float = 0.50,
) -> ProposalReport:
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("INVALID_MINIMUM_RECALL")
    thresholds = sorted(
        {float(row.get("score", float("nan"))) for row in predictions},
        reverse=True,
    )
    if any(not math.isfinite(value) for value in thresholds):
        raise ValueError("INVALID_PREDICTION_SCORE")
    selected_threshold = None
    selected_metrics = None
    low = 0
    high = len(thresholds) - 1
    while low <= high:
        middle = (low + high) // 2
        threshold = thresholds[middle]
        metrics = _coverage_metrics(predictions, truth, threshold=threshold)
        covered_truth, uncovered_truth = metrics[0], metrics[1]
        coverage_recall = covered_truth / (covered_truth + uncovered_truth) if covered_truth + uncovered_truth else 1.0
        if coverage_recall >= minimum_recall:
            selected_threshold = threshold
            selected_metrics = metrics
            high = middle - 1
        else:
            low = middle + 1
    if selected_threshold is None or selected_metrics is None:
        raise ValueError("NO_THRESHOLD_MEETS_RECALL")
    covered, uncovered, complete, eligible, relevant, images = selected_metrics
    strict = evaluate_at_threshold(
        predictions,
        truth,
        threshold=selected_threshold,
        iou_threshold=iou_threshold,
        minimum_precision=0.0,
        minimum_recall=0.0,
        minimum_complete_scene_rate=0.0,
    )
    return ProposalReport(
        threshold=selected_threshold,
        coverage_recall=covered / (covered + uncovered) if covered + uncovered else 1.0,
        candidate_relevance=relevant / eligible if eligible else 0.0,
        candidates_per_image=eligible / images if images else 0.0,
        complete_scenes=complete,
        images=images,
        covered_truth=covered,
        uncovered_truth=uncovered,
        eligible_candidates=eligible,
        irrelevant_candidates=eligible - relevant,
        strict_iou_recall=strict.recall,
        strict_iou_precision=strict.precision,
    )


def build_proposal_gate_document(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    model_sha256: str,
    truth_sha256: str,
    prediction_sha256: str,
    minimum_recall: float = 0.99,
    maximum_candidates_per_image: float = 20.0,
) -> dict[str, object]:
    for value in (model_sha256, truth_sha256, prediction_sha256):
        if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("INVALID_GATE_SHA256")
    report = select_proposal_threshold(
        predictions, truth, minimum_recall=minimum_recall
    )
    return {
        "schema_version": "marked-point-model-gate-v3",
        "minimum_recall": minimum_recall,
        "maximum_candidates_per_image": maximum_candidates_per_image,
        "proposal_match": "one_to_one_iou_gte_0.30",
        "strict_iou_threshold": 0.50,
        "model_sha256": model_sha256.upper(),
        "truth_sha256": truth_sha256.upper(),
        "prediction_sha256": prediction_sha256.upper(),
        "sealed_test_opened": False,
        "passed_recall": report.coverage_recall >= minimum_recall,
        "passed_burden": report.candidates_per_image <= maximum_candidates_per_image,
        "passed": report.coverage_recall >= minimum_recall
        and report.candidates_per_image <= maximum_candidates_per_image,
        "report": asdict(report),
    }
