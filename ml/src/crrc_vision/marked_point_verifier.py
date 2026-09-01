from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from crrc_vision.marked_point_model_gate import is_proposal_match


@dataclass(frozen=True)
class VerifierReport:
    threshold: float
    recall: float
    precision: float
    selected: int
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class PipelineVerifierReport:
    threshold: float
    truth_recall: float
    selected: int
    candidates_per_image: float
    covered_truth: int
    total_truth: int


@dataclass(frozen=True)
class DualPipelineVerifierReport:
    verifier_threshold: float
    proposal_threshold: float
    truth_recall: float
    selected: int
    candidates_per_image: float
    covered_truth: int
    total_truth: int


def verifier_resize_size(input_size: int) -> int:
    """Keep torchvision's 256-to-224 evaluation resize/crop ratio."""

    if input_size <= 0:
        raise ValueError("VERIFIER_INPUT_SIZE_MUST_BE_POSITIVE")
    return round(input_size * 256 / 224)


def combine_verifier_predictions(
    prediction_sets: Sequence[Sequence[Mapping[str, object]]],
    *,
    method: str,
) -> list[dict[str, object]]:
    """Combine candidate-aligned verifier scores without fitted ensemble weights."""
    if len(prediction_sets) < 2:
        raise ValueError("VERIFIER_ENSEMBLE_REQUIRES_MULTIPLE_MODELS")
    if method not in {"mean", "geometric_mean"}:
        raise ValueError(f"INVALID_VERIFIER_ENSEMBLE_METHOD:{method}")
    lengths = {len(rows) for rows in prediction_sets}
    if len(lengths) != 1:
        raise ValueError("VERIFIER_ENSEMBLE_LENGTH_MISMATCH")
    combined: list[dict[str, object]] = []
    for aligned in zip(*prediction_sets, strict=True):
        identities = {
            (
                row.get("prediction_index"),
                row.get("image_id"),
                tuple(row.get("candidate_bbox", [])),
            )
            for row in aligned
        }
        if len(identities) != 1:
            raise ValueError("VERIFIER_ENSEMBLE_IDENTITY_MISMATCH")
        scores = [float(row.get("score", float("nan"))) for row in aligned]
        if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("INVALID_VERIFIER_ENSEMBLE_SCORE")
        if method == "mean":
            score = sum(scores) / len(scores)
        else:
            score = math.prod(scores) ** (1.0 / len(scores))
        combined.append(
            {
                **aligned[0],
                "score": score,
                "seed_scores": scores,
                "ensemble_method": method,
            }
        )
    return combined


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
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection = max(0.0, min(lx + lw, rx + rw) - max(lx, rx)) * max(
        0.0, min(ly + lh, ry + rh) - max(ly, ry)
    )
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def select_verifier_examples(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    score_threshold: float,
    max_positive_per_truth: int = 2,
    max_negative_per_scene: int = 10,
) -> list[dict[str, object]]:
    if max_positive_per_truth <= 0 or max_negative_per_scene <= 0:
        raise ValueError("INVALID_VERIFIER_SAMPLE_LIMIT")
    images = truth.get("images")
    annotations = truth.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("INVALID_VERIFIER_TRUTH")
    image_by_id: dict[object, Mapping[str, object]] = {}
    for image in images:
        if not isinstance(image, Mapping) or image.get("id") in image_by_id:
            raise ValueError("INVALID_VERIFIER_IMAGES")
        image_by_id[image.get("id")] = image
    truth_by_image: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for annotation in annotations:
        if not isinstance(annotation, Mapping) or annotation.get("image_id") not in image_by_id:
            raise ValueError("INVALID_VERIFIER_ANNOTATIONS")
        _bbox(annotation)
        truth_by_image[annotation.get("image_id")].append(annotation)

    positives_by_truth: dict[object, list[dict[str, object]]] = defaultdict(list)
    negatives_by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prediction_index, prediction in enumerate(predictions):
        image_id = prediction.get("image_id")
        image = image_by_id.get(image_id)
        if image is None:
            raise ValueError(f"VERIFIER_UNKNOWN_IMAGE:{image_id}")
        score = float(prediction.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError("INVALID_PREDICTION_SCORE")
        box = _bbox(prediction)
        if score < score_threshold:
            continue
        matching = [
            annotation
            for annotation in truth_by_image[image_id]
            if is_proposal_match(box, _bbox(annotation))
        ]
        common = {
            "prediction_index": prediction_index,
            "image_id": image_id,
            "relative_path": str(image.get("file_name") or ""),
            "scene_group": str(image.get("scene_group") or ""),
            "candidate_bbox": list(box),
            "score": score,
        }
        if matching:
            assigned = max(
                matching,
                key=lambda annotation: (
                    _iou(box, _bbox(annotation)),
                    -int(annotation.get("id", 0)),
                ),
            )
            row = {
                **common,
                "label": "marked_point",
                "truth_id": assigned.get("id"),
                "truth_ids": sorted(
                    (annotation.get("id") for annotation in matching),
                    key=int,
                ),
            }
            for annotation in matching:
                positives_by_truth[annotation.get("id")].append(row)
        else:
            negatives_by_scene[common["scene_group"]].append(
                {
                    **common,
                    "label": "not_marked_point",
                    "truth_id": None,
                    "truth_ids": [],
                }
            )

    positives: list[dict[str, object]] = []
    selected_positive_indices: set[int] = set()
    for annotation in sorted(annotations, key=lambda row: int(row["id"])):
        rows = sorted(
            positives_by_truth.get(annotation.get("id"), []),
            key=lambda row: (-float(row["score"]), int(row["prediction_index"])),
        )
        if not rows:
            raise ValueError(f"VERIFIER_TRUTH_UNCOVERED:{annotation.get('id')}")
        for row in rows[:max_positive_per_truth]:
            prediction_index = int(row["prediction_index"])
            if prediction_index not in selected_positive_indices:
                positives.append(row)
                selected_positive_indices.add(prediction_index)
    negatives: list[dict[str, object]] = []
    for scene_group in sorted(negatives_by_scene):
        rows = sorted(
            negatives_by_scene[scene_group],
            key=lambda row: (-float(row["score"]), int(row["prediction_index"])),
        )
        negatives.extend(rows[:max_negative_per_scene])
    return [*positives, *negatives]


def select_semantic_review_examples(
    truth: Mapping[str, object],
    review: Mapping[str, object],
    *,
    max_negative_per_scene_per_class: int = 10,
) -> list[dict[str, object]]:
    """Build three-class crops only from explicit, scene-isolated review decisions."""
    if max_negative_per_scene_per_class <= 0:
        raise ValueError("INVALID_SEMANTIC_SAMPLE_LIMIT")
    images = truth.get("images")
    annotations = truth.get("annotations")
    review_images = review.get("images")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("INVALID_VERIFIER_TRUTH")
    if not isinstance(review_images, list):
        raise ValueError("INVALID_VERIFIER_REVIEW")
    image_by_id = {
        image.get("id"): image for image in images if isinstance(image, Mapping)
    }
    if len(image_by_id) != len(images):
        raise ValueError("INVALID_VERIFIER_IMAGES")
    review_by_id = {
        image.get("image_id"): image
        for image in review_images
        if isinstance(image, Mapping)
    }
    if set(review_by_id) != set(image_by_id):
        raise ValueError("VERIFIER_REVIEW_IMAGE_MISMATCH")
    positives: list[dict[str, object]] = []
    for annotation in sorted(annotations, key=lambda row: int(row["id"])):
        image_id = annotation.get("image_id")
        image = image_by_id.get(image_id)
        if image is None:
            raise ValueError(f"VERIFIER_UNKNOWN_IMAGE:{image_id}")
        positives.append(
            {
                "image_id": image_id,
                "relative_path": str(image.get("file_name") or ""),
                "scene_group": str(image.get("scene_group") or ""),
                "candidate_bbox": list(_bbox(annotation)),
                "label": "marked_point",
                "truth_id": annotation.get("id"),
                "truth_ids": [annotation.get("id")],
                "review_candidate_id": None,
            }
        )
    negative_buckets: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for image_id, review_image in review_by_id.items():
        image = image_by_id[image_id]
        decisions = review_image.get("candidate_decisions")
        if not isinstance(decisions, list):
            raise ValueError("INVALID_VERIFIER_REVIEW_DECISIONS")
        for decision in decisions:
            if not isinstance(decision, Mapping):
                raise ValueError("INVALID_VERIFIER_REVIEW_DECISION")
            label = decision.get("label")
            if label not in {"lookalike", "unmarked_fastener"}:
                continue
            xyxy = decision.get("xyxy")
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                raise ValueError("INVALID_VERIFIER_REVIEW_BOX")
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            _bbox({"bbox": bbox})
            row = {
                "image_id": image_id,
                "relative_path": str(image.get("file_name") or ""),
                "scene_group": str(image.get("scene_group") or ""),
                "candidate_bbox": bbox,
                "label": label,
                "truth_id": None,
                "truth_ids": [],
                "review_candidate_id": decision.get("candidate_id"),
            }
            negative_buckets[(str(image.get("scene_group") or ""), str(label))].append(row)
    negatives: list[dict[str, object]] = []
    for key in sorted(negative_buckets):
        rows = sorted(
            negative_buckets[key], key=lambda row: str(row["review_candidate_id"])
        )
        negatives.extend(rows[:max_negative_per_scene_per_class])
    return [*positives, *negatives]


def select_verifier_threshold(
    scored: Sequence[Mapping[str, object]], *, minimum_recall: float = 0.95
) -> VerifierReport:
    if not 0.0 <= minimum_recall <= 1.0:
        raise ValueError("INVALID_MINIMUM_RECALL")
    positives = sum(row.get("label") == "marked_point" for row in scored)
    if positives == 0:
        raise ValueError("VERIFIER_POSITIVES_REQUIRED")
    thresholds = sorted(
        {float(row.get("score", float("nan"))) for row in scored}, reverse=True
    )
    if any(not math.isfinite(value) for value in thresholds):
        raise ValueError("INVALID_VERIFIER_SCORE")
    for threshold in thresholds:
        selected = [row for row in scored if float(row["score"]) >= threshold]
        true_positives = sum(row.get("label") == "marked_point" for row in selected)
        false_positives = len(selected) - true_positives
        false_negatives = positives - true_positives
        recall = true_positives / positives
        if recall < minimum_recall:
            continue
        precision = true_positives / len(selected) if selected else 0.0
        return VerifierReport(
            threshold=threshold,
            recall=recall,
            precision=precision,
            selected=len(selected),
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )
    raise ValueError("NO_VERIFIER_THRESHOLD_MEETS_RECALL")


def select_pipeline_threshold(
    scored: Sequence[Mapping[str, object]],
    *,
    image_count: int,
    minimum_truth_recall: float = 0.99,
) -> PipelineVerifierReport:
    """Select the highest verifier threshold that preserves target-level coverage."""
    if image_count <= 0:
        raise ValueError("VERIFIER_IMAGE_COUNT_REQUIRED")
    if not 0.0 <= minimum_truth_recall <= 1.0:
        raise ValueError("INVALID_MINIMUM_TRUTH_RECALL")
    truth_ids = {
        truth_id
        for row in scored
        for truth_id in row.get("truth_ids", [])
    }
    if not truth_ids:
        raise ValueError("VERIFIER_TRUTH_IDS_REQUIRED")
    thresholds = sorted(
        {float(row.get("score", float("nan"))) for row in scored}, reverse=True
    )
    if any(not math.isfinite(value) for value in thresholds):
        raise ValueError("INVALID_VERIFIER_SCORE")
    for threshold in thresholds:
        selected = [row for row in scored if float(row["score"]) >= threshold]
        covered = {
            truth_id
            for row in selected
            for truth_id in row.get("truth_ids", [])
        }
        recall = len(covered) / len(truth_ids)
        if recall >= minimum_truth_recall:
            return PipelineVerifierReport(
                threshold=threshold,
                truth_recall=recall,
                selected=len(selected),
                candidates_per_image=len(selected) / image_count,
                covered_truth=len(covered),
                total_truth=len(truth_ids),
            )
    raise ValueError("NO_PIPELINE_THRESHOLD_MEETS_RECALL")


def select_dual_pipeline_thresholds(
    scored: Sequence[Mapping[str, object]], *, image_count: int
) -> DualPipelineVerifierReport:
    """Minimize burden while retaining every truth through either score channel."""
    if image_count <= 0:
        raise ValueError("VERIFIER_IMAGE_COUNT_REQUIRED")
    if not scored:
        raise ValueError("VERIFIER_SCORED_ROWS_REQUIRED")
    truth_ids = {
        truth_id
        for row in scored
        for truth_id in row.get("truth_ids", [])
    }
    if not truth_ids:
        raise ValueError("VERIFIER_TRUTH_IDS_REQUIRED")
    verifier_scores = [float(row.get("verifier_score", float("nan"))) for row in scored]
    proposal_scores = [float(row.get("proposal_score", float("nan"))) for row in scored]
    if any(not math.isfinite(value) for value in [*verifier_scores, *proposal_scores]):
        raise ValueError("INVALID_DUAL_VERIFIER_SCORE")
    verifier_thresholds = sorted(
        {max(verifier_scores) + 1e-12, *verifier_scores}, reverse=True
    )
    best: tuple[tuple[int, float, float], DualPipelineVerifierReport] | None = None
    for verifier_threshold in verifier_thresholds:
        verifier_selected = [
            row
            for row in scored
            if float(row["verifier_score"]) >= verifier_threshold
        ]
        verifier_covered = {
            truth_id
            for row in verifier_selected
            for truth_id in row.get("truth_ids", [])
        }
        uncovered = truth_ids - verifier_covered
        if uncovered:
            proposal_threshold = min(
                max(
                    float(row["proposal_score"])
                    for row in scored
                    if truth_id in row.get("truth_ids", [])
                )
                for truth_id in uncovered
            )
        else:
            proposal_threshold = max(proposal_scores) + 1e-12
        selected = [
            row
            for row in scored
            if float(row["verifier_score"]) >= verifier_threshold
            or float(row["proposal_score"]) >= proposal_threshold
        ]
        covered = {
            truth_id
            for row in selected
            for truth_id in row.get("truth_ids", [])
        }
        if covered != truth_ids:
            continue
        report = DualPipelineVerifierReport(
            verifier_threshold=verifier_threshold,
            proposal_threshold=proposal_threshold,
            truth_recall=1.0,
            selected=len(selected),
            candidates_per_image=len(selected) / image_count,
            covered_truth=len(covered),
            total_truth=len(truth_ids),
        )
        key = (len(selected), -verifier_threshold, -proposal_threshold)
        if best is None or key < best[0]:
            best = key, report
    if best is None:
        raise ValueError("NO_DUAL_PIPELINE_THRESHOLD_MEETS_RECALL")
    return best[1]


def suppress_overlapping_candidates(
    scored: Sequence[Mapping[str, object]],
    *,
    verifier_threshold: float,
    proposal_threshold: float,
    iou_threshold: float = 0.3,
) -> list[Mapping[str, object]]:
    """Apply deterministic semantic NMS after recall-preserving dual selection."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("INVALID_SEMANTIC_NMS_IOU")
    if verifier_threshold <= 0.0 or proposal_threshold <= 0.0:
        raise ValueError("SEMANTIC_NMS_THRESHOLDS_MUST_BE_POSITIVE")
    eligible = [
        row
        for row in scored
        if float(row.get("score", float("nan"))) >= verifier_threshold
        or float(row.get("proposal_score", float("nan"))) >= proposal_threshold
    ]
    if any(
        not math.isfinite(float(row.get(field, float("nan"))))
        for row in scored
        for field in ("score", "proposal_score")
    ):
        raise ValueError("INVALID_DUAL_VERIFIER_SCORE")
    by_image: dict[object, list[Mapping[str, object]]] = defaultdict(list)
    for row in eligible:
        by_image[row.get("image_id")].append(row)
    selected: list[Mapping[str, object]] = []
    for image_id in sorted(by_image, key=str):
        ranked = sorted(
            by_image[image_id],
            key=lambda row: (
                -max(
                    float(row["score"]) / verifier_threshold,
                    float(row["proposal_score"]) / proposal_threshold,
                ),
                -float(row["score"]),
                -float(row["proposal_score"]),
            ),
        )
        kept: list[Mapping[str, object]] = []
        for row in ranked:
            candidate = _bbox({"bbox": row.get("candidate_bbox")})
            if all(
                _iou(candidate, _bbox({"bbox": other.get("candidate_bbox")}))
                < iou_threshold
                for other in kept
            ):
                kept.append(row)
        selected.extend(kept)
    return selected
