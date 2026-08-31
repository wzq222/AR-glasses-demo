from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from crrc_vision.high_accuracy_gate import evaluate_at_threshold


@dataclass(frozen=True)
class ProposalReport:
    threshold: float
    recall: float
    precision: float
    candidates_per_image: float
    complete_scenes: int
    images: int
    true_positives: int
    false_positives: int
    false_negatives: int


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
    selected = None
    for threshold in thresholds:
        accuracy = evaluate_at_threshold(
            predictions,
            truth,
            threshold=threshold,
            iou_threshold=iou_threshold,
            minimum_precision=0.0,
            minimum_recall=minimum_recall,
            minimum_complete_scene_rate=0.0,
        )
        if accuracy.recall >= minimum_recall:
            selected = accuracy
            break
    if selected is None:
        raise ValueError("NO_THRESHOLD_MEETS_RECALL")
    eligible = sum(float(row["score"]) >= selected.threshold for row in predictions)
    images = selected.total_scenes
    return ProposalReport(
        threshold=selected.threshold,
        recall=selected.recall,
        precision=selected.precision,
        candidates_per_image=eligible / images if images else 0.0,
        complete_scenes=selected.complete_scenes,
        images=images,
        true_positives=selected.true_positives,
        false_positives=selected.false_positives,
        false_negatives=selected.false_negatives,
    )


def build_proposal_gate_document(
    predictions: Sequence[Mapping[str, object]],
    truth: Mapping[str, object],
    *,
    model_sha256: str,
    truth_sha256: str,
    prediction_sha256: str,
    minimum_recall: float = 0.99,
) -> dict[str, object]:
    for value in (model_sha256, truth_sha256, prediction_sha256):
        if len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value):
            raise ValueError("INVALID_GATE_SHA256")
    report = select_proposal_threshold(
        predictions, truth, minimum_recall=minimum_recall
    )
    return {
        "schema_version": "marked-point-model-gate-v1",
        "minimum_recall": minimum_recall,
        "iou_threshold": 0.50,
        "model_sha256": model_sha256.upper(),
        "truth_sha256": truth_sha256.upper(),
        "prediction_sha256": prediction_sha256.upper(),
        "sealed_test_opened": False,
        "report": asdict(report),
    }
