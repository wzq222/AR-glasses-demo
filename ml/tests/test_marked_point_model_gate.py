from __future__ import annotations

import pytest

from crrc_vision.marked_point_model_gate import (
    build_proposal_gate_document,
    select_proposal_threshold,
)


def _truth() -> dict[str, object]:
    return {
        "images": [{"id": 1}, {"id": 2}],
        "annotations": [
            {"id": 1, "image_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 2, "bbox": [0, 0, 10, 10]},
        ],
    }


def test_gate_selects_highest_threshold_that_keeps_required_recall() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.4},
        {"image_id": 2, "bbox": [20, 20, 10, 10], "score": 0.8},
    ]

    report = select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)

    assert report.threshold == 0.4
    assert report.recall == 1.0
    assert report.candidates_per_image == 1.5
    assert report.complete_scenes == 2
    assert (report.true_positives, report.false_positives, report.false_negatives) == (2, 1, 0)


def test_gate_rejects_unreachable_recall() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
    ]
    with pytest.raises(ValueError, match="NO_THRESHOLD_MEETS_RECALL"):
        select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)


def test_gate_counts_duplicate_predictions_as_candidate_burden() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    report = select_proposal_threshold(predictions, _truth(), minimum_recall=1.0)
    assert report.threshold == 0.7
    assert report.candidates_per_image == 1.5
    assert report.precision == pytest.approx(2 / 3)


def test_gate_document_binds_model_truth_and_predictions() -> None:
    predictions = [
        {"image_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "bbox": [0, 0, 10, 10], "score": 0.8},
    ]
    document = build_proposal_gate_document(
        predictions,
        _truth(),
        model_sha256="A" * 64,
        truth_sha256="B" * 64,
        prediction_sha256="C" * 64,
        minimum_recall=1.0,
    )
    assert document["schema_version"] == "marked-point-model-gate-v1"
    assert document["model_sha256"] == "A" * 64
    assert document["report"]["recall"] == 1.0
    assert document["sealed_test_opened"] is False
